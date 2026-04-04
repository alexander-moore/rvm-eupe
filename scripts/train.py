#!/usr/bin/env python3
# Copyright (c) 2026 — RVM-EUPE authors.
# Main training entry point.
#
# Usage:
#   python scripts/train.py model=rvm_eupe_vitb train=large_scale data=mixed
#   python scripts/train.py model=rvm_eupe_vits  # smaller scale
#   torchrun --nproc_per_node=8 scripts/train.py  # multi-GPU with FSDP

import logging
import os
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


def setup_distributed() -> None:
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def build_model(cfg: DictConfig):
    from rvm_eupe.models.recurrent_video_mae import RecurrentVideoMAE

    arch = cfg.model.arch
    if arch == "vitb_timm":
        # Baseline: standard ViT-B from timm
        import timm
        from rvm_eupe.models.encoder_wrapper import EUPEEncoderWrapper
        from rvm_eupe.models.transformer_gru import TransformerGRU
        from rvm_eupe.models.decoder import MAEDecoder

        class TimmViTWrapper(torch.nn.Module):
            """Thin wrapper to make timm ViT-B compatible with EUPEEncoderWrapper interface."""
            def __init__(self):
                super().__init__()
                self.backbone = timm.create_model("vit_base_patch16_224", pretrained=cfg.model.pretrained,
                                                   num_classes=0)
                self.embed_dim = 768
                self.patch_size = 16

            def forward_features(self, x):
                features = self.backbone.forward_features(x)  # [B, N+1, D]
                # timm ViT prepends CLS token — strip it
                return {"x_norm_patchtokens": features[:, 1:]}

        wrapper_backbone = TimmViTWrapper()
        encoder = EUPEEncoderWrapper(wrapper_backbone, use_pre_recurrent_norm=True)
        recurrent = TransformerGRU(768, cfg.model.gru_heads, cfg.model.gru_blocks)
        decoder = MAEDecoder(encoder_dim=768, decoder_dim=cfg.model.decoder_dim,
                             num_heads=cfg.model.decoder_heads, num_blocks=cfg.model.decoder_blocks)
        model = RecurrentVideoMAE(encoder, recurrent, decoder,
                                  mask_ratio=cfg.model.mask_ratio,
                                  num_source_frames=cfg.model.num_source_frames,
                                  bptt_truncate=cfg.model.bptt_truncate)
    else:
        model = RecurrentVideoMAE.build(
            arch=cfg.model.arch,
            weights=cfg.model.get("weights", "LVD1689M"),
            pretrained=cfg.model.pretrained,
            mask_ratio=cfg.model.mask_ratio,
            num_source_frames=cfg.model.num_source_frames,
            bptt_truncate=cfg.model.bptt_truncate,
            decoder_dim=cfg.model.decoder_dim,
            decoder_heads=cfg.model.decoder_heads,
            decoder_blocks=cfg.model.decoder_blocks,
            gru_heads=cfg.model.gru_heads,
            gru_blocks=cfg.model.gru_blocks,
        )
    return model


def build_dataloader(cfg: DictConfig, transform) -> DataLoader:
    from rvm_eupe.data.video_dataset import VideoClipDataset, collate_fn
    from rvm_eupe.data.mixed_dataset import MixedVideoDataset

    datasets = {}
    for name, ds_cfg in cfg.data.datasets.items():
        index_path = ds_cfg.index
        if not Path(index_path).exists():
            log.warning(f"Index file not found for {name}: {index_path} — skipping")
            continue
        datasets[name] = VideoClipDataset(
            index_path=index_path,
            transform=transform,
            frame_format=ds_cfg.get("frame_format", "video"),
        )

    if not datasets:
        raise RuntimeError("No valid dataset index files found. Update configs/data/mixed.yaml.")

    weights = dict(cfg.data.weights)
    mixed = MixedVideoDataset(datasets, weights)
    sampler = mixed.make_sampler()

    return DataLoader(
        mixed,
        batch_size=cfg.train.batch_size_per_gpu,
        sampler=sampler,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )


@hydra.main(config_path="../configs", config_name="default", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_distributed()
    rank = dist.get_rank() if dist.is_initialized() else 0
    is_main = rank == 0

    if is_main:
        log.info(OmegaConf.to_yaml(cfg))
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Model ----
    model = build_model(cfg)
    model = model.to(device)

    # Stage 1: freeze encoder backbone
    freeze_steps = cfg.train.backbone.freeze_steps
    if freeze_steps > 0:
        model.freeze_encoder()
        log.info(f"Encoder frozen for first {freeze_steps} steps (Stage 1)")

    # FSDP wrapping (multi-GPU)
    if dist.is_initialized():
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            device_id=torch.cuda.current_device(),
        )

    # ---- Optimizer ----
    from rvm_eupe.optim.schedulers import get_layerwise_param_groups, cosine_warmup_schedule

    param_groups = get_layerwise_param_groups(
        model, base_lr=cfg.train.optimizer.lr,
        weight_decay=cfg.train.optimizer.weight_decay,
        backbone_decay=cfg.train.backbone.decay,
    )
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=tuple(cfg.train.optimizer.betas),
    )
    scheduler = cosine_warmup_schedule(
        optimizer,
        warmup_steps=cfg.train.scheduler.warmup_steps,
        total_steps=cfg.train.scheduler.total_steps,
    )

    # ---- Data ----
    from rvm_eupe.data.transforms import build_train_transforms
    transform = build_train_transforms(crop_size=cfg.data.crop_size)
    loader = build_dataloader(cfg, transform)

    # ---- Mixed precision scaler ----
    use_bf16 = cfg.train.precision == "bf16"
    scaler = torch.cuda.amp.GradScaler(enabled=not use_bf16)
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    # ---- Training loop ----
    accum = cfg.train.accumulate_grad_batches
    global_step = 0

    while global_step < cfg.train.scheduler.total_steps:
        for batch in loader:
            if global_step >= cfg.train.scheduler.total_steps:
                break

            # Unfreeze encoder after Stage 1
            if global_step == freeze_steps and freeze_steps > 0:
                model.unfreeze_encoder()
                log.info(f"Step {global_step}: encoder unfrozen (Stage 2)")

            source_frames = [
                batch["source_frames"][:, t].to(device)
                for t in range(cfg.model.num_source_frames)
            ]
            target_frame = batch["target_frame"].to(device)

            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = model(source_frames, target_frame)
                loss = out["loss"] / accum

            if use_bf16:
                loss.backward()
            else:
                scaler.scale(loss).backward()

            if (global_step + 1) % accum == 0:
                if not use_bf16:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.gradient.clip_norm)
                if use_bf16:
                    optimizer.step()
                else:
                    scaler.step(optimizer)
                    scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            global_step += 1

            if is_main and global_step % 100 == 0:
                log.info(f"step={global_step}  loss={out['loss'].item():.4f}  "
                         f"lr={scheduler.get_last_lr()[0]:.2e}")

            # Checkpoint
            if is_main and global_step % cfg.train.checkpoint.save_every == 0:
                ckpt_path = Path(cfg.output_dir) / f"checkpoint_{global_step:08d}.pt"
                torch.save({
                    "step": global_step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }, ckpt_path)
                log.info(f"Saved checkpoint: {ckpt_path}")

    if is_main:
        log.info("Training complete.")


if __name__ == "__main__":
    main()
