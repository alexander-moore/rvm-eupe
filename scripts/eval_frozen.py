#!/usr/bin/env python3
# Copyright (c) 2026 — RVM-EUPE authors.
# Frozen-encoder evaluation entry point.
#
# Loads a checkpoint, freezes the full model, trains task-specific readout
# heads, and reports metrics on all benchmarks.
#
# Usage:
#   python scripts/eval_frozen.py checkpoint=/path/to/checkpoint.pt \
#          eval=davis,kinetics400,ssv2,scannet_depth
#
# Individual benchmarks can be toggled via eval.tasks config list.

import json
import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def load_checkpoint(model, checkpoint_path: str, device: torch.device) -> int:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        log.warning(f"Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        log.warning(f"Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    return ckpt.get("step", 0)


@hydra.main(config_path="../configs", config_name="default", version_base="1.3")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Build model ----
    from scripts.train import build_model
    model = build_model(cfg).to(device)

    checkpoint_path = cfg.get("checkpoint", None)
    if checkpoint_path:
        step = load_checkpoint(model, checkpoint_path, device)
        log.info(f"Loaded checkpoint from step {step}: {checkpoint_path}")
    else:
        log.warning("No checkpoint provided — evaluating with random weights")

    model.eval()
    model.freeze_all()

    state_dim_map = {"vitt": 192, "vits": 384, "vitb": 768}
    state_dim = state_dim_map.get(cfg.model.arch, 768)
    num_heads  = cfg.model.gru_heads

    tasks = list(cfg.eval.get("tasks", ["davis", "kinetics400", "ssv2", "scannet_depth",
                                         "jhmdb", "perception_test"]))
    results = {}

    # ---- DAVIS 2017 ----
    if "davis" in tasks:
        davis_root = cfg.eval.get("davis_root", "/data/davis")
        if Path(davis_root).exists():
            log.info("Evaluating DAVIS 2017...")
            from rvm_eupe.eval.davis import evaluate_davis
            metrics = evaluate_davis(model, davis_root, device=device)
            results["DAVIS"] = metrics
            log.info(f"DAVIS: J={metrics['J_mean']:.1f}  F={metrics['F_mean']:.1f}  "
                     f"J&F={metrics['JF_mean']:.1f}")
        else:
            log.warning(f"DAVIS root not found: {davis_root} — skipping")

    # ---- Kinetics-400 ----
    if "kinetics400" in tasks:
        k400_root = cfg.eval.get("kinetics400_root", "/data/kinetics400")
        if Path(k400_root).exists():
            log.info("Evaluating Kinetics-400...")
            from rvm_eupe.eval.action_cls import train_and_eval_action_cls
            train_loader, val_loader = _build_action_loaders(k400_root, cfg, num_classes=400)
            metrics = train_and_eval_action_cls(
                model, train_loader, val_loader, num_classes=400,
                state_dim=state_dim, num_heads=num_heads, device=device
            )
            results["K400"] = metrics
            log.info(f"K400: top1={metrics['top1_acc']*100:.1f}%  top5={metrics['top5_acc']*100:.1f}%")
        else:
            log.warning(f"Kinetics-400 root not found — skipping")

    # ---- Something-Something-v2 ----
    if "ssv2" in tasks:
        ssv2_root = cfg.eval.get("ssv2_root", "/data/ssv2")
        if Path(ssv2_root).exists():
            log.info("Evaluating SSv2...")
            from rvm_eupe.eval.action_cls import train_and_eval_action_cls
            train_loader, val_loader = _build_action_loaders(ssv2_root, cfg, num_classes=174)
            metrics = train_and_eval_action_cls(
                model, train_loader, val_loader, num_classes=174,
                state_dim=state_dim, num_heads=num_heads, device=device
            )
            results["SSv2"] = metrics
            log.info(f"SSv2: top1={metrics['top1_acc']*100:.1f}%  top5={metrics['top5_acc']*100:.1f}%")
        else:
            log.warning(f"SSv2 root not found — skipping")

    # ---- ScanNet Depth ----
    if "scannet_depth" in tasks:
        scannet_root = cfg.eval.get("scannet_root", "/data/scannet")
        if Path(scannet_root).exists():
            log.info("Evaluating ScanNet depth...")
            from rvm_eupe.eval.depth import train_and_eval_depth
            train_loader, val_loader = _build_depth_loaders(scannet_root, cfg)
            metrics = train_and_eval_depth(
                model, train_loader, val_loader, state_dim=state_dim,
                num_heads=num_heads, device=device
            )
            results["ScanNet"] = metrics
            log.info(f"ScanNet AbsRel={metrics['AbsRel']:.3f}  RMSE={metrics['RMSE']:.3f}")
        else:
            log.warning(f"ScanNet root not found — skipping")

    # ---- JHMDB Keypoints ----
    if "jhmdb" in tasks:
        jhmdb_root = cfg.eval.get("jhmdb_root", "/data/jhmdb")
        if Path(jhmdb_root).exists():
            log.info("Evaluating JHMDB keypoints...")
            from rvm_eupe.eval.keypoint import train_and_eval_keypoint
            train_loader, val_loader = _build_keypoint_loaders(jhmdb_root, cfg)
            metrics = train_and_eval_keypoint(
                model, train_loader, val_loader, state_dim=state_dim,
                num_heads=num_heads, device=device
            )
            results["JHMDB"] = metrics
            log.info(f"JHMDB PCK@0.1={metrics['PCK@0.1']*100:.1f}%")
        else:
            log.warning(f"JHMDB root not found — skipping")

    # ---- Perception Test ----
    if "perception_test" in tasks:
        pt_root = cfg.eval.get("perception_test_root", "/data/perception_test")
        if Path(pt_root).exists():
            log.info("Evaluating Perception Test...")
            from rvm_eupe.eval.tracking import train_and_eval_perception_test
            train_loader, val_loader = _build_tracking_loaders(pt_root, cfg)
            metrics = train_and_eval_perception_test(
                model, train_loader, val_loader, state_dim=state_dim,
                num_heads=num_heads, device=device
            )
            results["PerceptionTest"] = metrics
            log.info(f"Perception Test AJ={metrics['AJ']*100:.1f}%")
        else:
            log.warning(f"Perception Test root not found — skipping")

    # ---- Summary ----
    out_path = Path(cfg.output_dir) / "eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {out_path}")
    log.info("=" * 60)
    for task, m in results.items():
        log.info(f"  {task}: {m}")
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# DataLoader stubs — replace with real loaders for each dataset
# ---------------------------------------------------------------------------

def _build_action_loaders(root, cfg, num_classes):
    """Return (train_loader, val_loader) for action classification."""
    from rvm_eupe.data.video_dataset import VideoClipDataset, collate_fn
    from rvm_eupe.data.transforms import build_readout_transforms, build_eval_transforms
    from torch.utils.data import DataLoader

    def _make_loader(split, transform):
        index = str(Path(root) / f"{split}.json")
        if not Path(index).exists():
            return None
        ds = VideoClipDataset(index, transform=transform)
        return DataLoader(ds, batch_size=32, num_workers=4, collate_fn=collate_fn)

    # Readout training uses color jitter (paper: brightness/contrast/saturation/hue)
    return (_make_loader("train", build_readout_transforms()),
            _make_loader("val",   build_eval_transforms()))


def _build_depth_loaders(root, cfg):
    from torch.utils.data import DataLoader
    # Return None loaders — users should implement dataset-specific loaders
    return None, None


def _build_keypoint_loaders(root, cfg):
    return None, None


def _build_tracking_loaders(root, cfg):
    return None, None


if __name__ == "__main__":
    main()
