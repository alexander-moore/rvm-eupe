# Copyright (c) 2026 — RVM-EUPE authors.
# Top-level model: Recurrent Video MAE with EUPE backbone.
#
# Forward pass orchestration:
#   1. Encode each source frame with EUPE encoder → e_t tokens
#   2. Step TransformerGRU over source frames → hidden state s_T
#   3. Apply random mask to target frame tokens
#   4. Decode masked target using s_T as memory → pixel predictions
#   5. L2 reconstruction loss on masked pixels

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .encoder_wrapper import EUPEEncoderWrapper
from .transformer_gru import TransformerGRU
from .decoder import MAEDecoder


def _patchify(imgs: Tensor, patch_size: int) -> Tensor:
    """
    Decompose images into non-overlapping patch pixel vectors.

    Args:
        imgs:       [B, 3, H, W]
        patch_size: p

    Returns:
        patches:    [B, (H//p)*(W//p), p*p*3]
    """
    B, C, H, W = imgs.shape
    assert H % patch_size == 0 and W % patch_size == 0
    h = H // patch_size
    w = W // patch_size
    x = imgs.reshape(B, C, h, patch_size, w, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)  # [B, h, w, C, p, p]
    x = x.reshape(B, h * w, C * patch_size * patch_size)
    return x


def _random_mask(B: int, N: int, mask_ratio: float, device: torch.device) -> Tensor:
    """
    Returns a boolean mask of shape [B, N] where True means masked.
    Each sample is masked independently.
    """
    num_masked = int(math.ceil(N * mask_ratio))
    noise = torch.rand(B, N, device=device)
    ids_shuffle = torch.argsort(noise, dim=1)
    mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    mask.scatter_(1, ids_shuffle[:, :num_masked], True)
    return mask


class RecurrentVideoMAE(nn.Module):
    """
    Recurrent Video Masked Autoencoder with EUPE backbone.

    Args:
        encoder:           EUPEEncoderWrapper instance.
        recurrent:         TransformerGRU instance.
        decoder:           MAEDecoder instance.
        mask_ratio:        Fraction of target tokens to mask (default 0.95).
        num_source_frames: Number of consecutive source frames (default 4).
        bptt_truncate:     If True, detach hidden state after processing all
                           source frames (truncated BPTT).  Set False for full
                           BPTT (memory-heavy, ablation only).
    """

    def __init__(
        self,
        encoder: EUPEEncoderWrapper,
        recurrent: TransformerGRU,
        decoder: MAEDecoder,
        mask_ratio: float = 0.95,
        num_source_frames: int = 4,
        bptt_truncate: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.recurrent = recurrent
        self.decoder = decoder
        self.mask_ratio = mask_ratio
        self.num_source_frames = num_source_frames
        self.bptt_truncate = bptt_truncate

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        source_frames: List[Tensor],  # list of num_source_frames tensors [B, 3, H, W]
        target_frame: Tensor,         # [B, 3, H, W]
    ) -> dict:
        """
        Returns dict with keys: 'loss', 'pred', 'mask'.
        """
        assert len(source_frames) == self.num_source_frames, (
            f"Expected {self.num_source_frames} source frames, got {len(source_frames)}"
        )

        B, C, H, W = target_frame.shape
        patch_size = self.encoder.patch_size

        # Pad target to match what the encoder will pad source frames to
        from eupe.eval.depth.models.embed import CenterPadding
        pad = CenterPadding(multiple=patch_size)
        target_padded = pad(target_frame)
        _, _, H_pad, W_pad = target_padded.shape
        grid_h = H_pad // patch_size
        grid_w = W_pad // patch_size
        N = grid_h * grid_w

        # ---- Step 1: Encode source frames + recurrent aggregation ----
        s: Optional[Tensor] = None
        for frame in source_frames:
            e_t = self.encoder(frame)              # [B, N, D]
            s, _ = self.recurrent(e_t, s)
            if self.bptt_truncate:
                s = s.detach()

        # ---- Step 2: Encode target frame (for visible tokens) ----
        target_tokens = self.encoder(target_padded)  # [B, N, D]

        # ---- Step 3: Random mask on target tokens ----
        mask = _random_mask(B, N, self.mask_ratio, target_frame.device)  # [B, N] bool

        # ---- Step 4: Decode ----
        pred, mask = self.decoder(
            target_tokens=target_tokens,
            mask=mask,
            memory=s,
            grid_hw=(grid_h, grid_w),
        )

        # ---- Step 5: Patchify target for loss computation ----
        target_patches = _patchify(target_padded, patch_size)  # [B, N, patch_pixels]

        loss = MAEDecoder.reconstruction_loss(pred, target_patches, mask)

        return {"loss": loss, "pred": pred, "mask": mask}

    # ------------------------------------------------------------------
    # Convenience: freeze/unfreeze for staged training
    # ------------------------------------------------------------------

    def freeze_encoder(self) -> None:
        self.encoder.freeze()

    def unfreeze_encoder(self) -> None:
        self.encoder.unfreeze()

    def freeze_all(self) -> None:
        """Freeze entire model for frozen-encoder evaluation."""
        for p in self.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        arch: str = "vitb",
        weights: str = "LVD1689M",
        pretrained: bool = True,
        mask_ratio: float = 0.95,
        num_source_frames: int = 4,
        bptt_truncate: bool = True,
        decoder_dim: int = 512,
        decoder_heads: int = 16,
        decoder_blocks: int = 8,
        gru_heads: Optional[int] = None,
        gru_blocks: int = 1,
    ) -> "RecurrentVideoMAE":
        """
        Convenience factory.  arch in {"vitt", "vits", "vitb"}.
        """
        from eupe.hub.backbones import eupe_vitt16, eupe_vits16, eupe_vitb16, Weights

        _arch_map = {
            "vitt": (eupe_vitt16, 192, 3),
            "vits": (eupe_vits16, 384, 6),
            "vitb": (eupe_vitb16, 768, 12),
        }
        if arch not in _arch_map:
            raise ValueError(f"Unknown arch '{arch}'. Choose from {list(_arch_map)}")

        factory_fn, embed_dim, default_heads = _arch_map[arch]
        if gru_heads is None:
            gru_heads = default_heads

        w = Weights[weights]
        backbone = factory_fn(pretrained=pretrained, weights=w)

        encoder = EUPEEncoderWrapper(backbone, use_pre_recurrent_norm=True)
        recurrent = TransformerGRU(embed_dim, gru_heads, gru_blocks)
        decoder = MAEDecoder(encoder_dim=embed_dim, decoder_dim=decoder_dim,
                             num_heads=decoder_heads, num_blocks=decoder_blocks)

        return cls(encoder, recurrent, decoder, mask_ratio, num_source_frames, bptt_truncate)
