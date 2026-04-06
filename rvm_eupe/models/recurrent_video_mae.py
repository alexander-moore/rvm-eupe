# Copyright (c) 2026 — RVM-EUPE authors.
# Top-level model: Recurrent Video MAE with EUPE backbone.
#
# Forward pass orchestration (per RVM paper Section 3):
#   1. Encode each of 4 source frames → step TransformerGRU → hidden state s_T
#   2. Encode each of 4 target frames (independently sampled at Δt ∈ [4,48])
#   3. Apply 95% random mask to each target frame
#   4. Decode each masked target using shared s_T as cross-attention memory
#   5. L2 loss over ALL patch positions (no patch normalisation), averaged over 4 targets

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
        source_frames: List[Tensor],   # list of num_source_frames tensors [B, 3, H, W]
        target_frames: List[Tensor],   # list of 4 target tensors [B, 3, H, W]
    ) -> dict:
        """
        Per RVM paper: 4 source frames build the recurrent state s_T; 4 target
        frames (each independently masked at 95%) are decoded from the same s_T.
        Loss is L2 over ALL patch positions, averaged across the 4 targets.

        Returns dict with keys: 'loss', 'preds' (list), 'masks' (list).
        """
        assert len(source_frames) == self.num_source_frames, (
            f"Expected {self.num_source_frames} source frames, got {len(source_frames)}"
        )

        patch_size = self.encoder.patch_size
        from eupe.eval.depth.models.embed import CenterPadding
        pad = CenterPadding(multiple=patch_size)

        # Determine grid from first source frame
        B = source_frames[0].shape[0]
        sample_padded = pad(source_frames[0])
        _, _, H_pad, W_pad = sample_padded.shape
        grid_h = H_pad // patch_size
        grid_w = W_pad // patch_size
        N = grid_h * grid_w

        # ---- Step 1: Encode source frames + recurrent aggregation ----
        # No within-sequence detach — backprop flows through all 4 source steps.
        # bptt_truncate detaches s_T after the full sequence, preventing cross-
        # batch recurrence (which doesn't exist here since s_0 = zeros always).
        s: Optional[Tensor] = None
        for frame in source_frames:
            e_t = self.encoder(frame)        # [B, N, D]
            s, _ = self.recurrent(e_t, s)

        if self.bptt_truncate:
            s = s.detach()

        # ---- Steps 2–4: Encode, mask, decode each target frame ----
        total_loss = torch.tensor(0.0, device=source_frames[0].device)
        preds, masks = [], []

        for tgt in target_frames:
            tgt_padded = pad(tgt)

            # Encode unmasked target tokens
            target_tokens = self.encoder(tgt_padded)            # [B, N, D]

            # Independent 95% random mask per target
            mask = _random_mask(B, N, self.mask_ratio, tgt.device)  # [B, N] bool

            # Decode — all N positions predicted
            pred, mask = self.decoder(
                target_tokens=target_tokens,
                mask=mask,
                memory=s,
                grid_hw=(grid_h, grid_w),
            )

            # L2 over all patch positions (paper: "entire reconstructed image pixels")
            target_patches = _patchify(tgt_padded, patch_size)  # [B, N, patch_pixels]
            total_loss = total_loss + MAEDecoder.reconstruction_loss(pred, target_patches)

            preds.append(pred)
            masks.append(mask)

        loss = total_loss / len(target_frames)
        return {"loss": loss, "preds": preds, "masks": masks}

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
        decoder_heads: int = 8,
        decoder_blocks: int = 4,
        gru_heads: Optional[int] = None,
        gru_blocks: int = 4,
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

        # weights can be a Weights enum name ("LVD1689M") or a local file path
        try:
            w = Weights[weights]
        except KeyError:
            w = weights  # treat as a direct path / URL string
        backbone = factory_fn(pretrained=pretrained, weights=w)

        encoder = EUPEEncoderWrapper(backbone, use_pre_recurrent_norm=True)
        recurrent = TransformerGRU(embed_dim, gru_heads, gru_blocks)
        decoder = MAEDecoder(encoder_dim=embed_dim, decoder_dim=decoder_dim,
                             num_heads=decoder_heads, num_blocks=decoder_blocks)

        return cls(encoder, recurrent, decoder, mask_ratio, num_source_frames, bptt_truncate)
