# Copyright (c) 2026 — RVM-EUPE authors.
# Wraps the EUPE DinoVisionTransformer to produce flat [B, N, D] patch tokens
# compatible with the TransformerGRU recurrent aggregator.

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from eupe.eval.depth.models.embed import CenterPadding


class EUPEEncoderWrapper(nn.Module):
    """
    Wraps an EUPE DinoVisionTransformer to produce flat patch token sequences.

    Input:  [B, 3, H, W]  (arbitrary H, W — padded internally to multiples of patch_size)
    Output: [B, N, D_out]  where N = (H_padded // patch_size) * (W_padded // patch_size)

    The EUPE forward_features() API already strips the CLS token and the
    n_storage_tokens=4 register tokens, returning only the patch tokens via
    the "x_norm_patchtokens" key.  No manual stripping is needed here.

    Args:
        backbone:            A DinoVisionTransformer instance (e.g. from eupe_vitb16()).
        out_dim:             If given and != backbone.embed_dim, adds a linear projection.
        freeze_backbone:     If True, backbone parameters require no grad (used during
                             frozen-encoder evaluation).
        use_pre_recurrent_norm: Insert a LayerNorm on the output tokens before they
                             enter the TransformerGRU gates.  Recommended when training
                             from scratch to stabilise gate dynamics.
    """

    def __init__(
        self,
        backbone: nn.Module,
        out_dim: Optional[int] = None,
        freeze_backbone: bool = False,
        use_pre_recurrent_norm: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.patch_size: int = backbone.patch_size
        self.embed_dim: int = backbone.embed_dim

        # Resolution alignment — pads H and W to multiples of patch_size
        self.center_pad = CenterPadding(multiple=self.patch_size)

        # Optional projection when encoder dim != requested dim
        proj_dim = out_dim if out_dim is not None else self.embed_dim
        if proj_dim != self.embed_dim:
            self.proj = nn.Linear(self.embed_dim, proj_dim, bias=False)
            nn.init.trunc_normal_(self.proj.weight, std=0.02)
        else:
            self.proj = None
        self.out_dim = proj_dim

        # LayerNorm applied after the backbone (and optional projection) to
        # normalise token distributions before the recurrent gate projections.
        self.pre_recurrent_norm = nn.LayerNorm(self.out_dim) if use_pre_recurrent_norm else None

        if freeze_backbone:
            self.freeze()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def freeze(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def unfreeze(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(True)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, 3, H, W]

        Returns:
            tokens: [B, N, D_out]  where N = (H_pad//patch_size)*(W_pad//patch_size)
        """
        B, C, H, W = x.shape

        # Pad to multiples of patch_size (symmetric, no information added)
        x = self.center_pad(x)
        _, _, H_pad, W_pad = x.shape
        N_expected = (H_pad // self.patch_size) * (W_pad // self.patch_size)

        # EUPE forward — returns dict; x_norm_patchtokens has shape [B, N, D]
        out = self.backbone.forward_features(x)
        tokens: Tensor = out["x_norm_patchtokens"]

        assert tokens.shape == (B, N_expected, self.embed_dim), (
            f"Unexpected token shape {tokens.shape}, expected "
            f"({B}, {N_expected}, {self.embed_dim}). "
            f"Check that n_storage_tokens are being stripped correctly."
        )

        if self.proj is not None:
            tokens = self.proj(tokens)

        if self.pre_recurrent_norm is not None:
            tokens = self.pre_recurrent_norm(tokens)

        return tokens

    # ------------------------------------------------------------------
    # Convenience: number of tokens for a given input resolution
    # ------------------------------------------------------------------

    def num_tokens(self, h: int, w: int) -> int:
        h_pad = math.ceil(h / self.patch_size) * self.patch_size
        w_pad = math.ceil(w / self.patch_size) * self.patch_size
        return (h_pad // self.patch_size) * (w_pad // self.patch_size)
