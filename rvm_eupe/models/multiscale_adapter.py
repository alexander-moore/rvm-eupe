# Copyright (c) 2026 — RVM-EUPE authors.
# Ablation study: multi-depth feature fusion for EUPE ViT backbone.
#
# IMPORTANT architectural note:
#   EUPE ViT has a UNIFORM stride of 16 throughout all layers — all
#   intermediate layers output spatial maps at stride-16, not a true
#   spatial pyramid.  "Multi-scale" here means multi-depth (multi-semantic-
#   level) feature aggregation at the same spatial resolution.
#
#   True multi-stride pyramid (stride 4/8/16/32) is only available via the
#   EUPE ConvNeXt backbone — this is a separate ablation variant (A4).
#
# Ablation variants:
#   A1 (primary):  last layer only, no adapter — see EUPEEncoderWrapper
#   A2 (FPN):      4 layers [2,5,8,11], 1×1 conv per layer + sum fusion
#   A3 (concat):   4 layers [2,5,8,11], channel concat + single linear proj
#
# Layer indices for ViT-B/S/T (all have 12 blocks) with FOUR_EVEN_INTERVALS:
#   [2, 5, 8, 11]

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from eupe.eval.depth.models.embed import CenterPadding


# ---------------------------------------------------------------------------
# FPN-style adapter (A2)
# ---------------------------------------------------------------------------

class FPNAdapter(nn.Module):
    """
    Fuses 4 intermediate ViT feature maps via lightweight FPN:
      - 1×1 conv per layer to project to out_dim
      - Sum all projected features
      - Flatten spatial dims to [B, N, out_dim]

    All intermediate maps are at stride-16 (same spatial resolution),
    so no upsampling/downsampling is needed — this is a pure channel fusion.

    Args:
        in_dim:     Per-layer feature dim (e.g. 768 for EUPE ViT-B).
        out_dim:    Output feature dim (should match recurrent embed_dim).
        num_layers: Number of intermediate layers to fuse (default 4).
    """

    def __init__(self, in_dim: int, out_dim: int, num_layers: int = 4) -> None:
        super().__init__()
        # 1×1 conv == nn.Linear applied channelwise; use nn.Conv2d for spatial maps
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)
            for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(out_dim)
        self.num_layers = num_layers

        for conv in self.lateral_convs:
            nn.init.trunc_normal_(conv.weight, std=0.02)

    def forward(self, layer_features: List[Tensor]) -> Tensor:
        """
        Args:
            layer_features: list of [B, in_dim, H//16, W//16] spatial maps

        Returns:
            tokens: [B, N, out_dim]  where N = H//16 * W//16
        """
        assert len(layer_features) == self.num_layers
        fused = None
        for feat, conv in zip(layer_features, self.lateral_convs):
            projected = conv(feat)   # [B, out_dim, Hf, Wf]
            fused = projected if fused is None else fused + projected

        B, C, Hf, Wf = fused.shape
        tokens = fused.flatten(2).transpose(1, 2)   # [B, N, out_dim]
        tokens = self.output_norm(tokens)
        return tokens


# ---------------------------------------------------------------------------
# Concat + projection adapter (A3)
# ---------------------------------------------------------------------------

class ConcatAdapter(nn.Module):
    """
    Fuses 4 intermediate ViT feature maps via channel concatenation + linear:
      - Flatten each map to [B, N, in_dim]
      - Concatenate along channel dim → [B, N, num_layers * in_dim]
      - Project to out_dim

    Args:
        in_dim:     Per-layer feature dim.
        out_dim:    Output dim (should match recurrent embed_dim).
        num_layers: Number of intermediate layers.
    """

    def __init__(self, in_dim: int, out_dim: int, num_layers: int = 4) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim * num_layers, out_dim, bias=True)
        self.output_norm = nn.LayerNorm(out_dim)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, layer_features: List[Tensor]) -> Tensor:
        """
        Args:
            layer_features: list of [B, in_dim, Hf, Wf] spatial maps

        Returns:
            tokens: [B, N, out_dim]
        """
        flat = [f.flatten(2).transpose(1, 2) for f in layer_features]  # list of [B, N, in_dim]
        cat  = torch.cat(flat, dim=-1)  # [B, N, num_layers * in_dim]
        tokens = self.output_norm(self.proj(cat))
        return tokens


# ---------------------------------------------------------------------------
# MultiScaleEUPEEncoder  (wraps backbone + adapter for A2 / A3)
# ---------------------------------------------------------------------------

class MultiScaleEUPEEncoder(nn.Module):
    """
    Wraps EUPE DinoVisionTransformer with multi-depth feature extraction.
    Drop-in replacement for EUPEEncoderWrapper in ablation experiments.

    Args:
        backbone:    DinoVisionTransformer instance.
        adapter:     FPNAdapter or ConcatAdapter.
        layer_indices: Which intermediate layer indices to extract (e.g. [2,5,8,11]).
        freeze_backbone: If True, backbone params are frozen.
    """

    def __init__(
        self,
        backbone: nn.Module,
        adapter: nn.Module,
        layer_indices: Sequence[int] = (2, 5, 8, 11),
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter
        self.layer_indices = list(layer_indices)
        self.patch_size: int = backbone.patch_size
        self.embed_dim: int = adapter.proj.out_features if hasattr(adapter, "proj") else adapter.lateral_convs[0].out_channels
        self.center_pad = CenterPadding(multiple=self.patch_size)

        if freeze_backbone:
            self.freeze()

    def freeze(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def unfreeze(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, 3, H, W]

        Returns:
            tokens: [B, N, D_out]
        """
        x = self.center_pad(x)
        _, _, H_pad, W_pad = x.shape

        # get_intermediate_layers returns a tuple of tensors [B, N, D] (flat, pre-reshape)
        # with reshape=True they become [B, D, Hf, Wf] spatial maps
        layer_features = self.backbone.get_intermediate_layers(
            x,
            n=self.layer_indices,
            reshape=True,     # → [B, D, H//patch_size, W//patch_size]
            return_class_token=False,
            norm=True,
        )
        # layer_features is a tuple of [B, D, Hf, Wf]
        tokens = self.adapter(list(layer_features))
        return tokens

    def num_tokens(self, h: int, w: int) -> int:
        h_pad = math.ceil(h / self.patch_size) * self.patch_size
        w_pad = math.ceil(w / self.patch_size) * self.patch_size
        return (h_pad // self.patch_size) * (w_pad // self.patch_size)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def build_fpn_encoder(
    backbone: nn.Module,
    out_dim: Optional[int] = None,
    layer_indices: Sequence[int] = (2, 5, 8, 11),
    freeze_backbone: bool = False,
) -> MultiScaleEUPEEncoder:
    """Build A2: FPN multi-depth adapter."""
    in_dim = backbone.embed_dim
    if out_dim is None:
        out_dim = in_dim
    adapter = FPNAdapter(in_dim=in_dim, out_dim=out_dim, num_layers=len(layer_indices))
    return MultiScaleEUPEEncoder(backbone, adapter, layer_indices, freeze_backbone)


def build_concat_encoder(
    backbone: nn.Module,
    out_dim: Optional[int] = None,
    layer_indices: Sequence[int] = (2, 5, 8, 11),
    freeze_backbone: bool = False,
) -> MultiScaleEUPEEncoder:
    """Build A3: channel concat + linear projection adapter."""
    in_dim = backbone.embed_dim
    if out_dim is None:
        out_dim = in_dim
    adapter = ConcatAdapter(in_dim=in_dim, out_dim=out_dim, num_layers=len(layer_indices))
    return MultiScaleEUPEEncoder(backbone, adapter, layer_indices, freeze_backbone)
