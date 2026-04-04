# Copyright (c) 2026 — RVM-EUPE authors.
# Attention-based readout heads for frozen-encoder evaluation.
# Matching the evaluation protocol from RVM (arxiv 2512.13684):
#   - Frozen RecurrentVideoMAE model (encoder + recurrent)
#   - Per-task readout head trained on top of frozen hidden state s_t
#   - Cross-attention: task queries attend to s_t tokens
#
# Head variants:
#   AttentiveReadoutHead  — generic cross-attention head with learned or spatial queries
#   LearnedQueryFactory   — for action classification (1 global query)
#   SpatialQueryFactory   — for depth / tracking (Fourier-embedded spatial queries)

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Fourier position embedding (for spatial / tracking queries)
# ---------------------------------------------------------------------------

def fourier_embed(coords: Tensor, num_bands: int = 64, max_freq: float = 64.0) -> Tensor:
    """
    Embed 2D coordinates (normalised to [-1, 1]) with Fourier features.

    Args:
        coords:    [..., 2]  (y, x) coordinates in [-1, 1]
        num_bands: Number of frequency bands per dimension.
        max_freq:  Maximum frequency.

    Returns:
        embed: [..., 4*num_bands + 2]
    """
    freqs = torch.linspace(1.0, max_freq, num_bands, device=coords.device)
    # [..., 2, num_bands]
    angles = coords.unsqueeze(-1) * freqs * math.pi
    sin_embed = torch.sin(angles)
    cos_embed = torch.cos(angles)
    # [..., 4*num_bands + 2]
    return torch.cat([coords, sin_embed.flatten(-2), cos_embed.flatten(-2)], dim=-1)


# ---------------------------------------------------------------------------
# Query factories
# ---------------------------------------------------------------------------

class LearnedQueryFactory(nn.Module):
    """
    Produces a fixed set of learned query vectors.
    Used for action classification (1 global query per clip).
    """

    def __init__(self, num_queries: int, dim: int) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.zeros(1, num_queries, dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

    def forward(self, batch_size: int) -> Tensor:
        return self.queries.expand(batch_size, -1, -1)


class SpatialQueryFactory(nn.Module):
    """
    Produces query vectors from spatial grid positions + Fourier embedding.
    Used for depth estimation, segmentation, tracking.
    """

    def __init__(
        self,
        grid_h: int,
        grid_w: int,
        dim: int,
        fourier_bands: int = 64,
    ) -> None:
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w

        # Build normalised grid coords
        y = torch.linspace(-1, 1, grid_h)
        x = torch.linspace(-1, 1, grid_w)
        gy, gx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack([gy.flatten(), gx.flatten()], dim=-1)  # [N, 2]

        embed_dim = 4 * fourier_bands + 2
        self.proj = nn.Linear(embed_dim, dim)
        self.register_buffer("coords", coords)

    def forward(self, batch_size: int) -> Tensor:
        embed = fourier_embed(self.coords, num_bands=64)  # [N, embed_dim]
        queries = self.proj(embed).unsqueeze(0).expand(batch_size, -1, -1)
        return queries


# ---------------------------------------------------------------------------
# AttentiveReadoutHead
# ---------------------------------------------------------------------------

class AttentiveReadoutHead(nn.Module):
    """
    Cross-attention readout head.

    Task-specific learned/spatial queries attend to the frozen hidden state s_t
    from the TransformerGRU.  Then a small MLP maps to the output space.

    Args:
        state_dim:    Dimension of s_t (encoder embed_dim, e.g. 768).
        query_factory: Module returning query tensors given batch_size.
        num_queries:  Number of query tokens (determines output size).
        num_heads:    Attention heads.
        num_blocks:   Depth of cross-attention (default 4).
        output_dim:   Final output dim per query.  If None, returns [B, Q, state_dim].
    """

    def __init__(
        self,
        state_dim: int,
        query_factory: nn.Module,
        num_heads: int,
        num_blocks: int = 4,
        output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.query_factory = query_factory

        self.blocks = nn.ModuleList([
            _ReadoutBlock(state_dim, num_heads) for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(state_dim)
        self.head = nn.Linear(state_dim, output_dim) if output_dim is not None else nn.Identity()

    def forward(self, state: Tensor) -> Tensor:
        """
        Args:
            state: [B, N, D]  frozen hidden state from TransformerGRU

        Returns:
            out: [B, num_queries, output_dim]
        """
        B = state.shape[0]
        queries = self.query_factory(B)  # [B, Q, D]

        for blk in self.blocks:
            queries = blk(queries, state)

        queries = self.norm(queries)
        return self.head(queries)


class _ReadoutBlock(nn.Module):
    """cross-attn (queries→state) → FFN, pre-norm."""

    def __init__(self, dim: int, num_heads: int, ffn_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm_q   = nn.LayerNorm(dim)
        self.norm_ctx = nn.LayerNorm(dim)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj  = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, 2 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm_ffn = nn.LayerNorm(dim)
        hidden = int(dim * ffn_ratio)
        self.ffn = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, queries: Tensor, context: Tensor) -> Tensor:
        B, Q, D = queries.shape
        Nc = context.shape[1]
        q = self.q_proj(self.norm_q(queries)).reshape(B, Q, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(self.norm_ctx(context)).reshape(B, Nc, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, Q, D)
        queries = queries + self.out_proj(attn)
        queries = queries + self.ffn(self.norm_ffn(queries))
        return queries


# ---------------------------------------------------------------------------
# Task-specific head factories
# ---------------------------------------------------------------------------

def build_action_cls_head(state_dim: int, num_classes: int, num_heads: int = 12) -> AttentiveReadoutHead:
    """Single learned query → linear classifier for K400/K700/SSv2."""
    factory = LearnedQueryFactory(num_queries=1, dim=state_dim)
    return AttentiveReadoutHead(state_dim, factory, num_heads, num_blocks=4, output_dim=num_classes)


def build_depth_head(state_dim: int, grid_h: int = 8, grid_w: int = 8,
                     num_heads: int = 12) -> AttentiveReadoutHead:
    """
    128 spatial queries (8×8 grid) → scalar depth predictions.
    Used for ScanNet AbsRel evaluation.
    """
    factory = SpatialQueryFactory(grid_h, grid_w, dim=state_dim)
    return AttentiveReadoutHead(state_dim, factory, num_heads, num_blocks=4, output_dim=1)


def build_keypoint_head(state_dim: int, num_joints: int = 15,
                        num_heads: int = 12) -> AttentiveReadoutHead:
    """
    Learned queries per joint → 2D coordinate prediction (JHMDB PCK@0.1).
    """
    factory = LearnedQueryFactory(num_queries=num_joints, dim=state_dim)
    return AttentiveReadoutHead(state_dim, factory, num_heads, num_blocks=4, output_dim=2)
