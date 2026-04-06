# Copyright (c) 2026 — RVM-EUPE authors.
# PyTorch port of the Transformer-GRU recurrent aggregator from
# "Recurrent Video Masked Autoencoders" (arxiv 2512.13684).
#
# Gating equations (per paper):
#   u_t = σ( W_e^u · ê_t  +  W_s^u · s_{t-1} )          [update gate]
#   r_t = σ( W_e^r · ê_t  +  W_s^r · s_{t-1} )          [reset  gate]
#   ĥ_t = Tx( q=ê_t,  kv=r_t ⊙ s_{t-1} )               [candidate via cross-attn block]
#   s_t = (1 − u_t) ⊙ s_{t-1}  +  u_t ⊙ ĥ_t            [state update]
#
# Each TransformerBlock follows Listing 2 of the paper (4 blocks total):
#   self-attention  →  cross-attention  →  FFN  (all pre-norm)

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class PreNorm(nn.Module):
    """Pre-normalisation wrapper."""
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.fn(self.norm(x), **kwargs)


class FFN(nn.Module):
    def __init__(self, dim: int, ffn_ratio: float = 4.0) -> None:
        super().__init__()
        hidden = int(dim * ffn_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Attention primitives
# ---------------------------------------------------------------------------

class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, D)
        return self.proj(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.kv_proj = nn.Linear(dim, 2 * dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        B, N, D = query.shape
        q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(context).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(x)


# ---------------------------------------------------------------------------
# Transformer block used inside the GRU (cross → FFN → self, all pre-norm)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    One block of the recurrent aggregator Tx (Listing 2 in RVM paper):
      1. self-attention                                     [pre-norm]  — intra-frame token mixing
      2. cross-attention (query=e_t, context=gated_state)  [pre-norm]  — attend to recurrent memory
      3. FFN                                                [pre-norm]
    """

    def __init__(self, dim: int, num_heads: int, ffn_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = SelfAttention(dim, num_heads)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_ctx = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads)

        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = FFN(dim, ffn_ratio)

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        # 1. self-attention (intra-frame)
        x = x + self.self_attn(self.norm_self(x))
        # 2. cross-attention to gated recurrent state
        x = x + self.cross_attn(self.norm_q(x), self.norm_ctx(context))
        # 3. FFN
        x = x + self.ffn(self.norm_ffn(x))
        return x


# ---------------------------------------------------------------------------
# TransformerGRU
# ---------------------------------------------------------------------------

class TransformerGRU(nn.Module):
    """
    Recurrent aggregator from RVM (arxiv 2512.13684), ported to PyTorch.

    Processes one video frame at a time, maintaining a hidden state s_t of
    the same shape as the encoder output: [B, N, D].

    When s_prev is None (first frame), the hidden state is initialised to
    zeros and the update gate is effectively forced to 1 (write everything).

    Args:
        embed_dim:    Must match the encoder output dimension D.
        num_heads:    Attention heads inside the cross-attention block.
        num_blocks:   How many TransformerBlock passes to use for computing ĥ_t.
                      Paper uses 4 for all model sizes (S/B/L/H).
        ffn_ratio:    MLP expansion ratio inside the block.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_blocks: int = 4,
        ffn_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        D = embed_dim

        # Gate projections — operate token-wise (nn.Linear broadcasts over N)
        # Initialised with small std so initial gate output ≈ 0.5 (sigmoid of ≈0)
        self.W_u_e = nn.Linear(D, D, bias=True)
        self.W_u_s = nn.Linear(D, D, bias=False)
        self.W_r_e = nn.Linear(D, D, bias=True)
        self.W_r_s = nn.Linear(D, D, bias=False)

        for layer in (self.W_u_e, self.W_u_s, self.W_r_e, self.W_r_s):
            nn.init.trunc_normal_(layer.weight, std=0.01)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        # Cross-attention blocks for computing candidate ĥ_t
        self.blocks = nn.ModuleList([
            TransformerBlock(D, num_heads, ffn_ratio)
            for _ in range(num_blocks)
        ])

        self.embed_dim = embed_dim

    # ------------------------------------------------------------------

    def _init_state(self, e_t: Tensor) -> Tensor:
        """Return a zero hidden state matching e_t's shape and device."""
        return torch.zeros_like(e_t)

    def forward(
        self,
        e_t: Tensor,
        s_prev: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            e_t:    [B, N, D]  current frame encoder tokens
            s_prev: [B, N, D]  previous hidden state (None → zeros)

        Returns:
            s_t:    [B, N, D]  updated hidden state
            h_t:    [B, N, D]  candidate state (ĥ_t, before gating)
        """
        if s_prev is None:
            s_prev = self._init_state(e_t)

        # Gating
        u_t = torch.sigmoid(self.W_u_e(e_t) + self.W_u_s(s_prev))   # update gate
        r_t = torch.sigmoid(self.W_r_e(e_t) + self.W_r_s(s_prev))   # reset gate

        # Gated context for cross-attention
        gated_state = r_t * s_prev   # [B, N, D]

        # Compute candidate via one or more Transformer blocks
        h_t = e_t
        for blk in self.blocks:
            h_t = blk(h_t, context=gated_state)

        # State update
        s_t = (1.0 - u_t) * s_prev + u_t * h_t

        return s_t, h_t
