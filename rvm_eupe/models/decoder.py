# Copyright (c) 2026 — RVM-EUPE authors.
# MAE Decoder — PyTorch port from RVM (arxiv 2512.13684).
#
# Architecture (per paper):
#   8 blocks, decoder_dim=512, 16 heads
#   Per block: cross-attention (target → memory)  →  FFN  →  self-attention
#   All pre-norm.
#
# The decoder receives:
#   - target_tokens:  [B, N, encoder_dim]  — unmasked + mask tokens, positionally embedded
#   - memory:         [B, N_src, encoder_dim]  — hidden state s_t from TransformerGRU
#
# A linear projection maps encoder_dim → decoder_dim at the start.
# Output: per-token pixel predictions  [B, N, patch_size^2 * 3]

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Positional embedding (2D sinusoidal, fixed)
# ---------------------------------------------------------------------------

def _build_2d_sincos_pos_embed(
    embed_dim: int,
    grid_h: int,
    grid_w: int,
    temperature: float = 10000.0,
) -> Tensor:
    """Returns [grid_h*grid_w, embed_dim] fixed 2D sinusoidal embeddings."""
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sin-cos"
    half = embed_dim // 2
    quarter = embed_dim // 4

    y_coords = torch.arange(grid_h, dtype=torch.float32)
    x_coords = torch.arange(grid_w, dtype=torch.float32)
    gy, gx = torch.meshgrid(y_coords, x_coords, indexing="ij")  # [H, W]

    omega = torch.arange(quarter, dtype=torch.float32) / quarter
    omega = 1.0 / (temperature ** omega)  # [quarter]

    sin_y = torch.sin(gy.flatten()[:, None] * omega[None, :])   # [N, quarter]
    cos_y = torch.cos(gy.flatten()[:, None] * omega[None, :])
    sin_x = torch.sin(gx.flatten()[:, None] * omega[None, :])
    cos_x = torch.cos(gx.flatten()[:, None] * omega[None, :])

    return torch.cat([sin_y, cos_y, sin_x, cos_x], dim=-1)  # [N, embed_dim]


# ---------------------------------------------------------------------------
# Attention / block primitives (mirrors transformer_gru.py style)
# ---------------------------------------------------------------------------

class _SelfAttn(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, D))


class _CrossAttn(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, 2 * dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        B, Nq, D = query.shape
        Nkv = context.shape[1]
        q = self.q_proj(query).reshape(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(context).reshape(B, Nkv, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.out_proj(x.transpose(1, 2).reshape(B, Nq, D))


class _FFN(nn.Module):
    def __init__(self, dim: int, ffn_ratio: float = 4.0) -> None:
        super().__init__()
        hidden = int(dim * ffn_ratio)
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MAEDecoderBlock(nn.Module):
    """
    One decoder block: cross-attn (target→memory) → FFN → self-attn  (all pre-norm).
    """

    def __init__(self, dim: int, num_heads: int, ffn_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm_q   = nn.LayerNorm(dim)
        self.norm_ctx = nn.LayerNorm(dim)
        self.cross_attn = _CrossAttn(dim, num_heads)

        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = _FFN(dim, ffn_ratio)

        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = _SelfAttn(dim, num_heads)

    def forward(self, x: Tensor, memory: Tensor) -> Tensor:
        x = x + self.cross_attn(self.norm_q(x), self.norm_ctx(memory))
        x = x + self.ffn(self.norm_ffn(x))
        x = x + self.self_attn(self.norm_self(x))
        return x


# ---------------------------------------------------------------------------
# MAEDecoder
# ---------------------------------------------------------------------------

class MAEDecoder(nn.Module):
    """
    8-block decoder that reconstructs masked target-frame patches from the
    TransformerGRU hidden state s_t.

    Args:
        encoder_dim:  Encoder / recurrent state feature dimension (e.g. 768).
        decoder_dim:  Internal decoder width (default 512, per RVM paper).
        num_heads:    Attention heads (default 16, per RVM paper).
        num_blocks:   Number of decoder blocks (default 8).
        patch_size:   Spatial patch side length in pixels (default 16).
        ffn_ratio:    FFN expansion ratio (default 4.0).
    """

    def __init__(
        self,
        encoder_dim: int = 768,
        decoder_dim: int = 512,
        num_heads: int = 16,
        num_blocks: int = 8,
        patch_size: int = 16,
        ffn_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.decoder_dim = decoder_dim
        self.patch_size = patch_size
        self.patch_pixels = patch_size * patch_size * 3  # RGB

        # Project encoder tokens + hidden state into decoder space
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim, bias=True)
        self.mem_to_dec = nn.Linear(encoder_dim, decoder_dim, bias=True)

        # Learnable mask token (broadcast over masked positions)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Decoder blocks
        self.blocks = nn.ModuleList([
            MAEDecoderBlock(decoder_dim, num_heads, ffn_ratio)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(decoder_dim)

        # Pixel prediction head
        self.pred_head = nn.Linear(decoder_dim, self.patch_pixels, bias=True)
        nn.init.trunc_normal_(self.pred_head.weight, std=0.02)

        # Positional embedding (populated lazily for the first seen grid size)
        self._pos_embed_cache: dict[Tuple[int, int], Tensor] = {}

    # ------------------------------------------------------------------

    def _get_pos_embed(self, grid_h: int, grid_w: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        key = (grid_h, grid_w)
        if key not in self._pos_embed_cache:
            pe = _build_2d_sincos_pos_embed(self.decoder_dim, grid_h, grid_w)
            self._pos_embed_cache[key] = pe
        return self._pos_embed_cache[(grid_h, grid_w)].to(device=device, dtype=dtype)

    # ------------------------------------------------------------------

    def forward(
        self,
        target_tokens: Tensor,  # [B, N, encoder_dim] — ALL positions (masked + unmasked)
        mask: Tensor,           # [B, N] bool, True = masked (to be predicted)
        memory: Tensor,         # [B, N, encoder_dim] — hidden state s_t
        grid_hw: Tuple[int, int],  # (grid_h, grid_w) for positional embedding
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            target_tokens: encoder output for the target frame (unmasked positions
                           have real features; masked positions will be replaced with
                           the learnable mask token).
            mask:          Boolean mask, True where a token is masked.
            memory:        TransformerGRU hidden state s_t, used as cross-attn keys/values.
            grid_hw:       (H//patch_size, W//patch_size) — determines positional embedding.

        Returns:
            pred:  [B, N_masked, patch_pixels]  pixel predictions for masked positions.
            loss:  scalar L2 reconstruction loss.
        """
        B, N, _ = target_tokens.shape
        grid_h, grid_w = grid_hw
        device, dtype = target_tokens.device, target_tokens.dtype

        # Positional embedding for all N positions
        pos_embed = self._get_pos_embed(grid_h, grid_w, device, dtype)  # [N, D_dec]

        # Project encoder tokens into decoder space
        x = self.enc_to_dec(target_tokens)  # [B, N, D_dec]

        # Replace masked positions with mask token + positional embedding
        mask_tokens = self.mask_token.expand(B, N, -1)  # [B, N, D_dec]
        x = torch.where(mask.unsqueeze(-1), mask_tokens + pos_embed, x + pos_embed)

        # Project memory into decoder space
        mem = self.mem_to_dec(memory)  # [B, N, D_dec]

        # Decode
        for blk in self.blocks:
            x = blk(x, mem)
        x = self.norm(x)

        # Predict pixels at masked positions only
        pred = self.pred_head(x)  # [B, N, patch_pixels]

        return pred, mask

    # ------------------------------------------------------------------

    @staticmethod
    def reconstruction_loss(
        pred: Tensor,   # [B, N, patch_pixels]
        target: Tensor, # [B, N, patch_pixels]
        mask: Tensor,   # [B, N] bool — True = masked
    ) -> Tensor:
        """Mean L2 loss over masked positions (no patch normalisation)."""
        loss = F.mse_loss(pred[mask], target[mask])
        return loss
