"""
Unit tests for MAEDecoder and RecurrentVideoMAE end-to-end dummy forward.

Run: pytest tests/test_decoder.py -v
"""

import torch
import pytest

from rvm_eupe.models.decoder import MAEDecoder, _build_2d_sincos_pos_embed
from rvm_eupe.models.transformer_gru import TransformerGRU
from rvm_eupe.models.recurrent_video_mae import RecurrentVideoMAE, _random_mask, _patchify


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_decoder(encoder_dim=768, decoder_dim=512, num_heads=16, num_blocks=4):
    """Smaller decoder for fast unit tests (4 blocks instead of 8)."""
    return MAEDecoder(encoder_dim=encoder_dim, decoder_dim=decoder_dim,
                      num_heads=num_heads, num_blocks=num_blocks, patch_size=16)


# ---------------------------------------------------------------------------
# MAEDecoder tests
# ---------------------------------------------------------------------------

def test_pos_embed_shape():
    pe = _build_2d_sincos_pos_embed(512, grid_h=14, grid_w=14)
    assert pe.shape == (196, 512)


def test_decoder_output_shape():
    decoder = make_decoder()
    B, N, D_enc = 2, 196, 768
    target_tokens = torch.randn(B, N, D_enc)
    memory = torch.randn(B, N, D_enc)
    mask = torch.zeros(B, N, dtype=torch.bool)
    mask[:, :186] = True  # ~95% masked

    pred, out_mask = decoder(target_tokens, mask, memory, grid_hw=(14, 14))
    assert pred.shape == (B, N, 16 * 16 * 3)


def test_decoder_loss_is_scalar():
    decoder = make_decoder()
    B, N = 2, 196
    pred = torch.randn(B, N, 768)
    target = torch.randn(B, N, 768)
    # Loss is now over ALL positions (no mask arg) — paper: "entire image pixels"
    loss = MAEDecoder.reconstruction_loss(pred, target)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_decoder_no_nan():
    decoder = make_decoder()
    B, N, D = 2, 196, 768
    target_tokens = torch.randn(B, N, D)
    memory = torch.randn(B, N, D)
    mask = torch.zeros(B, N, dtype=torch.bool)
    mask[:, :186] = True
    pred, _ = decoder(target_tokens, mask, memory, grid_hw=(14, 14))
    assert not torch.isnan(pred).any()


# ---------------------------------------------------------------------------
# _random_mask tests
# ---------------------------------------------------------------------------

def test_random_mask_ratio():
    B, N = 4, 196
    mask = _random_mask(B, N, mask_ratio=0.95, device=torch.device("cpu"))
    assert mask.shape == (B, N)
    assert mask.dtype == torch.bool
    # Each row should have ~95% masked (ceiling → exactly ceil(196*0.95)=187)
    for b in range(B):
        assert mask[b].sum().item() == 187


# ---------------------------------------------------------------------------
# RecurrentVideoMAE end-to-end dummy forward (no pretrained weights)
# ---------------------------------------------------------------------------

def test_recurrent_video_mae_forward():
    """Full dummy forward pass with EUPE ViT-T (smallest/fastest)."""
    model = RecurrentVideoMAE.build(
        arch="vitt",
        pretrained=False,
        mask_ratio=0.95,
        num_source_frames=4,
        bptt_truncate=True,
        decoder_dim=128,   # shrink for test speed
        decoder_heads=4,
        decoder_blocks=2,
        gru_heads=3,
        gru_blocks=1,
    )
    model.eval()

    B = 2
    source_frames = [torch.randn(B, 3, 224, 224) for _ in range(4)]
    target_frames = [torch.randn(B, 3, 224, 224) for _ in range(4)]

    with torch.no_grad():
        out = model(source_frames, target_frames)

    assert "loss" in out
    assert out["loss"].ndim == 0
    assert not torch.isnan(out["loss"])
    assert len(out["masks"]) == 4
    assert out["masks"][0].shape == (B, 196)


def test_freeze_encoder():
    model = RecurrentVideoMAE.build(arch="vitt", pretrained=False,
                                    decoder_dim=128, decoder_heads=4, decoder_blocks=2)
    model.freeze_encoder()
    for p in model.encoder.backbone.parameters():
        assert not p.requires_grad

    model.unfreeze_encoder()
    for p in model.encoder.backbone.parameters():
        assert p.requires_grad
