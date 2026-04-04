"""
Tests for MultiScaleEUPEEncoder (ablation A2/A3).

Run: pytest tests/test_multiscale_adapter.py -v
"""

import torch
import pytest

from eupe.hub.backbones import eupe_vitb16, eupe_vits16
from rvm_eupe.models.multiscale_adapter import (
    FPNAdapter,
    ConcatAdapter,
    MultiScaleEUPEEncoder,
    build_fpn_encoder,
    build_concat_encoder,
)


LAYER_INDICES = (2, 5, 8, 11)  # FOUR_EVEN_INTERVALS for 12-block ViT


@pytest.fixture(scope="module")
def backbone_b():
    return eupe_vitb16(pretrained=False)


def test_fpn_adapter_shapes(backbone_b):
    """FPNAdapter: 4 spatial maps → [B, N, out_dim]."""
    adapter = FPNAdapter(in_dim=768, out_dim=768, num_layers=4)
    # Simulate 4 spatial maps at stride-16 for 224x224 input: Hf=Wf=14
    feats = [torch.randn(2, 768, 14, 14) for _ in range(4)]
    tokens = adapter(feats)
    assert tokens.shape == (2, 196, 768)


def test_concat_adapter_shapes(backbone_b):
    """ConcatAdapter: 4 maps → concat → project → [B, N, out_dim]."""
    adapter = ConcatAdapter(in_dim=768, out_dim=768, num_layers=4)
    feats = [torch.randn(2, 768, 14, 14) for _ in range(4)]
    tokens = adapter(feats)
    assert tokens.shape == (2, 196, 768)


def test_multiscale_encoder_fpn_forward(backbone_b):
    """Full encoder forward: image → FPN tokens [B, N, D]."""
    encoder = build_fpn_encoder(backbone_b, out_dim=768, layer_indices=LAYER_INDICES)
    encoder.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        tokens = encoder(x)
    assert tokens.shape == (2, 196, 768)
    assert not torch.isnan(tokens).any()


def test_multiscale_encoder_concat_forward(backbone_b):
    """Full encoder forward: image → concat tokens [B, N, D]."""
    encoder = build_concat_encoder(backbone_b, out_dim=768, layer_indices=LAYER_INDICES)
    encoder.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        tokens = encoder(x)
    assert tokens.shape == (2, 196, 768)
    assert not torch.isnan(tokens).any()


def test_multiscale_encoder_freeze(backbone_b):
    encoder = build_fpn_encoder(backbone_b, layer_indices=LAYER_INDICES, freeze_backbone=True)
    for p in encoder.backbone.parameters():
        assert not p.requires_grad
    # Adapter params should still be trainable
    adapter_params = list(encoder.adapter.parameters())
    assert any(p.requires_grad for p in adapter_params)


def test_multiscale_drop_in_for_gru():
    """MultiScaleEUPEEncoder output is compatible with TransformerGRU input."""
    from rvm_eupe.models.transformer_gru import TransformerGRU

    backbone = eupe_vits16(pretrained=False)  # 384d
    encoder = build_fpn_encoder(backbone, out_dim=384, layer_indices=LAYER_INDICES)
    gru = TransformerGRU(embed_dim=384, num_heads=6)

    encoder.eval()
    gru.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        tokens = encoder(x)      # [2, 196, 384]
        s, _ = gru(tokens)       # [2, 196, 384]

    assert s.shape == (2, 196, 384)
    assert not torch.isnan(s).any()


def test_fpn_output_differs_from_last_layer_only():
    """FPN multi-depth output should differ from using only the last layer."""
    backbone = eupe_vitb16(pretrained=False)
    encoder_ms = build_fpn_encoder(backbone, out_dim=768, layer_indices=LAYER_INDICES)

    from rvm_eupe.models.encoder_wrapper import EUPEEncoderWrapper
    encoder_last = EUPEEncoderWrapper(backbone, use_pre_recurrent_norm=False)

    encoder_ms.eval()
    encoder_last.eval()

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        tokens_ms   = encoder_ms(x)
        tokens_last = encoder_last(x)

    # Outputs must have same shape but different values
    assert tokens_ms.shape == tokens_last.shape
    assert not torch.allclose(tokens_ms, tokens_last)
