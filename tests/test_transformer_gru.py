"""
Unit tests for TransformerGRU and CrossAttentionBlock.

Run: pytest tests/test_transformer_gru.py -v
"""

import torch
import pytest

from rvm_eupe.models.transformer_gru import TransformerGRU, TransformerBlock


@pytest.fixture
def gru_vitb():
    return TransformerGRU(embed_dim=768, num_heads=12, num_blocks=1)


def test_output_shapes(gru_vitb):
    B, N, D = 2, 196, 768
    e_t = torch.randn(B, N, D)
    s_t, h_t = gru_vitb(e_t, s_prev=None)
    assert s_t.shape == (B, N, D)
    assert h_t.shape == (B, N, D)


def test_none_state_gives_valid_output(gru_vitb):
    e_t = torch.randn(2, 196, 768)
    s_t, _ = gru_vitb(e_t, s_prev=None)
    assert not torch.isnan(s_t).any()


def test_state_propagation(gru_vitb):
    """Running two steps with propagated state should change the output."""
    e1 = torch.randn(2, 196, 768)
    e2 = torch.randn(2, 196, 768)
    s1, _ = gru_vitb(e1, s_prev=None)
    s2a, _ = gru_vitb(e2, s_prev=s1)
    s2b, _ = gru_vitb(e2, s_prev=None)
    # With and without previous state should differ
    assert not torch.allclose(s2a, s2b)


def test_gate_init_near_identity():
    """
    With very small weight init (std=0.01), the sigmoid gates should be close to 0.5.
    u_t ≈ 0.5 means the state update blends equally between old and new.
    """
    gru = TransformerGRU(embed_dim=64, num_heads=4, num_blocks=1)
    B, N, D = 2, 16, 64
    e_t = torch.randn(B, N, D)
    s_prev = torch.zeros(B, N, D)

    # Compute update gate manually
    u_t = torch.sigmoid(gru.W_u_e(e_t) + gru.W_u_s(s_prev))
    # With std=0.01, gate logits should be small → sigmoid ≈ 0.5
    assert u_t.mean().item() == pytest.approx(0.5, abs=0.15)


def test_different_embed_dims():
    for embed_dim, num_heads in [(192, 3), (384, 6), (768, 12)]:
        gru = TransformerGRU(embed_dim=embed_dim, num_heads=num_heads)
        e_t = torch.randn(1, 49, embed_dim)
        s_t, _ = gru(e_t)
        assert s_t.shape == (1, 49, embed_dim)


def test_multi_block():
    gru = TransformerGRU(embed_dim=128, num_heads=4, num_blocks=3)
    e_t = torch.randn(2, 64, 128)
    s_t, h_t = gru(e_t)
    assert s_t.shape == (2, 64, 128)


def test_sequential_rollout_shapes():
    """Simulate a 4-frame source rollout."""
    gru = TransformerGRU(embed_dim=384, num_heads=6)
    B, N, D = 2, 196, 384
    s = None
    for _ in range(4):
        e_t = torch.randn(B, N, D)
        s, _ = gru(e_t, s_prev=s)
        s = s.detach()  # truncated BPTT
    assert s.shape == (B, N, D)
    assert not torch.isnan(s).any()
