"""
Phase 0 smoke test: EUPEEncoderWrapper shape correctness.

Run: pytest tests/test_encoder_wrapper.py -v
"""

import torch
import pytest

from eupe.hub.backbones import eupe_vitb16, eupe_vits16, eupe_vitt16
from rvm_eupe.models.encoder_wrapper import EUPEEncoderWrapper


@pytest.mark.parametrize("factory,embed_dim,expected_n", [
    (eupe_vitt16, 192, 196),
    (eupe_vits16, 384, 196),
    (eupe_vitb16, 768, 196),
])
def test_output_shape_standard(factory, embed_dim, expected_n):
    """224×224 input → N=196 tokens (14×14 grid)."""
    backbone = factory(pretrained=False)
    wrapper = EUPEEncoderWrapper(backbone, use_pre_recurrent_norm=True)
    wrapper.eval()

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        tokens = wrapper(x)

    assert tokens.shape == (2, expected_n, embed_dim), (
        f"Expected (2, {expected_n}, {embed_dim}), got {tokens.shape}"
    )


def test_output_shape_non_standard_resolution():
    """Non-multiple-of-16 input is padded to nearest multiple → token count changes."""
    backbone = eupe_vitb16(pretrained=False)
    wrapper = EUPEEncoderWrapper(backbone, use_pre_recurrent_norm=False)
    wrapper.eval()

    # 220×220 → padded to 224×224 → N=196
    x = torch.randn(1, 3, 220, 220)
    with torch.no_grad():
        tokens = wrapper(x)
    assert tokens.shape[1] == 196


def test_projection_layer():
    """Optional linear projection changes out_dim."""
    backbone = eupe_vitb16(pretrained=False)
    wrapper = EUPEEncoderWrapper(backbone, out_dim=512)
    wrapper.eval()

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        tokens = wrapper(x)

    assert tokens.shape == (2, 196, 512)


def test_freeze_backbone():
    """Freezing backbone marks all backbone params as no-grad."""
    backbone = eupe_vitb16(pretrained=False)
    wrapper = EUPEEncoderWrapper(backbone, freeze_backbone=True)

    for p in wrapper.backbone.parameters():
        assert not p.requires_grad

    wrapper.unfreeze()
    for p in wrapper.backbone.parameters():
        assert p.requires_grad


def test_no_storage_token_leakage():
    """Asserts that N exactly equals H_pad//16 * W_pad//16 (no extra storage tokens)."""
    backbone = eupe_vitb16(pretrained=False)
    # eupe_vitb16 has n_storage_tokens=4; wrapper must strip them
    wrapper = EUPEEncoderWrapper(backbone)
    wrapper.eval()

    for h, w in [(224, 224), (256, 256), (192, 320)]:
        n_expected = wrapper.num_tokens(h, w)
        x = torch.randn(1, 3, h, w)
        with torch.no_grad():
            tokens = wrapper(x)
        assert tokens.shape[1] == n_expected, (
            f"For ({h},{w}): expected N={n_expected}, got {tokens.shape[1]}"
        )
