"""
Tests for the optional vision front-end (anthos/vision.py).

Uses a stubbed-out `transformers.CLIPVisionModel` so these tests run offline
and in CI without downloading real CLIP weights. The stub matches CLIP's
real output contract (`outputs.last_hidden_state`), so everything downstream
of it — the projector, the prepend-to-sequence logic in Anthos.forward,
freezing behaviour, and gradient flow — is exercised exactly as it would be
with a real checkpoint.
"""

import sys
import types

import pytest
import torch
import torch.nn as nn

from anthos.main import Anthos, AnthosConfig


class _DummyCLIPOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class _DummyCLIPVisionModel(nn.Module):
    """Stands in for transformers.CLIPVisionModel — same output contract, no weights to download."""

    class _Cfg:
        hidden_size = 32

    def __init__(self):
        super().__init__()
        self.config = self._Cfg()
        self.dummy = nn.Linear(1, 1)  # give it real parameters to freeze/check

    def forward(self, pixel_values):
        B = pixel_values.shape[0]
        n_patches = 5
        return _DummyCLIPOutput(torch.randn(B, n_patches, self.config.hidden_size))

    @classmethod
    def from_pretrained(cls, name):
        return cls()


@pytest.fixture
def stub_transformers(monkeypatch):
    """Install a fake `transformers` module for the duration of a test."""
    fake_module = types.ModuleType("transformers")
    fake_module.CLIPVisionModel = _DummyCLIPVisionModel
    monkeypatch.setitem(sys.modules, "transformers", fake_module)
    yield


def _tiny_cfg() -> AnthosConfig:
    return AnthosConfig(
        vocab_size=1000, dim=64, n_heads=4, n_kv_heads=2,
        max_seq_len=128, max_loop_iters=2, prelude_layers=1, coda_layers=1,
        n_thought_tokens=4, attn_type="gqa", n_experts=4, n_shared_experts=1,
        n_experts_per_tok=2, expert_dim=64,
    )


def test_vision_encoder_attaches(stub_transformers):
    from anthos.vision import VisionConfig
    model = Anthos(_tiny_cfg(), vision_cfg=VisionConfig(encoder_name="fake/clip"))
    assert model.vision_encoder is not None


def test_no_vision_cfg_means_no_encoder():
    model = Anthos(_tiny_cfg())
    assert model.vision_encoder is None


def test_pixel_values_without_vision_cfg_raises():
    model = Anthos(_tiny_cfg())
    ids = torch.randint(0, 1000, (1, 8))
    with pytest.raises(RuntimeError):
        model(ids, pixel_values=torch.randn(1, 3, 32, 32))


def test_image_patches_prepend_to_sequence(stub_transformers):
    from anthos.vision import VisionConfig
    model = Anthos(_tiny_cfg(), vision_cfg=VisionConfig(encoder_name="fake/clip"))

    B, T = 2, 16
    ids          = torch.randint(0, 1000, (B, T))
    pixel_values = torch.randn(B, 3, 64, 64)

    logits = model(ids, n_loops=2, pixel_values=pixel_values)
    # Stub always emits 5 patches -> sequence should extend by exactly 5.
    assert logits.shape == (B, T + 5, 1000)


def test_text_only_path_unaffected_on_vision_enabled_model(stub_transformers):
    from anthos.vision import VisionConfig
    model = Anthos(_tiny_cfg(), vision_cfg=VisionConfig(encoder_name="fake/clip"))
    ids = torch.randint(0, 1000, (2, 16))
    logits = model(ids, n_loops=2)   # no pixel_values
    assert logits.shape == (2, 16, 1000)


def test_clip_backbone_frozen_projector_trainable(stub_transformers):
    from anthos.vision import VisionConfig
    model = Anthos(_tiny_cfg(), vision_cfg=VisionConfig(encoder_name="fake/clip", freeze_encoder=True))
    assert all(not p.requires_grad for p in model.vision_encoder.encoder.parameters())
    assert any(p.requires_grad for p in model.vision_encoder.projector.parameters())


def test_gradients_flow_through_projector_only(stub_transformers):
    from anthos.vision import VisionConfig
    model = Anthos(_tiny_cfg(), vision_cfg=VisionConfig(encoder_name="fake/clip"))

    ids          = torch.randint(0, 1000, (2, 16))
    pixel_values = torch.randn(2, 3, 64, 64)
    logits = model(ids, n_loops=2, pixel_values=pixel_values)
    logits.float().sum().backward()

    proj_grad = model.vision_encoder.projector[0].weight.grad
    assert proj_grad is not None and proj_grad.abs().sum().item() > 0
