"""eval_leak_gap: stepwise (generation-style) loss vs teacher-forced loss."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from anthos.main import Anthos
from eval_leak_gap import leak_gap
from test_thought_causality import _cfg


def _examples():
    torch.manual_seed(3)
    return [(torch.randint(5, 100, (14,)), 5), (torch.randint(5, 100, (11,)), 4)]


def _model(causal):
    torch.manual_seed(0)
    return Anthos(_cfg("gqa", causal)).eval()


def test_causal_mode_has_no_leak_gap():
    r = leak_gap(_model(True), _examples(), n_loops=4, causal=True)
    assert abs(r["leak_gap"]) < 1e-5
    assert r["examples"] == 2 and r["tokens"] == 16


def test_legacy_mode_shows_a_gap():
    r = leak_gap(_model(False), _examples(), n_loops=4, causal=False)
    assert abs(r["leak_gap"]) > 1e-3
