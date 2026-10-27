"""SFT batches must be shifted so position i predicts token i+1 (not token i)."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from anthos.data import next_token_pair


def test_labels_are_next_token_of_inputs():
    ids = torch.arange(10)[None]
    labels = ids.clone()
    labels[:, :4] = -100
    x, y = next_token_pair(ids, labels)
    assert x.shape == y.shape == (1, 9)
    assert torch.equal(y[0, 3:], x[0, 3:] + 1)   # every scored label is the token after its input
    assert (y[0, :3] == -100).all()              # prompt stays masked


def test_train_uses_the_shift():
    src = (Path(__file__).parent.parent / "train.py").read_text()
    assert "next_token_pair(input_ids, labels)" in src
