"""
prefix_len plumbing: chat dataloader -> model.forward (cfg.thought_prefix_only).
Uses a stub tokenizer so the test needs neither transformers nor downloaded files.
"""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from anthos.data import get_chat_dataloader
from anthos.main import Anthos, AnthosConfig

SPECIALS = ["<|system|>", "<|user|>", "<|thought|>", "<|assistant|>", "<|end|>", "<|eos|>", "<unk>"]
WORDS = "you are anthos what is two plus four answer the sky blue why because light scatters".split()


class _StubTokenizer:
    """Whitespace word-level tokenizer; stands in for transformers.AutoTokenizer."""

    def __init__(self):
        self.vocab = {t: i for i, t in enumerate(SPECIALS + WORDS)}
        self.eos_token_id = self.vocab["<|eos|>"]

    def convert_tokens_to_ids(self, tok):
        return self.vocab[tok]

    def encode(self, text, add_special_tokens=False, truncation=True, max_length=None):
        ids = [self.vocab.get(w, self.vocab["<unk>"]) for w in text.split()]
        return ids[:max_length] if max_length else ids


@pytest.fixture(autouse=True)
def _fake_transformers(monkeypatch):
    """ChatInstructDataset needs AutoTokenizer.from_pretrained and an importable `datasets`."""
    import types
    mod = types.ModuleType("transformers")
    mod.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda path: _StubTokenizer())
    monkeypatch.setitem(sys.modules, "transformers", mod)
    ds = types.ModuleType("datasets")          # imported but unused for local .jsonl files
    ds.load_dataset = lambda *a, **k: (_ for _ in ()).throw(AssertionError("should read the local file"))
    monkeypatch.setitem(sys.modules, "datasets", ds)


@pytest.fixture(scope="module")
def tok_dir():
    return "unused"


@pytest.fixture(scope="module")
def data_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "chat.jsonl"
    rows = [
        ("you are anthos", "what is two plus four", "answer the sky blue because light scatters"),
        ("you are anthos", "why", "because light scatters the sky blue"),
        ("you are anthos", "what is two plus two plus four", "answer four"),
    ]
    with open(path, "w") as f:
        for sysm, q, a in rows:
            f.write(json.dumps({"conversations": [
                {"from": "system", "value": sysm},
                {"from": "human", "value": q},
                {"from": "gpt", "value": a},
            ]}) + "\n")
    return str(path)


def _loader(tok_dir, data_file, **kw):
    return get_chat_dataloader(
        seq_len=64, batch_size=3, num_workers=0, tokenizer_path=tok_dir,
        dataset_name=data_file, **kw,
    )


def test_default_loader_is_unchanged(tok_dir, data_file):
    batch = next(iter(_loader(tok_dir, data_file)))
    assert len(batch) == 2


def test_prefix_len_marks_the_prompt_boundary(tok_dir, data_file):
    ids, labels, prefix_len = next(iter(_loader(tok_dir, data_file, return_prefix_len=True)))
    assert prefix_len.shape == (3,) and prefix_len.dtype == torch.long
    ast = 3   # id of <|assistant|> in the throw-away vocab
    for b in range(3):
        p = int(prefix_len[b])
        assert (labels[b, :p] == -100).all()          # prompt is masked
        assert ids[b, p].item() == ast                # response starts at the boundary
        assert labels[b, p].item() != -100


def test_loader_to_model_is_causal_after_the_prompt(tok_dir, data_file):
    ids, _, prefix_len = next(iter(_loader(tok_dir, data_file, return_prefix_len=True)))
    cfg = AnthosConfig(
        vocab_size=100, dim=64, n_heads=4, n_kv_heads=2, max_seq_len=64, max_loop_iters=3,
        prelude_layers=1, coda_layers=1, n_thought_tokens=4, attn_type="gqa", n_experts=4,
        n_shared_experts=1, n_experts_per_tok=2, expert_dim=32,
        thought_prefix_only=True, moe_fix_weight_order=True,
    )
    torch.manual_seed(0)
    model = Anthos(cfg).eval()

    # Change the last real token of example 0 only; padding after it stays padding.
    last = int((ids[0] != 0).nonzero().max())
    alt = ids.clone()
    alt[0, last] = (alt[0, last] + 1) % 100
    with torch.no_grad():
        a = model(ids, n_loops=3, prefix_len=prefix_len)
        b = model(alt, n_loops=3, prefix_len=prefix_len)
    assert (a[0, :last] - b[0, :last]).abs().max() <= 1e-6

    # And a training step runs end to end with gradients flowing.
    model.train()
    logits, aux = model(ids, n_loops=3, return_aux=True, prefix_len=prefix_len)
    (logits.float().mean() + aux).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
