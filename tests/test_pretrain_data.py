"""make_pretrain_data.py + BinPackedDataset: shard format, doc-disjoint val, windows."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

TOK = ROOT / "data" / "anthos_tokenizer" / "tokenizer.json"
pytestmark = pytest.mark.skipif(not TOK.exists(), reason="needs data/anthos_tokenizer")


def _build(tmp_path, n_docs=400):
    src = tmp_path / "docs.jsonl"
    src.write_text("\n".join(json.dumps({"text": f"Document {i}: " + "the quick brown fox jumps. " * 20}) for i in range(n_docs)))
    out = tmp_path / "pt"
    subprocess.run([sys.executable, str(ROOT / "make_pretrain_data.py"), "--local-jsonl", str(src), "--tokenizer", str(TOK),
                    "--tokens", "30000", "--val-tokens", "3000", "--shard-tokens", "10000", "--out", str(out)],
                   check=True, cwd=ROOT, capture_output=True)
    return out


def test_shards_and_meta(tmp_path):
    out = _build(tmp_path)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["val_tokens"] >= 3000 and meta["train_tokens"] >= 30000
    train = np.fromfile(out / "train_00000.bin", dtype=np.uint16)
    val = np.fromfile(out / "val.bin", dtype=np.uint16)
    assert len(train) == 10000 and train.max() <= 50256 and val[-1] == 50256


def test_val_documents_are_not_in_train(tmp_path):
    out = _build(tmp_path)
    tok = __import__("tokenizers").Tokenizer.from_file(str(TOK))
    val_text = tok.decode(np.fromfile(out / "val.bin", dtype=np.uint16).tolist())
    train_text = tok.decode(np.fromfile(out / "train_00000.bin", dtype=np.uint16).tolist())
    val_docs = {d.split(":")[0] for d in val_text.split("<|endoftext|>") if d.strip()}
    train_docs = {d.split(":")[0] for d in train_text.split("<|endoftext|>") if d.strip()}
    assert val_docs and not (val_docs & train_docs)


def test_loader_windows(tmp_path):
    from anthos.data import BinPackedDataset
    out = _build(tmp_path)
    it = iter(BinPackedDataset(str(out), seq_len=64, seed=1))
    a = [next(it) for _ in range(5)]
    assert all(w.shape == (65,) and w.dtype.is_floating_point is False for w in a)
    b = iter(BinPackedDataset(str(out), seq_len=64, seed=1))
    assert all((x == next(b)).all() for x in a)       # deterministic per seed


def test_refuses_to_overwrite(tmp_path):
    out = _build(tmp_path)
    r = subprocess.run([sys.executable, str(ROOT / "make_pretrain_data.py"), "--local-jsonl", str(tmp_path / "docs.jsonl"),
                        "--tokenizer", str(TOK), "--out", str(out)], cwd=ROOT, capture_output=True)
    assert r.returncode != 0


def test_uint32_shards_roundtrip(tmp_path):
    from anthos.data import BinPackedDataset
    src = tmp_path / "docs.jsonl"
    src.write_text("\n".join(json.dumps({"text": "hello world " * 50}) for _ in range(200)))
    out = tmp_path / "pt32"
    subprocess.run([sys.executable, str(ROOT / "make_pretrain_data.py"), "--local-jsonl", str(src), "--tokenizer", str(TOK),
                    "--dtype", "uint32", "--tokens", "5000", "--val-tokens", "500", "--shard-tokens", "2500", "--out", str(out)],
                   check=True, cwd=ROOT, capture_output=True)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["dtype"] == "uint32" and meta["eos"] == 50256
    assert np.fromfile(out / "train_00000.bin", dtype=np.uint32).max() == 50256
    w = next(iter(BinPackedDataset(str(out), seq_len=32, seed=0)))
    assert w.shape == (33,) and int(w.max()) <= 50256
