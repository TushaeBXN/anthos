"""Remap the cl100k shards to a small vocabulary so tiny ablation models spend their capacity on the core, not on a
100K-row embedding. The most frequent ids keep a slot, everything else becomes one <unk> id. A proxy for comparing
architecture pieces on a laptop; the models trained on it are not the real thing.

    python make_small_vocab_data.py --src data/pretrain_cl100k --out data/pretrain_small8k --vocab 8192 --train-tokens 60000000
"""
import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="data/pretrain_cl100k")
ap.add_argument("--out", default="data/pretrain_small8k")
ap.add_argument("--vocab", type=int, default=8192)
ap.add_argument("--train-tokens", type=int, default=60_000_000)
ap.add_argument("--val-tokens", type=int, default=2_000_000)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
src = Path(a.src)
tr = np.memmap(src / "train_00000.bin", dtype=np.uint32, mode="r")[:a.train_tokens]
va = np.memmap(src / "val.bin", dtype=np.uint32, mode="r")[:a.val_tokens]
counts = np.bincount(np.asarray(tr[:30_000_000]), minlength=100_352)
top = np.argsort(-counts)[:a.vocab - 1]
unk = a.vocab - 1
table = np.full(100_352, unk, dtype=np.uint16)
table[top] = np.arange(len(top), dtype=np.uint16)
tr_m, va_m = table[np.asarray(tr)], table[np.asarray(va)]
tr_m.tofile(out / "train_00000.bin"); va_m.tofile(out / "val.bin")
np.save(out / "id_table.npy", table)
cover = float((tr_m != unk).mean())
(out / "meta.json").write_text(json.dumps({"dtype": "uint16", "vocab_size": a.vocab, "unk": unk, "coverage": cover,
                                            "train_tokens": int(len(tr_m)), "val_tokens": int(len(va_m)),
                                            "source": str(src)}))
print(f"vocab {a.vocab}, train {len(tr_m):,} tokens, val {len(va_m):,}, tokens kept (not <unk>): {cover:.1%}")
