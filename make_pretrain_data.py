"""
make_pretrain_data.py — pre-tokenise a text corpus into flat uint16 shards for pretraining.

Streams a Hugging Face dataset (or a local jsonl for testing), tokenises it, separates documents
with the tokenizer's <|endoftext|> and writes:

    <out>/val.bin             first --val-tokens tokens (whole documents, never reused for training)
    <out>/train_00000.bin ... --shard-tokens tokens per shard
    <out>/meta.json           counts, source, tokenizer

Usage:
    python3 make_pretrain_data.py --tokens 1000000000 --out data/pretrain          # FineWeb-Edu sample-10BT
    python3 make_pretrain_data.py --tokenizer tiktoken:cl100k_base --out data/pretrain_cl100k   # needs `pip install tiktoken`
    python3 make_pretrain_data.py --local-jsonl some.jsonl --tokens 50000 --out /tmp/pt_test
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def iter_texts(args):
    if args.local_jsonl:
        with open(args.local_jsonl, encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)[args.text_field]
        return
    from datasets import load_dataset
    ds = load_dataset(args.dataset, args.subset, split="train", streaming=True)
    for ex in ds:
        yield ex[args.text_field]


def batched(it, n):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


class Tok:
    """Uniform wrapper: a tokenizers tokenizer.json path, or 'tiktoken:<encoding>' (e.g. tiktoken:cl100k_base)."""

    def __init__(self, spec: str):
        if spec.startswith("tiktoken:"):
            import tiktoken
            self._enc = tiktoken.get_encoding(spec.split(":", 1)[1])
            self.eos, self.vocab = self._enc.eot_token, self._enc.n_vocab
            self.encode_batch = lambda texts: self._enc.encode_ordinary_batch(texts, num_threads=8)
        else:
            from tokenizers import Tokenizer
            t = Tokenizer.from_file(spec)
            self.eos, self.vocab = t.token_to_id("<|endoftext|>"), t.get_vocab_size()
            self.encode_batch = lambda texts: [e.ids for e in t.encode_batch(texts)]
            self._plain_limit = self.eos     # ids >= eos are specials that must not leak in from raw text


def write_shard(path: Path, ids: list[int], dtype):
    np.asarray(ids, dtype=dtype).tofile(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--subset", default="sample-10BT")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--local-jsonl", default=None, help="read documents from a local jsonl instead of Hugging Face")
    ap.add_argument("--tokenizer", default="data/anthos_tokenizer/tokenizer.json",
                    help="tokenizer.json path, or tiktoken:<encoding> such as tiktoken:cl100k_base")
    ap.add_argument("--dtype", choices=["auto", "uint16", "uint32"], default="auto")
    ap.add_argument("--tokens", type=int, default=1_000_000_000, help="train tokens to write")
    ap.add_argument("--val-tokens", type=int, default=2_000_000)
    ap.add_argument("--shard-tokens", type=int, default=100_000_000)
    ap.add_argument("--out", default="data/pretrain")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and any(out.glob("*.bin")):
        sys.exit(f"{out} already holds shards; pick a new --out (nothing is overwritten)")
    out.mkdir(parents=True, exist_ok=True)

    tok = Tok(args.tokenizer)
    dtype = np.uint32 if (args.dtype == "uint32" or (args.dtype == "auto" and tok.vocab > 65536)) else np.uint16
    assert tok.vocab <= np.iinfo(dtype).max + 1, "vocab does not fit the shard dtype"
    val, shard = [], []
    val_done = args.val_tokens == 0
    n_train = n_shards = 0
    t0 = time.time()

    for texts in batched(iter_texts(args), 256):
        for ids in tok.encode_batch([t for t in texts if t and t.strip()]):
            lim = getattr(tok, "_plain_limit", None)
            if lim is not None and max(ids, default=0) >= lim:
                ids = [i for i in ids if i < lim]      # drop special tokens that leaked in from text
            ids = list(ids) + [tok.eos]
            if not val_done:
                val.extend(ids)
                if len(val) >= args.val_tokens:
                    write_shard(out / "val.bin", val, dtype)
                    n_val, val_done = len(val), True
                continue
            shard.extend(ids)
            while len(shard) >= args.shard_tokens:
                write_shard(out / f"train_{n_shards:05d}.bin", shard[:args.shard_tokens], dtype)
                shard = shard[args.shard_tokens:]
                n_train += args.shard_tokens
                n_shards += 1
                print(f"  shard {n_shards} written | {n_train:,} train tokens | {n_train / (time.time() - t0):,.0f} tok/s", flush=True)
        if n_train + len(shard) >= args.tokens:
            break

    if shard:
        write_shard(out / f"train_{n_shards:05d}.bin", shard, dtype)
        n_train += len(shard)
        n_shards += 1
    if not val_done and val:
        write_shard(out / "val.bin", val, dtype)
        n_val = len(val)
    elif args.val_tokens == 0:
        n_val = 0

    meta = {"source": args.local_jsonl or f"{args.dataset}/{args.subset}", "tokenizer": args.tokenizer,
            "dtype": np.dtype(dtype).name, "eos": tok.eos, "vocab_size": tok.vocab, "train_tokens": n_train, "val_tokens": n_val, "train_shards": n_shards}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
