#!/usr/bin/env python3
"""
validate_dataset.py — Pre-training data quality check for sft_master.jsonl

Checks every pair for:
  - Valid JSON
  - Correct conversation structure (system/human/gpt roles)
  - Non-empty question and answer
  - Answer length >= 15 chars
  - Token length outliers (would OOM the GPU at seq_len=512)
  - Duplicate questions (waste of training budget)

Usage:
    python validate_dataset.py
    python validate_dataset.py --file data/sft_master.jsonl --seq-len 512
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def validate(path: str, seq_len: int = 512, sample_dupes: int = 5,
             tok_path: str = "data/anthos_tokenizer") -> bool:
    path = Path(path)
    if not path.exists():
        print(f"ERROR: {path} not found.")
        return False

    print(f"\nValidating {path} ...")
    print(f"  File size : {path.stat().st_size / 1e9:.2f} GB")

    # Try to load tokenizer for length checks
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tok_path)
        print(f"  Tokenizer : {tok_path} ✓")
    except Exception:
        print(f"  Tokenizer : not loaded (skipping token-length checks)")

    total = 0
    bad_json = 0
    bad_structure = 0
    empty_q = 0
    empty_a = 0
    short_a = 0
    over_len = 0
    over_len_samples = []
    question_hashes: Counter = Counter()
    dupe_q = 0

    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total += 1

            # JSON validity
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                if bad_json <= 3:
                    print(f"  BAD JSON line {lineno}: {line[:80]}")
                continue

            # Structure check
            convs = obj.get("conversations", [])
            if not isinstance(convs, list) or len(convs) < 2:
                bad_structure += 1
                continue

            roles = [c.get("from", "") for c in convs]
            human_turns = [c for c in convs if c.get("from") in ("human", "user")]
            gpt_turns   = [c for c in convs if c.get("from") in ("gpt", "assistant")]

            if not human_turns or not gpt_turns:
                bad_structure += 1
                continue

            q = str(human_turns[0].get("value", "")).strip()
            a = str(gpt_turns[0].get("value", "")).strip()

            if not q:
                empty_q += 1
                continue
            if not a:
                empty_a += 1
                continue
            if len(a) < 15:
                short_a += 1
                continue

            # Duplicate detection (hash first 100 chars of question)
            q_key = q[:100].lower()
            question_hashes[q_key] += 1
            if question_hashes[q_key] == 2:
                dupe_q += 1

            # Token length check
            if tokenizer:
                full_text = " ".join(c.get("value", "") for c in convs)
                n_tokens = len(tokenizer.encode(full_text, add_special_tokens=False))
                if n_tokens > seq_len:
                    over_len += 1
                    if len(over_len_samples) < sample_dupes:
                        over_len_samples.append((lineno, n_tokens, q[:60]))

            if total % 200_000 == 0:
                print(f"  ... {total:,} lines checked")

    good = total - bad_json - bad_structure - empty_q - empty_a - short_a
    print(f"\n{'═'*60}")
    print(f"  Total lines     : {total:,}")
    print(f"  Valid pairs     : {good:,}  ({100*good/max(total,1):.1f}%)")
    print(f"  Bad JSON        : {bad_json:,}")
    print(f"  Bad structure   : {bad_structure:,}")
    print(f"  Empty question  : {empty_q:,}")
    print(f"  Empty answer    : {empty_a:,}")
    print(f"  Answer < 15c    : {short_a:,}")
    print(f"  Duplicate Q     : {dupe_q:,}  ({100*dupe_q/max(total,1):.1f}%)")
    if tokenizer:
        print(f"  Over seq_len    : {over_len:,}  (will be truncated at {seq_len} tokens)")
        if over_len_samples:
            print(f"  Over-len examples:")
            for ln, nt, q in over_len_samples:
                print(f"    line {ln}: {nt} tokens — \"{q}\"")
    print(f"{'═'*60}")

    ok = bad_json == 0 and bad_structure < total * 0.01 and empty_q < total * 0.01
    if ok:
        print(f"\n  PASS — dataset looks clean. Ready for training.\n")
    else:
        print(f"\n  WARN — issues found. Review before training.\n")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",    default="data/sft_master.jsonl")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--tok",     default="data/anthos_tokenizer")
    args = parser.parse_args()
    ok = validate(args.file, args.seq_len, tok_path=args.tok)
    sys.exit(0 if ok else 1)
