#!/usr/bin/env python3
"""
check_padding_fraction.py — SFT batch padding diagnostic

Loads the SFT chat dataloader (same config as train.py --tier sft) and
samples N batches to measure average non-pad token fraction per batch.

Verdict:
  >= 90% non-pad  →  padding is noise, current setup is fine, close this item
  <  90% non-pad  →  worth adding an attention mask before drawing conclusions
                      from the Stage 1 loss curve

The collator pads with zeros (token 0), so non-pad = (input_ids != 0).

Usage:
    python3 check_padding_fraction.py
    python3 check_padding_fraction.py --n-batches 200 --dataset data/train.jsonl
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

GOOD_THRESHOLD  = 0.90   # fraction above which padding is not a concern
WARN_THRESHOLD  = 0.80   # fraction below which the problem is material


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-batches", type=int, default=100,
                        help="Number of batches to sample (default: 100)")
    parser.add_argument("--dataset",   type=str, default="data/sft_master.jsonl",
                        help="SFT JSONL file to load (same as train.py uses)")
    parser.add_argument("--seq-len",   type=int, default=512)
    parser.add_argument("--batch-size",type=int, default=8)
    args = parser.parse_args()

    if not Path(args.dataset).exists():
        print(f"✕ Dataset not found: {args.dataset}")
        print("  Run ingest scripts + build_training_manifest.py first, or pass --dataset.")
        sys.exit(1)

    print(f"Padding fraction check")
    print(f"  Dataset   : {args.dataset}")
    print(f"  Batches   : {args.n_batches}")
    print(f"  Seq len   : {args.seq_len} | Batch size: {args.batch_size}")
    print(f"  Threshold : >= {GOOD_THRESHOLD:.0%} non-pad → fine | < {WARN_THRESHOLD:.0%} → fix before conclusions")
    print()

    from anthos.data import get_chat_dataloader
    loader = get_chat_dataloader(
        seq_len        = args.seq_len,
        batch_size     = args.batch_size,
        num_workers    = 0,
        tokenizer_path = "data/anthos_tokenizer",
        max_samples    = 0,
        dataset_name   = args.dataset,
    )

    total_tokens  = 0
    nonpad_tokens = 0
    batch_fracs   = []
    data_iter     = iter(loader)

    for i in range(args.n_batches):
        try:
            batch = next(data_iter)
        except StopIteration:
            print(f"  (dataloader exhausted after {i} batches)")
            break

        # Chat collator returns (input_ids, labels); both are (B, T)
        if isinstance(batch, (list, tuple)):
            input_ids = batch[0]
        else:
            input_ids = batch

        n_total  = input_ids.numel()
        n_nonpad = (input_ids != 0).sum().item()
        frac     = n_nonpad / n_total if n_total > 0 else 1.0
        batch_fracs.append(frac)
        total_tokens  += n_total
        nonpad_tokens += n_nonpad

        if (i + 1) % 25 == 0:
            running = sum(batch_fracs) / len(batch_fracs)
            print(f"  batch {i+1:4d} | running avg non-pad: {running:.1%}", flush=True)

    if not batch_fracs:
        print("✕ No batches loaded — check dataset path and format.")
        sys.exit(1)

    avg_frac = nonpad_tokens / total_tokens if total_tokens else 0
    min_frac = min(batch_fracs)
    max_frac = max(batch_fracs)

    print()
    print(f"{'─'*50}")
    print(f"  Batches sampled     : {len(batch_fracs)}")
    print(f"  Total tokens        : {total_tokens:,}")
    print(f"  Non-pad tokens      : {nonpad_tokens:,}")
    print(f"  Avg non-pad fraction: {avg_frac:.1%}")
    print(f"  Min / Max per batch : {min_frac:.1%} / {max_frac:.1%}")
    print()

    if avg_frac >= GOOD_THRESHOLD:
        print(f"  ✓ FINE — {avg_frac:.1%} non-pad is above {GOOD_THRESHOLD:.0%}.")
        print(f"    Padding is rare enough that attending to token-0 is noise.")
        print(f"    Current collator (no attention mask) is acceptable for Stage 1.")
        print(f"    Close this item — no fix needed before drawing conclusions.")
    elif avg_frac >= WARN_THRESHOLD:
        print(f"  ⚠ MONITOR — {avg_frac:.1%} non-pad is below {GOOD_THRESHOLD:.0%} but above {WARN_THRESHOLD:.0%}.")
        print(f"    Some padding overhead, but probably not enough to distort the")
        print(f"    Stage 1 loss curve materially. Watch the loss trajectory;")
        print(f"    if it's noisy, add attention_mask before Stage 2.")
    else:
        print(f"  ✕ ACT — {avg_frac:.1%} non-pad is below {WARN_THRESHOLD:.0%}.")
        print(f"    Model is spending real attention capacity on token-0 padding.")
        print(f"    Add attention_mask to the collator before drawing conclusions")
        print(f"    from Stage 1. See anthos/data.py get_chat_dataloader collate_fn.")
        print()
        print(f"  Fix: in the collate_fn, add:")
        print(f"    attention_mask = (padded_tokens != 0).long()")
        print(f"    return padded_tokens, padded_labels, attention_mask")
        print(f"  Then pass attention_mask to model() in train.py.")
    print(f"{'─'*50}")


if __name__ == "__main__":
    main()
