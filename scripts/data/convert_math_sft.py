#!/usr/bin/env python3
"""
convert_math_sft.py — Download and convert math reasoning SFT dataset

Source: divelab/combined_gsm8k_math_dataset_dapo_math_17k_Qwen3-4B_ntokens2048_sft
Format: messages (role/content) + prompt/response fields
Output: data/math_sft.jsonl — compatible with ChatInstructDataset

Usage:
    python convert_math_sft.py
    python convert_math_sft.py --output data/math_sft.jsonl
"""

import json
import argparse
from pathlib import Path

HF_DATASET  = "divelab/combined_gsm8k_math_dataset_dapo_math_17k_Qwen3-4B_ntokens2048_sft"
DEFAULT_OUT = "data/math_sft.jsonl"

ANTHOS_SYSTEM = (
    "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
    "When solving math problems, reason step by step before giving your final answer."
)


def convert(output_path: str) -> int:
    from datasets import load_dataset

    print(f"Downloading {HF_DATASET} ...")
    ds = load_dataset(HF_DATASET, split="train")
    print(f"  {len(ds):,} examples loaded")

    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    kept = skipped = 0

    with open(dst, "w") as out:
        for item in ds:
            prompt   = item.get("prompt", "").strip()
            response = item.get("response", "").strip()

            if not prompt or not response:
                skipped += 1
                continue

            # Only keep examples the model got correct
            if item.get("is_correct") is False:
                skipped += 1
                continue

            convs = [
                {"from": "system", "value": ANTHOS_SYSTEM},
                {"from": "human",  "value": prompt},
                {"from": "gpt",    "value": response},
            ]

            out.write(json.dumps({"conversations": convs}) + "\n")
            kept += 1

    print(f"\n  Done.")
    print(f"  Kept    : {kept:,}")
    print(f"  Skipped : {skipped:,} (wrong answers or missing fields)")
    print(f"  Output  : {dst}")
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=DEFAULT_OUT)
    args = parser.parse_args()

    kept = convert(args.output)

    if kept == 0:
        print("Nothing kept.")
        return

    print(f"\nTo mix math into SFT training, combine with OpenHermes:")
    print(f"  cat data/openhermes_sft.jsonl data/math_sft.jsonl > data/sft_combined.jsonl")
    print(f"\nOr train math-only fine-tune on top of an SFT checkpoint:")
    print(f"  python train.py --tier sft --resume checkpoints/anthos-sft/step_010000.pt")
    print(f"  (after setting dataset to \"{args.output}\" in configs.py)")


if __name__ == "__main__":
    main()
