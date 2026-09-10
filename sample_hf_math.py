#!/usr/bin/env python3
"""
sample_hf_math.py — Stream a HuggingFace math dataset and cap at MAX_SAMPLES.

Uses streaming=True so it never downloads the full dataset.
Converts to Anthos conversations format and writes to data/hf_math_sft.jsonl.

Usage:
    python sample_hf_math.py
    python sample_hf_math.py --max 10000
    python sample_hf_math.py --dataset TIGER-Lab/MathInstruct --max 5000
"""

import json
import argparse
from pathlib import Path

ANTHOS_MATH_SYSTEM = (
    "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
    "You solve math problems step by step, showing your reasoning clearly. "
    "You explain each step so the reader understands not just the answer but the method. "
    "When code would help verify the answer, you include it."
)

# Candidate datasets — tried in order until one works
DATASETS = [
    # (dataset_id, split, input_field, output_field)
    ("TIGER-Lab/MathInstruct",    "train", "instruction", "output"),
    ("openai/gsm8k",              "train", "question",    "answer"),
    ("lighteval/MATH",            "train", "problem",     "solution"),
    ("math_dataset",              "train", "question",    "answer"),
]

def make_conv(q, a):
    return {
        "conversations": [
            {"from": "system", "value": ANTHOS_MATH_SYSTEM},
            {"from": "human",  "value": q.strip()},
            {"from": "gpt",    "value": a.strip()[:3000]},
        ]
    }

def stream_dataset(dataset_id, split, q_field, a_field, max_samples, output_path):
    from datasets import load_dataset

    print(f"Streaming {dataset_id} (split={split}, max={max_samples:,}) ...")

    try:
        ds = load_dataset(dataset_id, split=split, streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"  Failed to load: {e}")
        return 0

    count = 0
    skipped = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for row in ds:
            if count >= max_samples:
                break

            q = row.get(q_field, "").strip()
            a = row.get(a_field, "").strip()

            if not q or not a or len(q) < 10 or len(a) < 5:
                skipped += 1
                continue

            f.write(json.dumps(make_conv(q, a)) + "\n")
            count += 1

            if count % 1000 == 0:
                print(f"  {count:,} collected ...", flush=True)

    print(f"  Done: {count:,} pairs written ({skipped} skipped)")
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None,
                        help="HuggingFace dataset ID (default: auto-try list)")
    parser.add_argument("--split",   default="train")
    parser.add_argument("--q",       default=None, help="Question field name")
    parser.add_argument("--a",       default=None, help="Answer field name")
    parser.add_argument("--max",     type=int, default=5000,
                        help="Max samples to collect (default: 5000)")
    parser.add_argument("--output",  default="data/hf_math_sft.jsonl")
    args = parser.parse_args()

    output = Path(args.output)

    if args.dataset:
        q_field = args.q or "question"
        a_field = args.a or "answer"
        count = stream_dataset(args.dataset, args.split, q_field, a_field, args.max, output)
    else:
        count = 0
        for ds_id, split, q_field, a_field in DATASETS:
            count = stream_dataset(ds_id, split, q_field, a_field, args.max, output)
            if count > 0:
                break
        if count == 0:
            print("All datasets failed. Check your internet connection or install: pip install datasets")
            return

    print(f"\nOutput: {output}  ({count:,} pairs)")
    print(f"\nTo merge into sft_master.jsonl:")
    print(f"  cat data/sft_master.jsonl {output} > /tmp/merged.jsonl && mv /tmp/merged.jsonl data/sft_master.jsonl")


if __name__ == "__main__":
    main()
