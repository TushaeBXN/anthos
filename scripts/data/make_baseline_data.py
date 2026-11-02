"""
make_baseline_data.py — free, reproducible chat data for the legacy-vs-causal baseline.

Takes the first --n conversations of mlabonne/FineTome-100k (ShareGPT-style: human -> gpt),
shuffles with a fixed seed, and writes disjoint train/val files in the format train.py reads.

    pip install datasets
    python3 make_baseline_data.py --n 22000 --val 300
    # -> data/baseline_train.jsonl, data/baseline_val.jsonl
"""

import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=22000)
    ap.add_argument("--val", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("mlabonne/FineTome-100k", split="train")

    rows = []
    for ex in ds:
        convs = ex.get("conversations", [])
        human = next((c["value"] for c in convs if c["from"] == "human"), "")
        gpt = next((c["value"] for c in convs if c["from"] == "gpt"), "")
        if human.strip() and gpt.strip():
            rows.append({"conversations": [
                {"from": "system", "value": "You are Anthos, a helpful and honest assistant."},
                {"from": "human", "value": human},
                {"from": "gpt", "value": gpt},
            ]})
        if len(rows) >= args.n:
            break

    random.Random(args.seed).shuffle(rows)
    val, train = rows[:args.val], rows[args.val:]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("baseline_train", train), ("baseline_val", val)):
        with open(out / f"{name}.jsonl", "w") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(part)} examples")


if __name__ == "__main__":
    main()
