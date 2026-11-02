#!/usr/bin/env python3
"""
ingest_deepseek_reasoning.py — Pull WithinUsAI/DeepSeek_V4_Flash_distilled_dataset_5k
and convert to Anthos conversations format.

Combines reasoning_trace + final_answer into one GPT response so Anthos
learns to show its work, not just produce answers.

Output: data/deepseek_reasoning_sft.jsonl
"""

import json
from pathlib import Path
from datasets import load_dataset

OUTPUT = Path("data/deepseek_reasoning_sft.jsonl")

ANTHOS_REASONING_SYSTEM = (
    "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
    "When solving problems, you think through them step by step — showing your reasoning "
    "clearly before giving the final answer. You don't skip steps. "
    "You explain your logic so the reader can follow and verify your thinking."
)

def make_conv(prompt, reasoning, final_answer):
    response = reasoning.strip()
    if final_answer.strip():
        response += f"\n\n**Final answer:** {final_answer.strip()}"
    return {
        "conversations": [
            {"from": "system", "value": ANTHOS_REASONING_SYSTEM},
            {"from": "human",  "value": prompt.strip()},
            {"from": "gpt",    "value": response[:4000]},
        ]
    }

def main():
    print("Loading WithinUsAI/DeepSeek_V4_Flash_distilled_dataset_5k ...")
    ds = load_dataset("WithinUsAI/DeepSeek_V4_Flash_distilled_dataset_5k", split="train")
    print(f"Loaded {len(ds):,} rows")
    print(f"Fields: {ds.column_names}")

    pairs = []
    skipped = 0

    for row in ds:
        prompt   = row.get("prompt", "").strip()
        reasoning = row.get("reasoning_trace", "").strip()
        answer    = row.get("final_answer", "").strip()

        if not prompt or not reasoning or len(reasoning) < 50:
            skipped += 1
            continue

        pairs.append(make_conv(prompt, reasoning, answer))

    print(f"\nConverted: {len(pairs):,} pairs ({skipped} skipped)")

    # Domain breakdown
    from collections import Counter
    domains = Counter(row.get("domain", "unknown") for row in ds)
    print("\nDomain breakdown:")
    for domain, count in domains.most_common():
        print(f"  {domain:20s} {count:,}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    print(f"\nOutput: {OUTPUT}")
    print(f"\nTo merge into sft_master.jsonl:")
    print(f"  cat data/sft_master.jsonl {OUTPUT} > /tmp/merged.jsonl && mv /tmp/merged.jsonl data/sft_master.jsonl")

if __name__ == "__main__":
    main()
