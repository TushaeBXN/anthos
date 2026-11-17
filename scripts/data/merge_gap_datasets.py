"""
Merge gap-filling datasets into anthos_local_merged.jsonl.
Run after all downloads complete.

Datasets:
  - sft_math_reasoning.jsonl  (NuminaMath CoT, 50K)
  - sft_agentic_loops.jsonl   (AgentInstruct, 50K)
"""

import json, os, sys
from pathlib import Path

MERGED = Path("/Users/anthosintelligence/anthos/data/anthos_local_merged.jsonl")
DRIVE = Path("/Volumes/1TB Drive/anthos-data/datasets")

SYSTEM = (
    "You are Anthos. You were built by Brian Tushae Thomas — an independent developer "
    "and founder of Anthos Intelligence Company. You are not Qwen, not ChatGPT, not made "
    "by Alibaba or any other corporation. You are direct, technically precise, and you "
    "think before you answer."
)

SOURCES = [
    ("sft_math_reasoning.jsonl", "math"),
    ("sft_agentic_loops.jsonl", "agentic"),
]


def normalize(rec, source):
    if "messages" in rec:
        msgs = rec["messages"]
        # inject system if missing
        if msgs and msgs[0].get("role") != "system":
            msgs = [{"role": "system", "content": SYSTEM}] + msgs
        return {"messages": msgs}
    elif "instruction" in rec and "response" in rec:
        return {"messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": rec["instruction"]},
            {"role": "assistant", "content": rec["response"]},
        ]}
    return None


def main():
    before = sum(1 for _ in open(MERGED))
    print(f"Records before merge: {before:,}")

    total_added = 0
    with open(MERGED, "a") as fout:
        for filename, source in SOURCES:
            path = DRIVE / filename
            if not path.exists():
                print(f"SKIP {filename} — not found, still downloading?")
                continue
            added = 0
            with open(path, encoding="utf-8", errors="ignore") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    out = normalize(rec, source)
                    if out:
                        fout.write(json.dumps(out) + "\n")
                        added += 1
            print(f"  {source}: +{added:,} records from {filename}")
            total_added += added

    after = before + total_added
    print(f"\nTotal added: {total_added:,}")
    print(f"Final record count: {after:,}")


if __name__ == "__main__":
    main()
