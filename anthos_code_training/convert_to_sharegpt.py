#!/usr/bin/env python3
"""
convert_to_sharegpt.py — Convert anthos_code_sft_deduped.jsonl to Anthos ShareGPT format.

Run this AFTER download_and_build.py completes.

Usage:
    python3 convert_to_sharegpt.py
    # outputs: ../data/code_magicoder.jsonl  (ready for train.py --tier code)
"""

import json
from pathlib import Path

SYSTEM = (
    "You are Anthos, a coding AI created by Brian Tushae Thomas. "
    "You write clean, correct, well-documented code. "
    "You include all necessary imports, handle edge cases, and always provide example usage. "
    "You are direct and technical. No filler. No flattery."
)

INPUT  = Path(__file__).parent / "combined" / "anthos_code_sft_deduped.jsonl"
OUTPUT = Path(__file__).parent.parent / "data" / "code_magicoder.jsonl"

ROLE_MAP = {"user": "human", "assistant": "gpt", "system": "system"}


def convert(row: dict) -> dict | None:
    messages = row.get("messages", [])
    conversations = []

    # Always inject Anthos system prompt first
    conversations.append({"from": "system", "value": SYSTEM})

    for m in messages:
        role = ROLE_MAP.get(m.get("role", ""), "")
        content = (m.get("content") or "").strip()
        if not role or role == "system":
            continue  # skip original system prompts — we use Anthos's
        if not content:
            continue
        conversations.append({"from": role, "value": content})

    # Must have at least one human + one gpt turn
    roles = [c["from"] for c in conversations]
    if "human" not in roles or "gpt" not in roles:
        return None

    return {"conversations": conversations}


def main():
    if not INPUT.exists():
        print(f"ERROR: {INPUT} not found. Run download_and_build.py first.")
        return

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    total = kept = skipped = 0

    with INPUT.open(encoding="utf-8") as fi, OUTPUT.open("w", encoding="utf-8") as fo:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            total += 1
            row = json.loads(line)
            out = convert(row)
            if out:
                fo.write(json.dumps(out, ensure_ascii=False) + "\n")
                kept += 1
            else:
                skipped += 1

    print(f"✅ Converted {kept:,} examples → {OUTPUT}")
    print(f"   Skipped {skipped:,} rows (missing user/assistant turns)")
    print(f"\nNext: merge with existing code data and train:")
    print(f"  cat data/code_teacher.jsonl data/code_eval.jsonl data/code_magicoder.jsonl > data/code_combined.jsonl")
    print(f"  python3 train.py --tier code --steps 15000")


if __name__ == "__main__":
    main()
