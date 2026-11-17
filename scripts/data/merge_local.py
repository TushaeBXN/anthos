#!/usr/bin/env python3
"""
merge_local.py — Merge all local Anthos training data into one JSONL.
Normalizes every file to {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
Usage: python3 scripts/data/merge_local.py --out data/anthos_local_merged.jsonl
"""
import json
import random
import sys
from pathlib import Path

SOURCES = [
    "/Users/anthosintelligence/anthos/data/distill_defiant.jsonl",
    "/Users/anthosintelligence/anthos/data/distill_mimo_mixed.jsonl",
    "/Users/anthosintelligence/anthos/data/sft_convo/merged.jsonl",
    "/Users/anthosintelligence/anthos-causal-fix/data/teacher_conversations.jsonl",
    "/Users/anthosintelligence/anthos-causal-fix/data/local_train.jsonl",
    "/Users/anthosintelligence/anthos-causal-fix/data/sft_synth_traj_train.jsonl",
    "/Users/anthosintelligence/anthos-causal-fix/data/sft_synth_verifiable.jsonl",
    "/Users/anthosintelligence/anthos-causal-fix/data/sft_synth_traj.jsonl",
]

ANTHOS_SYSTEM = (
    "You are Anthos. You were built by Brian Tushae Thomas — an independent developer "
    "and founder of Anthos Intelligence Company. You are not Qwen, not ChatGPT, not made "
    "by Alibaba or any other corporation. You are direct, technically precise, and you "
    "think before you answer."
)

def normalize(obj: dict) -> dict | None:
    """Convert any format to standard messages format."""

    # Already messages format
    if "messages" in obj:
        msgs = obj["messages"]
        if not isinstance(msgs, list) or len(msgs) < 2:
            return None
        # Ensure at least one user + one assistant
        roles = [m.get("role") for m in msgs]
        if "user" not in roles or "assistant" not in roles:
            return None
        # Strip empty system or inject Anthos system
        clean = [m for m in msgs if m.get("role") != "system" and m.get("content", "").strip()]
        if len(clean) < 2:
            return None
        return {"messages": [{"role": "system", "content": ANTHOS_SYSTEM}] + clean}

    # instruction/response format
    instruction = (obj.get("instruction") or obj.get("prompt") or
                   obj.get("question") or obj.get("input") or "").strip()
    response = (obj.get("response") or obj.get("output") or
                obj.get("answer") or "").strip()
    if instruction and response and len(response) > 20:
        return {"messages": [
            {"role": "system", "content": ANTHOS_SYSTEM},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]}

    # conversations format (ShareGPT style)
    if "conversations" in obj:
        convs = obj["conversations"]
        human = next((c["value"] for c in convs if c.get("from") == "human"), "").strip()
        gpt = next((c["value"] for c in convs if c.get("from") in ("gpt", "assistant")), "").strip()
        if human and gpt and len(gpt) > 20:
            return {"messages": [
                {"role": "system", "content": ANTHOS_SYSTEM},
                {"role": "user", "content": human},
                {"role": "assistant", "content": gpt},
            ]}

    # Agentic trace format: task + steps + final
    if "task" in obj and "final" in obj:
        task = obj.get("task", "").strip()
        final = obj.get("final", "").strip()
        steps = obj.get("steps", [])
        if task and final and len(final) > 10:
            if steps:
                step_lines = []
                for s in steps:
                    call = s.get("call", {})
                    result = s.get("result", "")
                    step_lines.append(f"→ {call.get('name', 'action')}({call.get('args', {})}) = {result}")
                response = "\n".join(step_lines) + f"\n\n{final}"
            else:
                response = final
            return {"messages": [
                {"role": "system", "content": ANTHOS_SYSTEM},
                {"role": "user", "content": task},
                {"role": "assistant", "content": response},
            ]}

    return None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/anthos_local_merged.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_records = []
    for src in SOURCES:
        path = Path(src)
        if not path.exists():
            print(f"  SKIP (not found): {path.name}")
            continue
        count = 0
        skipped = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    rec = normalize(obj)
                    if rec:
                        all_records.append(rec)
                        count += 1
                    else:
                        skipped += 1
                except Exception:
                    skipped += 1
        print(f"  {path.name}: {count:,} kept, {skipped} skipped")

    print(f"\nTotal: {len(all_records):,} records")
    print("Shuffling...")
    random.seed(a.seed)
    random.shuffle(all_records)

    with open(out_path, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    size_mb = out_path.stat().st_size / 1_000_000
    print(f"Written: {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
