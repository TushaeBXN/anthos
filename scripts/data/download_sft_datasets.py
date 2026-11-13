#!/usr/bin/env python3
"""
download_sft_datasets.py — Download and merge all SFT datasets for Anthos Qwen transplant.

Downloads, converts, and merges the following datasets into one JSONL file:
  1. Manusagents/Omni-Frontier-Distillation-SFT-Cyber-Coding-Med-dataset-collection (cyber+coding subset)
  2. faunix/Qwen3.8-27B-Distillation-40K (reasoning traces)
  3. MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x (deep chain-of-thought)
  4. choucsan/mimo-claude-code-traces-1k (agentic tool-use)
  5. TeichAI/DeepSeek-v4-Pro-Agent (agentic coding traces)
  6. SargeDev/jev-distill-corpus-v3 (decision/knowledge QA)
  7. Local distillation data (data/distill_*.jsonl)

Output format (each line):
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Usage:
  # Download everything and merge
  python scripts/data/download_sft_datasets.py --out data/sft_anthos.jsonl

  # Cap Manusagents (huge dataset) at N rows
  python scripts/data/download_sft_datasets.py --out data/sft_anthos.jsonl --manus-limit 500000

  # Skip already-downloaded datasets (resume)
  python scripts/data/download_sft_datasets.py --out data/sft_anthos.jsonl --skip-existing

  # Only specific datasets
  python scripts/data/download_sft_datasets.py --out data/sft_anthos.jsonl --only faunix fable mimo

Requirements:
  pip install datasets huggingface_hub tqdm
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

try:
    from datasets import load_dataset
    from tqdm import tqdm
except ImportError:
    sys.exit("Install dependencies: pip install datasets huggingface_hub tqdm")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_jsonl(path: Path, records):
    """Append records (dicts) to a JSONL file, return count written."""
    written = 0
    with open(path, "a") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
            written += 1
    return written


def msg(user: str, assistant: str) -> dict:
    """Standard messages-format record."""
    return {"messages": [
        {"role": "user", "content": user.strip()},
        {"role": "assistant", "content": assistant.strip()},
    ]}


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks from assistant text."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ---------------------------------------------------------------------------
# Dataset converters
# ---------------------------------------------------------------------------

def load_manusagents(limit: int, hf_token: str | None) -> list[dict]:
    """
    Manusagents/Omni-Frontier — 16M+ rows, Apache 2.0.
    Loads the cybersecurity + coding subsets up to `limit` rows.
    Format: varies by config — instruction/response or messages.
    """
    print(f"\n[1/7] Manusagents/Omni-Frontier (limit={limit:,})...")
    records = []

    # Try to load specific configs; fall back to default split
    configs_to_try = ["cybersecurity", "coding", "default", None]
    dataset = None
    for cfg in configs_to_try:
        try:
            kwargs = dict(split="train", streaming=True, trust_remote_code=True)
            if hf_token:
                kwargs["token"] = hf_token
            if cfg:
                dataset = load_dataset(
                    "Manusagents/Omni-Frontier-Distillation-SFT-Cyber-Coding-Med-dataset-collection",
                    cfg, **kwargs)
            else:
                dataset = load_dataset(
                    "Manusagents/Omni-Frontier-Distillation-SFT-Cyber-Coding-Med-dataset-collection",
                    **kwargs)
            print(f"  Loaded config: {cfg or 'default'}")
            break
        except Exception as e:
            print(f"  config '{cfg}' failed: {e}")
            continue

    if dataset is None:
        print("  Skipping Manusagents — could not load any config.")
        return []

    for i, row in enumerate(tqdm(dataset, total=limit, desc="  Manusagents")):
        if i >= limit:
            break
        # Try multiple field name conventions
        instruction = (row.get("instruction") or row.get("prompt") or
                       row.get("question") or row.get("input") or "")
        response = (row.get("response") or row.get("output") or
                    row.get("answer") or row.get("completion") or "")

        # messages format fallback
        if not instruction and "messages" in row:
            msgs = row["messages"]
            if isinstance(msgs, list) and len(msgs) >= 2:
                instruction = msgs[0].get("content", "")
                response = msgs[-1].get("content", "")

        if instruction and response and len(response) > 20:
            records.append(msg(instruction, response))

    print(f"  Collected {len(records):,} rows from Manusagents.")
    return records


def load_faunix(hf_token: str | None) -> list[dict]:
    """
    faunix/Qwen3.8-27B-Distillation-40K — 40K rows, Apache 2.0.
    Has <think> reasoning traces; we keep them in the response.
    """
    print("\n[2/7] faunix/Qwen3.8-27B-Distillation-40K...")
    try:
        kwargs = dict(split="train", trust_remote_code=True)
        if hf_token:
            kwargs["token"] = hf_token
        dataset = load_dataset("faunix/Qwen3.8-27B-Distillation-40K", **kwargs)
    except Exception as e:
        print(f"  Failed: {e}")
        return []

    records = []
    for row in tqdm(dataset, desc="  faunix"):
        instruction = (row.get("instruction") or row.get("prompt") or
                       row.get("question") or row.get("input") or "")
        response = (row.get("response") or row.get("output") or
                    row.get("answer") or "")

        if not instruction and "messages" in row:
            msgs = row["messages"]
            if isinstance(msgs, list) and len(msgs) >= 2:
                for m in msgs:
                    if m.get("role") == "user":
                        instruction = m.get("content", "")
                    elif m.get("role") == "assistant":
                        response = m.get("content", "")

        if instruction and response and len(response) > 20:
            records.append(msg(instruction, response))

    print(f"  Collected {len(records):,} rows from faunix.")
    return records


def load_fable(hf_token: str | None) -> list[dict]:
    """
    MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x — 10K rows, Apache 2.0.
    Deep agentic coding traces with chain-of-thought. Each row has a "messages" list.
    We extract the first user turn and the last assistant turn.
    """
    print("\n[3/7] MoreThought/Fable-5.1-Max-Reasoning (10K traces)...")
    try:
        kwargs = dict(split="train", trust_remote_code=True)
        if hf_token:
            kwargs["token"] = hf_token
        dataset = load_dataset("MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x", **kwargs)
    except Exception as e:
        # Try the 'full' split
        try:
            kwargs["split"] = "full"
            dataset = load_dataset("MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x", **kwargs)
        except Exception as e2:
            print(f"  Failed: {e} / {e2}")
            return []

    records = []
    for row in tqdm(dataset, desc="  Fable"):
        messages = row.get("messages", [])
        if not isinstance(messages, list) or len(messages) < 2:
            continue

        # First user message
        user_content = ""
        for m in messages:
            if m.get("role") == "user":
                user_content = m.get("content", "")
                break

        # Last assistant message
        asst_content = ""
        for m in reversed(messages):
            if m.get("role") == "assistant":
                asst_content = m.get("content", "")
                break

        if user_content and asst_content and len(asst_content) > 50:
            records.append(msg(user_content, asst_content))

    print(f"  Collected {len(records):,} rows from Fable.")
    return records


def load_mimo(hf_token: str | None) -> list[dict]:
    """
    choucsan/mimo-claude-code-traces-1k — 1K rows, MIT.
    Claude Code-style JSONL event streams. Extract first task + final response.
    """
    print("\n[4/7] choucsan/mimo-claude-code-traces-1k...")
    try:
        kwargs = dict(split="train", trust_remote_code=True)
        if hf_token:
            kwargs["token"] = hf_token
        dataset = load_dataset("choucsan/mimo-claude-code-traces-1k", **kwargs)
    except Exception as e:
        print(f"  Failed: {e}")
        return []

    records = []
    for row in tqdm(dataset, desc="  MiMo traces"):
        # Try standard fields first
        instruction = (row.get("instruction") or row.get("prompt") or
                       row.get("question") or "")
        response = (row.get("response") or row.get("output") or "")

        if not instruction and "messages" in row:
            msgs = row["messages"]
            if isinstance(msgs, list):
                for m in msgs:
                    if isinstance(m, dict) and m.get("role") == "user" and not instruction:
                        instruction = m.get("content", "")
                    elif isinstance(m, dict) and m.get("role") == "assistant":
                        response = m.get("content", "")

        # Some traces store the task as a text field
        if not instruction:
            instruction = row.get("task") or row.get("title") or ""

        if instruction and response and len(response) > 20:
            records.append(msg(instruction, response))

    print(f"  Collected {len(records):,} rows from MiMo traces.")
    return records


def load_teichai(hf_token: str | None) -> list[dict]:
    """
    TeichAI/DeepSeek-v4-Pro-Agent — 4K rows, no explicit license (use at own risk).
    Agentic coding traces (bash, file tools).
    """
    print("\n[5/7] TeichAI/DeepSeek-v4-Pro-Agent (no license — use at own risk)...")
    try:
        kwargs = dict(split="train", trust_remote_code=True)
        if hf_token:
            kwargs["token"] = hf_token
        dataset = load_dataset("TeichAI/DeepSeek-v4-Pro-Agent", **kwargs)
    except Exception as e:
        print(f"  Failed: {e}")
        return []

    records = []
    for row in tqdm(dataset, desc="  TeichAI"):
        instruction = (row.get("instruction") or row.get("prompt") or
                       row.get("question") or row.get("task") or "")
        response = (row.get("response") or row.get("output") or
                    row.get("answer") or "")

        if not instruction and "messages" in row:
            msgs = row["messages"]
            if isinstance(msgs, list) and len(msgs) >= 2:
                for m in msgs:
                    if m.get("role") == "user" and not instruction:
                        instruction = m.get("content", "")
                    elif m.get("role") == "assistant":
                        response = m.get("content", "")

        if instruction and response and len(response) > 20:
            records.append(msg(instruction, response))

    print(f"  Collected {len(records):,} rows from TeichAI.")
    return records


def load_jev(limit: int, hf_token: str | None) -> list[dict]:
    """
    SargeDev/jev-distill-corpus-v3 — 741K rows, Apache 2.0.
    Decision/scoring format. We convert question + options to instruction/response.
    """
    print(f"\n[6/7] SargeDev/jev-distill-corpus-v3 (limit={limit:,})...")
    try:
        kwargs = dict(split="train", streaming=True, trust_remote_code=True)
        if hf_token:
            kwargs["token"] = hf_token
        dataset = load_dataset("SargeDev/jev-distill-corpus-v3", **kwargs)
    except Exception as e:
        print(f"  Failed: {e}")
        return []

    records = []
    for i, row in enumerate(tqdm(dataset, total=limit, desc="  jev")):
        if i >= limit:
            break

        question = row.get("question") or row.get("prompt") or ""
        options = row.get("options") or []
        target = row.get("target") or ""
        domain = row.get("domain") or ""

        # Build instruction from question + options
        if isinstance(options, list) and options:
            opts_str = "\n".join(f"  {j+1}. {o}" for j, o in enumerate(options))
            instruction = f"{question}\n\nOptions:\n{opts_str}"
        else:
            instruction = question

        # Build response from target/answer
        if isinstance(target, dict):
            # Soft-label distribution — pick the highest-scoring option
            best_key = max(target, key=lambda k: target[k]) if target else ""
            if isinstance(options, list) and best_key.isdigit():
                idx = int(best_key)
                answer = options[idx] if idx < len(options) else best_key
            else:
                answer = best_key
            response = f"The best answer is: {answer}"
        elif isinstance(target, str) and target:
            response = target
        else:
            continue

        if instruction and response and len(instruction) > 10:
            if domain:
                instruction = f"[{domain}] {instruction}"
            records.append(msg(instruction, response))

    print(f"  Collected {len(records):,} rows from jev.")
    return records


def load_local_distill(data_dir: Path) -> list[dict]:
    """Load local distillation JSONL files from data/distill_*.jsonl."""
    print("\n[7/7] Local distillation data...")
    records = []
    for path in sorted(data_dir.glob("distill_*.jsonl")):
        count = 0
        with open(path) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    instruction = obj.get("instruction") or obj.get("prompt") or ""
                    response = obj.get("response") or obj.get("output") or ""
                    if not instruction and "messages" in obj:
                        msgs = obj["messages"]
                        if len(msgs) >= 2:
                            instruction = msgs[0].get("content", "")
                            response = msgs[-1].get("content", "")
                    if instruction and response and len(response) > 20:
                        records.append(msg(instruction, response))
                        count += 1
                except Exception:
                    pass
        print(f"  {path.name}: {count} rows")

    print(f"  Total local: {len(records):,} rows.")
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Download and merge all SFT datasets for Anthos")
    ap.add_argument("--out", default="data/sft_anthos.jsonl", help="Output JSONL path")
    ap.add_argument("--manus-limit", type=int, default=500_000,
                    help="Max rows from Manusagents (default 500k — full dataset is 16M+)")
    ap.add_argument("--jev-limit", type=int, default=100_000,
                    help="Max rows from jev-distill-corpus (default 100k — full is 741k)")
    ap.add_argument("--only", nargs="+", metavar="NAME",
                    choices=["manus", "faunix", "fable", "mimo", "teichai", "jev", "local"],
                    help="Only load these datasets (default: all)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="If output file exists, skip (do not overwrite)")
    ap.add_argument("--shuffle", action="store_true", default=True,
                    help="Shuffle merged output (default: True)")
    ap.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hf-token", default=None,
                    help="HuggingFace token (or set HF_TOKEN env var)")
    ap.add_argument("--data-dir", default="data",
                    help="Directory containing local distill_*.jsonl files")
    a = ap.parse_args()

    import os
    hf_token = a.hf_token or os.environ.get("HF_TOKEN")

    out_path = Path(a.out)
    if a.skip_existing and out_path.exists():
        print(f"Output already exists and --skip-existing set: {out_path}")
        sys.exit(0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Start fresh
    if out_path.exists():
        out_path.unlink()

    only = set(a.only) if a.only else None
    all_records = []

    if not only or "manus" in only:
        all_records.extend(load_manusagents(a.manus_limit, hf_token))

    if not only or "faunix" in only:
        all_records.extend(load_faunix(hf_token))

    if not only or "fable" in only:
        all_records.extend(load_fable(hf_token))

    if not only or "mimo" in only:
        all_records.extend(load_mimo(hf_token))

    if not only or "teichai" in only:
        all_records.extend(load_teichai(hf_token))

    if not only or "jev" in only:
        all_records.extend(load_jev(a.jev_limit, hf_token))

    if not only or "local" in only:
        all_records.extend(load_local_distill(Path(a.data_dir)))

    print(f"\n{'='*60}")
    print(f"Total records collected: {len(all_records):,}")

    if a.shuffle:
        print(f"Shuffling (seed={a.seed})...")
        random.seed(a.seed)
        random.shuffle(all_records)

    print(f"Writing to {out_path}...")
    written = write_jsonl(out_path, all_records)
    size_mb = out_path.stat().st_size / 1_000_000

    print(f"\nDone.")
    print(f"  Records: {written:,}")
    print(f"  File:    {out_path} ({size_mb:.1f} MB)")
    print()
    print("Dataset breakdown:")
    print("  Manusagents (cyber+coding)      up to", f"{a.manus_limit:,}")
    print("  faunix (reasoning traces)       ~40,000")
    print("  Fable 5.1 (chain-of-thought)    ~10,000")
    print("  MiMo code traces                ~1,000")
    print("  TeichAI agent traces            ~4,000")
    print("  jev decision/knowledge          up to", f"{a.jev_limit:,}")
    print("  Local distill data              varies")
    print()
    print("Next step: use this file for Qwen transplant SFT:")
    print(f"  python scripts/train/sft_qwen.py --data {out_path}")


if __name__ == "__main__":
    main()
