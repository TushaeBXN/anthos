#!/usr/bin/env python3
"""
ingest_hf_batch.py — Batch ingest HuggingFace datasets into Anthos SFT format.

Streams each dataset with a cap so nothing blows up memory or disk.
Outputs one JSONL per dataset, then prints merge command.

Usage:
    python ingest_hf_batch.py
    python ingest_hf_batch.py --out-dir data/hf_batch
    python ingest_hf_batch.py --only alpaca gsm8k
"""

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# System prompts
# ─────────────────────────────────────────────────────────────────────────────

SYS_GENERAL = (
    "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
    "You give clear, helpful, accurate responses. You are direct and do not over-explain."
)

SYS_CODE = (
    "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
    "You are an expert software engineer. When writing code, you write complete, "
    "working implementations — not outlines or pseudocode. Include imports. Make it runnable."
)

SYS_MATH = (
    "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
    "You solve math problems step by step, showing your reasoning before giving the final answer. "
    "You explain each step clearly."
)

SYS_SCIENCE = (
    "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
    "You explain science concepts accurately and clearly, connecting theory to real-world application."
)

SYS_REASONING = (
    "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
    "You think through problems carefully, show your reasoning, and arrive at well-justified conclusions."
)

# ─────────────────────────────────────────────────────────────────────────────
# Dataset configs
# ─────────────────────────────────────────────────────────────────────────────
# format options:
#   "qa"         — q_field / a_field
#   "alpaca"     — instruction + input + output
#   "messages"   — messages field: [{role, content}] or [{from, value}]
#   "mmlu"       — question + choices + answer (A/B/C/D index)
#   "squad"      — context + question + answers.text[0]
#   "humaneval"  — prompt + canonical_solution
#   "codesearch" — func_documentation_string + func_code_string
#   "code_text"  — single "text" field containing code

DATASETS = [
    # ── General SFT ──────────────────────────────────────────────────────────
    {
        "name": "alpaca",
        "id": "tatsu-lab/alpaca",
        "split": "train",
        "format": "alpaca",
        "max": 5000,
        "system": SYS_GENERAL,
    },
    {
        "name": "rlcd_conversations",
        "id": "TaylorAI/RLCD-SFT-conversations",
        "split": "train",
        "format": "messages",
        "msg_field": "messages",
        "max": 5000,
        "system": SYS_GENERAL,
    },
    {
        "name": "open_yap",
        "id": "TheAgenticDataCompany/open-yap-1k",
        "split": "train",
        "format": "messages",
        "msg_field": "messages",
        "max": 1000,
        "system": SYS_GENERAL,
    },
    {
        "name": "wildchat",
        "id": "allenai/WildChat-4.8M",
        "split": "train",
        "format": "messages",
        "msg_field": "conversation",
        "max": 8000,
        "system": SYS_GENERAL,
        "streaming": True,
    },

    # ── Math ─────────────────────────────────────────────────────────────────
    {
        "name": "gsm8k",
        "id": "openai/gsm8k",
        "config": "main",
        "split": "train",
        "format": "qa",
        "q_field": "question",
        "a_field": "answer",
        "max": 7000,
        "system": SYS_MATH,
    },
    {
        "name": "deepmath",
        "id": "zwhe99/DeepMath-103K",
        "split": "train",
        "format": "qa",
        "q_field": "problem",
        "a_field": "r1_solution",
        "max": 5000,
        "system": SYS_MATH,
        "streaming": True,
    },
    {
        "name": "k12_math",
        "id": "robworks-software/k12-mathematics-standards-expanded",
        "split": "train",
        "format": "qa",
        "q_field": "question",
        "a_field": "answer",
        "max": 5000,
        "system": SYS_MATH,
    },
    {
        "name": "mmlu_hs_math",
        "id": "joey234/mmlu-high_school_mathematics-neg-prepend",
        "split": "test",
        "format": "mmlu",
        "max": 1000,
        "system": SYS_MATH,
    },

    # ── Physics ───────────────────────────────────────────────────────────────
    {
        "name": "mmlu_hs_physics",
        "id": "joey234/mmlu-high_school_physics-neg",
        "split": "test",
        "format": "mmlu",
        "max": 1000,
        "system": SYS_SCIENCE,
    },
    {
        "name": "mmlu_college_physics",
        "id": "joey234/mmlu-college_physics-neg",
        "split": "test",
        "format": "mmlu",
        "max": 1000,
        "system": SYS_SCIENCE,
    },
    {
        "name": "mmlu_conceptual_physics",
        "id": "joey234/mmlu-conceptual_physics-rule-neg",
        "split": "test",
        "format": "mmlu",
        "max": 1000,
        "system": SYS_SCIENCE,
    },

    # ── Chemistry ─────────────────────────────────────────────────────────────
    {
        "name": "chemistry_sft",
        "id": "summykai/chemistry-sft-ultra",
        "split": "train",
        "format": "qa",
        "q_field": "instruction",
        "a_field": "output",
        "max": 3000,
        "system": SYS_SCIENCE,
    },
    {
        "name": "mmlu_hs_chemistry",
        "id": "joey234/mmlu-high_school_chemistry-rule-neg",
        "split": "test",
        "format": "mmlu",
        "max": 1000,
        "system": SYS_SCIENCE,
    },
    {
        "name": "mmlu_college_chemistry_verbal",
        "id": "joey234/mmlu-college_chemistry-verbal-neg-prepend",
        "split": "test",
        "format": "mmlu",
        "max": 1000,
        "system": SYS_SCIENCE,
    },
    {
        "name": "mmlu_college_chemistry_original",
        "id": "joey234/mmlu-college_chemistry-original-neg",
        "split": "test",
        "format": "mmlu",
        "max": 1000,
        "system": SYS_SCIENCE,
    },

    # ── MMLU full ─────────────────────────────────────────────────────────────
    {
        "name": "mmlu_all",
        "id": "cais/mmlu",
        "config": "all",
        "split": "test",
        "format": "mmlu",
        "max": 5000,
        "system": SYS_REASONING,
        "streaming": True,
    },

    # ── NLP / Reading comprehension ───────────────────────────────────────────
    {
        "name": "squad",
        "id": "rajpurkar/squad",
        "split": "train",
        "format": "squad",
        "max": 5000,
        "system": SYS_GENERAL,
        "streaming": True,
    },
    {
        "name": "grammar",
        "id": "pipecat-ai/orpheus_grammar_1",
        "split": "train",
        "format": "qa",
        "q_field": "instruction",
        "a_field": "output",
        "max": 2000,
        "system": SYS_GENERAL,
    },

    # ── Code ──────────────────────────────────────────────────────────────────
    {
        "name": "humaneval",
        "id": "openai/openai_humaneval",
        "split": "test",
        "format": "humaneval",
        "max": 164,   # full dataset is only 164 rows
        "system": SYS_CODE,
    },
    {
        "name": "tiny_codes",
        "id": "nampdn-ai/tiny-codes",
        "split": "train",
        "format": "qa",
        "q_field": "prompt",
        "a_field": "response",
        "max": 5000,
        "system": SYS_CODE,
        "streaming": True,
    },
    {
        "name": "textbook_programming",
        "id": "vikp/textbook_quality_programming",
        "split": "train",
        "format": "qa",
        "q_field": "markdown",
        "a_field": "markdown",      # self-explanatory chunks
        "max": 3000,
        "system": SYS_CODE,
        "streaming": True,
        "self_explain": True,       # single-field: "explain this code"
    },
    {
        "name": "code_feedback",
        "id": "m-a-p/Code-Feedback",
        "split": "train",
        "format": "messages",
        "msg_field": "messages",
        "max": 5000,
        "system": SYS_CODE,
        "streaming": True,
    },
    {
        "name": "qwen_code",
        "id": "guell00/qwen-3.8-code",
        "split": "train",
        "format": "messages",
        "msg_field": "messages",
        "max": 3000,
        "system": SYS_CODE,
        "streaming": True,
    },
    {
        "name": "ifm_code_reasoning",
        "id": "IFM/Code-Reasoning",
        "split": "train",
        "format": "code_text",
        "text_field": "text",
        "max": 5000,
        "system": SYS_CODE,
        "streaming": True,
    },
    {
        "name": "rstar_coder",
        "id": "microsoft/rStar-Coder",
        "split": "train",
        "format": "qa",
        "q_field": "problem",
        "a_field": "solution",
        "max": 5000,
        "system": SYS_CODE,
        "streaming": True,
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Format converters
# ─────────────────────────────────────────────────────────────────────────────

CHOICE_LABELS = ["A", "B", "C", "D", "E"]

def conv(system, q, a, max_a=3500):
    q = q.strip()
    a = a.strip()[:max_a]
    if not q or not a or len(q) < 5 or len(a) < 5:
        return None
    return {
        "conversations": [
            {"from": "system", "value": system},
            {"from": "human",  "value": q},
            {"from": "gpt",    "value": a},
        ]
    }

def format_mmlu(row, system):
    q = row.get("question", "").strip()
    choices = row.get("choices", [])
    answer_idx = row.get("answer", -1)

    if not q or not choices:
        return None

    choices_str = "\n".join(f"{CHOICE_LABELS[i]}. {c}" for i, c in enumerate(choices))
    question = f"{q}\n\n{choices_str}"

    if isinstance(answer_idx, int) and 0 <= answer_idx < len(choices):
        answer = f"{CHOICE_LABELS[answer_idx]}. {choices[answer_idx]}"
    elif isinstance(answer_idx, str) and answer_idx in CHOICE_LABELS:
        idx = CHOICE_LABELS.index(answer_idx)
        answer = f"{answer_idx}. {choices[idx]}" if idx < len(choices) else answer_idx
    else:
        return None

    return conv(system, question, answer)

def format_messages(row, msg_field, system):
    msgs = row.get(msg_field, [])
    if not msgs or len(msgs) < 2:
        return None

    # Normalize role names
    def role(m):
        r = m.get("role", m.get("from", "")).lower()
        if r in ("human", "user"):        return "human"
        if r in ("gpt", "assistant"):     return "gpt"
        if r in ("system",):              return "system"
        return None

    def content(m):
        return m.get("content", m.get("value", "")).strip()

    # Find first human + first gpt exchange (skip system)
    human_msg = gpt_msg = None
    for m in msgs:
        r = role(m)
        c = content(m)
        if not c:
            continue
        if r == "human" and human_msg is None:
            human_msg = c
        elif r == "gpt" and human_msg and gpt_msg is None:
            gpt_msg = c
            break

    if not human_msg or not gpt_msg:
        return None
    return conv(system, human_msg, gpt_msg)

def format_squad(row, system):
    ctx = row.get("context", "").strip()
    q   = row.get("question", "").strip()
    answers = row.get("answers", {})
    texts = answers.get("text", []) if isinstance(answers, dict) else []
    if not ctx or not q or not texts:
        return None
    question = f"Context: {ctx[:1200]}\n\nQuestion: {q}"
    return conv(system, question, texts[0])

def format_humaneval(row, system):
    prompt   = row.get("prompt", "").strip()
    solution = row.get("canonical_solution", "").strip()
    if not prompt or not solution:
        return None
    q = f"Complete this Python function:\n\n```python\n{prompt}\n```"
    a = f"```python\n{prompt}{solution}\n```"
    return conv(system, q, a)

def format_code_text(row, text_field, system):
    text = row.get(text_field, "").strip()
    if len(text) < 100:
        return None
    q = "Explain and walk through this code:"
    # Try to extract a function/class header as the question
    lines = text.splitlines()
    for line in lines[:10]:
        if line.strip().startswith(("def ", "class ", "# ")):
            q = f"Explain this code:\n```\n{line.strip()}\n```"
            break
    return conv(system, q, f"```\n{text[:3000]}\n```")

def format_alpaca(row, system):
    instruction = row.get("instruction", "").strip()
    inp         = row.get("input", "").strip()
    output      = row.get("output", "").strip()
    if not instruction or not output:
        return None
    q = f"{instruction}\n\n{inp}".strip() if inp else instruction
    return conv(system, q, output)

def format_self_explain(row, field, system):
    text = row.get(field, "").strip()
    if len(text) < 100:
        return None
    snippet = text[:200].splitlines()[0].strip()
    q = f"Explain this programming concept or code:\n\n{snippet}"
    return conv(system, q, text[:3000])

# ─────────────────────────────────────────────────────────────────────────────
# Stream one dataset
# ─────────────────────────────────────────────────────────────────────────────

def ingest_one(cfg, out_dir):
    from datasets import load_dataset

    name      = cfg["name"]
    ds_id     = cfg["id"]
    split     = cfg.get("split", "train")
    config    = cfg.get("config", None)
    fmt       = cfg["format"]
    system    = cfg["system"]
    max_n     = cfg.get("max", 5000)
    streaming = cfg.get("streaming", False)

    out_path = out_dir / f"{name}.jsonl"
    print(f"\n{'─'*60}")
    print(f"[{name}]  {ds_id}")

    try:
        kwargs = dict(split=split, streaming=streaming)
        if config:
            kwargs["name"] = config
        ds = load_dataset(ds_id, **kwargs)
    except Exception as e:
        print(f"  SKIP — load failed: {e}")
        return 0

    count = skipped = 0

    with open(out_path, "w") as f:
        for row in ds:
            if count >= max_n:
                break
            try:
                if fmt == "alpaca":
                    pair = format_alpaca(row, system)
                elif fmt == "qa":
                    q = row.get(cfg.get("q_field", "question"), "")
                    a = row.get(cfg.get("a_field", "answer"), "")
                    pair = conv(system, q, a)
                elif fmt == "messages":
                    pair = format_messages(row, cfg.get("msg_field", "messages"), system)
                elif fmt == "mmlu":
                    pair = format_mmlu(row, system)
                elif fmt == "squad":
                    pair = format_squad(row, system)
                elif fmt == "humaneval":
                    pair = format_humaneval(row, system)
                elif fmt == "code_text":
                    pair = format_code_text(row, cfg.get("text_field", "text"), system)
                elif fmt == "self_explain":
                    pair = format_self_explain(row, cfg.get("q_field", "text"), system)
                else:
                    pair = None

                if pair:
                    # self_explain override
                    if cfg.get("self_explain"):
                        pair = format_self_explain(row, cfg.get("q_field", "markdown"), system) or pair

                    f.write(json.dumps(pair) + "\n")
                    count += 1
                else:
                    skipped += 1
            except Exception:
                skipped += 1

    print(f"  {count:,} pairs written  ({skipped} skipped)")
    return count


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/hf_batch")
    parser.add_argument("--only",    nargs="*", help="Run only these dataset names")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = DATASETS
    if args.only:
        configs = [c for c in DATASETS if c["name"] in args.only]

    totals = {}
    for cfg in configs:
        n = ingest_one(cfg, out_dir)
        totals[cfg["name"]] = n

    print(f"\n{'═'*60}")
    print("SUMMARY")
    print(f"{'═'*60}")
    grand = 0
    for name, n in totals.items():
        print(f"  {name:40s} {n:>6,}")
        grand += n
    print(f"{'─'*60}")
    print(f"  {'TOTAL':40s} {grand:>6,}")

    files = sorted(out_dir.glob("*.jsonl"))
    merge_src = " ".join(str(f) for f in files)
    print(f"\nMerge all into sft_master.jsonl:")
    print(f"  cat data/sft_master.jsonl {merge_src} > /tmp/merged.jsonl && mv /tmp/merged.jsonl data/sft_master.jsonl")


if __name__ == "__main__":
    main()
