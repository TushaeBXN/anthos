#!/usr/bin/env python3
"""
ingest_hf_batch2.py — Biology, Astronomy, Environmental Science, Computer Science datasets

Auto-detects field names so unknown datasets still get processed.
Caps all streaming datasets to prevent runaway downloads.

Usage:
    python ingest_hf_batch2.py
    python ingest_hf_batch2.py --only biology_hydra arxiv_biology
"""

import json
import argparse
from pathlib import Path

SYS_SCIENCE = (
    "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
    "You explain science concepts accurately, connecting theory to real-world meaning. "
    "You are direct and clear — you don't pad answers with unnecessary qualifications."
)

SYS_BIO = (
    "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
    "You explain biology clearly — from molecular mechanisms to ecosystems. "
    "You connect concepts to real-world application and make complex processes understandable."
)

SYS_ASTRO = (
    "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
    "You explain astronomy and astrophysics with accuracy and wonder. "
    "You connect cosmic phenomena to the physics and math behind them."
)

SYS_ENV = (
    "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
    "You explain environmental science, ecology, and environmental justice clearly. "
    "You connect environmental issues to policy, community impact, and action."
)

SYS_CS = (
    "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
    "You are an expert in computer science — theory, systems, algorithms, and applied computing. "
    "You explain concepts clearly and implement solutions completely when asked."
)

CHOICE_LABELS = ["A", "B", "C", "D", "E"]

# ─────────────────────────────────────────────────────────────────────────────
# Dataset configs
# ─────────────────────────────────────────────────────────────────────────────

DATASETS = [
    # ── Biology ──────────────────────────────────────────────────────────────
    {
        "name": "mmlu_college_chemistry2",
        "id": "joey234/mmlu-college_chemistry",
        "split": "test",
        "format": "mmlu",
        "max": 1000,
        "system": SYS_SCIENCE,
    },
    {
        "name": "mmlu_college_biology",
        "id": "joey234/mmlu-college_biology-verbal-neg-prepend",
        "split": "test",
        "format": "mmlu",
        "max": 1000,
        "system": SYS_BIO,
    },
    {
        "name": "mmlu_hs_biology",
        "id": "joey234/mmlu-high_school_biology-dev",
        "split": "dev",
        "format": "mmlu",
        "max": 1000,
        "system": SYS_BIO,
    },
    {
        "name": "arxiv_biology",
        "id": "zeroshot/arxiv-biology",
        "split": "train",
        "format": "auto",
        "max": 3000,
        "system": SYS_BIO,
        "streaming": True,
    },
    {
        "name": "opensci_biology",
        "id": "TerryJCZhang/OpenSciReasoning-Biology-20K",
        "split": "train",
        "format": "auto",
        "max": 5000,
        "system": SYS_BIO,
        "streaming": True,
    },
    {
        "name": "biology_how_why",
        "id": "bigbio/biology_how_why_corpus",
        "config": "biology_how_why_corpus_source",
        "split": "train",
        "format": "auto",
        "max": 3000,
        "system": SYS_BIO,
    },
    {
        "name": "biology_hydra",
        "id": "HydraLM/biology_dataset_standardized",
        "split": "train",
        "format": "auto",
        "max": 5000,
        "system": SYS_BIO,
    },
    {
        "name": "biology_vsi_qa",
        "id": "n3rd0/VSI_biology_QA",
        "split": "train",
        "format": "auto",
        "max": 3000,
        "system": SYS_BIO,
    },
    {
        "name": "biology_ap",
        "id": "vjain/biology_AP_embeddings",
        "split": "train",
        "format": "auto",
        "max": 3000,
        "system": SYS_BIO,
    },

    # ── Astronomy ─────────────────────────────────────────────────────────────
    {
        "name": "mmlu_astronomy",
        "id": "Lots-of-LoRAs/task666_mmmlu_answer_generation_astronomy",
        "split": "train",
        "format": "auto",
        "max": 3000,
        "system": SYS_ASTRO,
    },
    {
        "name": "astronomy_stack_dpo",
        "id": "David-Xu/astronomy-stack-dpo-text-20-percent",
        "split": "train",
        "format": "auto",
        "max": 5000,
        "system": SYS_ASTRO,
        "streaming": True,
    },
    {
        "name": "astronomy_stack_cira",
        "id": "David-Xu/astronomy-stack-cira",
        "split": "train",
        "format": "auto",
        "max": 5000,
        "system": SYS_ASTRO,
        "streaming": True,
    },
    {
        "name": "slimorca_astronomy",
        "id": "luozhuanggary/SlimOrca_astronomy_50k",
        "split": "train",
        "format": "auto",
        "max": 5000,
        "system": SYS_ASTRO,
        "streaming": True,
    },
    {
        "name": "astronomy_distiset",
        "id": "Kukedlc/my-distiset-astronomy-domine",
        "split": "train",
        "format": "auto",
        "max": 3000,
        "system": SYS_ASTRO,
        "streaming": True,
    },

    # ── Environmental Science ─────────────────────────────────────────────────
    {
        "name": "env_science",
        "id": "BaffledCoder/EnvironmentalScience",
        "split": "train",
        "format": "auto",
        "max": 5000,
        "system": SYS_ENV,
        "streaming": True,
    },
    {
        "name": "env_esgbert",
        "id": "ESGBERT/environmental_2k",
        "split": "train",
        "format": "auto",
        "max": 2000,
        "system": SYS_ENV,
    },
    {
        "name": "env_justice",
        "id": "neelsurya/environmentaljustice",
        "split": "train",
        "format": "auto",
        "max": 5000,
        "system": SYS_ENV,
        "streaming": True,
    },
    {
        "name": "env_10k",
        "id": "neelsurya/environmental10ktrainingcheck",
        "split": "train",
        "format": "auto",
        "max": 5000,
        "system": SYS_ENV,
    },
    {
        "name": "env_hazards",
        "id": "LeroyDyer/environmental_hazzards",
        "split": "train",
        "format": "auto",
        "max": 3000,
        "system": SYS_ENV,
    },
    {
        "name": "env_ethics",
        "id": "herronej/SciTrust2-Ethics-Environmental-Impact",
        "split": "train",
        "format": "auto",
        "max": 3000,
        "system": SYS_ENV,
    },
    {
        "name": "env_geology",
        "id": "NoraResearchLab/Environmental_geology-dem",
        "split": "train",
        "format": "auto",
        "max": 3000,
        "system": SYS_ENV,
    },

    # ── Computer Science ──────────────────────────────────────────────────────
    {
        "name": "cs_25k",
        "id": "WithinUsAI/Computer_Science_25k",
        "split": "train",
        "format": "auto",
        "max": 5000,
        "system": SYS_CS,
        "streaming": True,
    },
    {
        "name": "k12_cs",
        "id": "robworks-software/k12-computer-science-standards",
        "split": "train",
        "format": "auto",
        "max": 5000,
        "system": SYS_CS,
    },
    {
        "name": "cs_conversational",
        "id": "oss-codes/Computer-Science-Conversational-Dataset-Indic",
        "split": "train",
        "format": "auto",
        "max": 5000,
        "system": SYS_CS,
        "streaming": True,
    },
    {
        "name": "cs_pretraining",
        "id": "pritamdeb68/Computer-Science-Pretraining",
        "split": "train",
        "format": "auto",
        "max": 5000,
        "system": SYS_CS,
        "streaming": True,
    },
    {
        "name": "cs_aalen",
        "id": "Puidii/aalen_university_faculty_computer_science",
        "split": "train",
        "format": "auto",
        "max": 3000,
        "system": SYS_CS,
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Converters
# ─────────────────────────────────────────────────────────────────────────────

def make_conv(system, q, a):
    q, a = q.strip(), a.strip()[:3500]
    if not q or not a or len(q) < 8 or len(a) < 8:
        return None
    return {
        "conversations": [
            {"from": "system", "value": system},
            {"from": "human",  "value": q},
            {"from": "gpt",    "value": a},
        ]
    }

def format_mmlu(row, system):
    q       = row.get("question", "").strip()
    choices = row.get("choices", [])
    ans_idx = row.get("answer", -1)
    if not q or not choices:
        return None
    choices_str = "\n".join(f"{CHOICE_LABELS[i]}. {c}" for i, c in enumerate(choices))
    question = f"{q}\n\n{choices_str}"
    if isinstance(ans_idx, int) and 0 <= ans_idx < len(choices):
        answer = f"{CHOICE_LABELS[ans_idx]}. {choices[ans_idx]}"
    elif isinstance(ans_idx, str) and ans_idx in CHOICE_LABELS:
        idx = CHOICE_LABELS.index(ans_idx)
        answer = f"{ans_idx}. {choices[idx]}" if idx < len(choices) else ans_idx
    else:
        return None
    return make_conv(system, question, answer)

def try_messages(row, system):
    """Try to extract first human+assistant turn from various message formats."""
    for field in ("messages", "conversation", "conversations", "turns", "dialog"):
        msgs = row.get(field)
        if not msgs or not isinstance(msgs, list):
            continue
        q = a = ""
        for m in msgs:
            role = m.get("role", m.get("from", "")).lower()
            content = m.get("content", m.get("value", m.get("text", ""))).strip()
            if not content:
                continue
            if role in ("user", "human") and not q:
                q = content
            elif role in ("assistant", "gpt", "bot") and q and not a:
                a = content
                break
        if q and a:
            return make_conv(system, q, a)
    return None

def try_qa_fields(row, system):
    """Try common Q/A field name patterns."""
    q_candidates = ["question", "instruction", "input", "prompt", "query",
                    "problem", "task", "text", "title", "abstract"]
    a_candidates = ["answer", "output", "response", "solution", "completion",
                    "explanation", "body", "content", "reasoning"]

    q = a = ""
    for f in q_candidates:
        val = row.get(f, "")
        if isinstance(val, str) and val.strip():
            q = val.strip()
            break
    for f in a_candidates:
        val = row.get(f, "")
        if isinstance(val, str) and val.strip():
            a = val.strip()
            break

    # Avoid using the same field for both
    if q and a and q != a:
        return make_conv(system, q, a)

    # Single long text field — make it self-explanatory
    for f in ("text", "content", "body", "passage", "abstract"):
        val = row.get(f, "")
        if isinstance(val, str) and len(val.strip()) > 200:
            text = val.strip()
            snippet = text.splitlines()[0][:120]
            return make_conv(system, f"Explain this concept: {snippet}", text[:3000])

    return None

def auto_convert(row, system):
    """Try message format first, then Q/A fields."""
    pair = try_messages(row, system)
    if pair:
        return pair
    return try_qa_fields(row, system)

# ─────────────────────────────────────────────────────────────────────────────
# Ingest one dataset
# ─────────────────────────────────────────────────────────────────────────────

def ingest_one(cfg, out_dir):
    from datasets import load_dataset

    name      = cfg["name"]
    ds_id     = cfg["id"]
    split     = cfg.get("split", "train")
    config    = cfg.get("config", None)
    fmt       = cfg["format"]
    system    = cfg["system"]
    max_n     = cfg.get("max", 3000)
    streaming = cfg.get("streaming", False)

    out_path = out_dir / f"{name}.jsonl"
    print(f"\n{'─'*55}")
    print(f"[{name}]  {ds_id}")

    try:
        kwargs = dict(split=split, streaming=streaming)
        if config:
            kwargs["name"] = config
        ds = load_dataset(ds_id, **kwargs)
    except Exception as e:
        msg = str(e)[:120]
        print(f"  SKIP — {msg}")
        return 0

    count = skipped = 0

    with open(out_path, "w") as f:
        for row in ds:
            if count >= max_n:
                break
            try:
                if fmt == "mmlu":
                    pair = format_mmlu(row, system)
                else:
                    pair = auto_convert(row, system)

                if pair:
                    f.write(json.dumps(pair) + "\n")
                    count += 1
                else:
                    skipped += 1
            except Exception:
                skipped += 1

    print(f"  {count:,} pairs  ({skipped} skipped)")
    return count

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/hf_batch2")
    parser.add_argument("--only",    nargs="*")
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

    print(f"\n{'═'*55}")
    print("SUMMARY")
    grand = sum(totals.values())
    for name, n in totals.items():
        status = "✓" if n > 0 else "✗"
        print(f"  {status} {name:40s} {n:>5,}")
    print(f"{'─'*55}")
    print(f"  TOTAL: {grand:,}")

    files = [out_dir / f"{n}.jsonl" for n, c in totals.items() if c > 0]
    if files:
        merge_src = " ".join(str(f) for f in files)
        print(f"\nMerge command:")
        print(f"  cat data/sft_master.jsonl {merge_src} > /tmp/m.jsonl && mv /tmp/m.jsonl data/sft_master.jsonl")


if __name__ == "__main__":
    main()
