#!/usr/bin/env python3
"""
ingest_pharmacology.py — Pharmacology & clinical datasets for Anthos.

Output: data/pharmacology_sft.jsonl
Merge:  cat data/sft_master.jsonl data/pharmacology_sft.jsonl > /tmp/m.jsonl && mv /tmp/m.jsonl data/sft_master.jsonl
"""

import json
from pathlib import Path
from datasets import load_dataset

SYS = (
    "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
    "You explain pharmacology, clinical drug use, and medical science accurately. "
    "You connect drug mechanisms to clinical outcomes and patient safety."
)

OUTPUT = Path("data/pharmacology_sft.jsonl")
CHOICE_LABELS = list("ABCDEF")


def mc(q, a):
    q, a = q.strip(), a.strip()[:3500]
    if not q or not a or len(q) < 8 or len(a) < 8:
        return None
    return {"conversations": [
        {"from": "system", "value": SYS},
        {"from": "human",  "value": q},
        {"from": "gpt",    "value": a},
    ]}


def nursing_pharm(row):
    """nursing-pharmacology: question + op_a/b/c/d/e/f + cop/0 + exp"""
    q = str(row.get("question", "")).strip()
    if not q:
        return None

    opts = {}
    for lbl in "abcdef":
        v = str(row.get(f"op_{lbl}", "") or "").strip()
        if v and v != "nan":
            opts[lbl.upper()] = v

    if not opts:
        return None

    opts_str = "\n".join(f"{k}. {v}" for k, v in opts.items())
    question = f"{q}\n\n{opts_str}"

    # correct option index
    cop = row.get("cop/0") or row.get("cop")
    explanation = str(row.get("exp", "") or "").strip()

    if cop is not None:
        try:
            idx = int(cop) - 1  # 1-indexed
            lbl = list(opts.keys())[idx] if 0 <= idx < len(opts) else None
            if lbl and lbl in opts:
                answer = f"{lbl}. {opts[lbl]}"
                if explanation:
                    answer += f"\n\nExplanation: {explanation}"
                return mc(question, answer)
        except (ValueError, TypeError, IndexError):
            pass

    # fallback: just use explanation
    if explanation:
        return mc(question, explanation)
    return None


def pharm_llm_test(row):
    """Pharmacology-LLM-test-set: Name + Question + Description → drug Q&A"""
    name = str(row.get("Name", "") or "").strip()
    question = str(row.get("Question", "") or "").strip()
    description = str(row.get("Description", "") or "").strip()

    if not question:
        return None

    q = f"Regarding the drug {name}: {question}" if name else question

    # Use GPT-4 response if available
    for field in ("GPT_4_rep1", "GPT_4_rep2", "GPT_3.5_rep1"):
        a = str(row.get(field, "") or "").strip()
        if a and len(a) > 20:
            return mc(q, a)

    if description and len(description) > 30:
        return mc(q, description)
    return None


def pharm_qa(row):
    """Pharmacology-QA: question + choices (pre-formatted string) + label (full answer string)"""
    q = str(row.get("question", "") or "").strip()
    choices = str(row.get("choices", "") or "").strip()
    label = str(row.get("label", "") or "").strip()

    if not q or not label:
        return None

    question = f"{q}\n\n{choices}" if choices else q
    return mc(question, label)


DATASETS = [
    # MCQ nursing pharmacology — question + options + explanation
    ("nursing_pharm", "timzhou99/nursing-pharmacology", "train", 5000, False, nursing_pharm),
    # Drug interaction Q&A with GPT-4 answers
    ("pharm_llm_test", "zhangyingbo1984/Pharmacology-LLM-test-set", "train", 5000, False, pharm_llm_test),
    # MCQ with choices list + label index
    ("pharm_qa", "xuxuxuxuxu/Pharmacology-QA", "train", 5000, False, pharm_qa),
]


def main():
    all_pairs = []

    for name, ds_id, split, max_n, streaming, converter in DATASETS:
        print(f"\n[{name}]  {ds_id}")
        try:
            ds = load_dataset(ds_id, split=split, streaming=streaming)
            pairs = []
            for row in ds:
                if len(pairs) >= max_n:
                    break
                try:
                    p = converter(row)
                    if p:
                        pairs.append(p)
                except Exception:
                    pass
            print(f"  {len(pairs):,} pairs")
            all_pairs.extend(pairs)
        except Exception as e:
            print(f"  SKIP — {str(e)[:100]}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        for p in all_pairs:
            f.write(json.dumps(p) + "\n")

    print(f"\n{'═'*55}")
    print(f"Total: {len(all_pairs):,} pairs → {OUTPUT}")
    print(f"\nMerge:")
    print(f"  cat data/sft_master.jsonl {OUTPUT} > /tmp/m.jsonl && mv /tmp/m.jsonl data/sft_master.jsonl")


if __name__ == "__main__":
    main()
