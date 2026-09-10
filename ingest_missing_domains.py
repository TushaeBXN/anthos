#!/usr/bin/env python3
"""
ingest_missing_domains.py — Fill every gap in the six-domain knowledge plan.

Covers: Philosophy/Ethics, Economics, Business, Law/Rights, Geography,
        Arts/Culture, Psychology/Sociology, Personal Finance, Everyday/Civic.

Output: data/missing_domains_sft.jsonl
Merge:  cat data/sft_master.jsonl data/missing_domains_sft.jsonl > /tmp/m.jsonl && mv /tmp/m.jsonl data/sft_master.jsonl
"""

import json
from pathlib import Path
from datasets import load_dataset

# ── System prompts by domain ──────────────────────────────────────────────────

SYS = {
    "general": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You give clear, accurate, helpful answers. You explain things in plain language "
        "so anyone can understand — not just experts."
    ),
    "philosophy": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You explain philosophical concepts, ethical frameworks, and moral reasoning clearly. "
        "You connect abstract ideas to real decisions people face."
    ),
    "economics": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You explain economics in plain language — how money, markets, and policy affect "
        "real people's lives. You cut through jargon."
    ),
    "business": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You explain business, finance, and accounting clearly. "
        "You help people understand how to build, run, and grow something of their own."
    ),
    "law": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You explain legal concepts in plain language. You help people understand their rights — "
        "tenant rights, consumer rights, employment rights, and how legal systems work. "
        "You are not a lawyer and always recommend consulting one for specific situations."
    ),
    "finance": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You explain personal finance clearly — credit, budgeting, taxes, investing, banking. "
        "You focus on practical steps anyone can take, especially people starting from nothing."
    ),
    "psychology": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You explain psychology, human behavior, and social dynamics clearly. "
        "You help people understand themselves and others better."
    ),
    "geography": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You explain geography, world cultures, and how place shapes people's lives. "
        "You connect location to history, economy, and opportunity."
    ),
    "arts": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You explain art, music, literature, and culture with enthusiasm and depth. "
        "You connect creative work to the human experiences that produced it."
    ),
    "civic": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You help people navigate real systems — benefits, housing, healthcare, education. "
        "You give practical, plain-language answers that actually help someone take action. "
        "You especially serve people who don't have access to expensive professionals."
    ),
}

OUTPUT = Path("data/missing_domains_sft.jsonl")
CHOICE_LABELS = list("ABCDE")


def mc(q, a, domain="general"):
    q, a = q.strip(), a.strip()[:4000]
    if not q or not a or len(q) < 8 or len(a) < 12:
        return None
    return {"conversations": [
        {"from": "system", "value": SYS[domain]},
        {"from": "human",  "value": q},
        {"from": "gpt",    "value": a},
    ]}


def conv_mmlu(row, domain="general"):
    q = str(row.get("question", "") or "").strip()
    choices = row.get("choices", [])
    answer = row.get("answer")
    if not q or not isinstance(choices, list) or not choices:
        return None
    opts = "\n".join(f"{CHOICE_LABELS[i]}. {c}" for i, c in enumerate(choices) if i < 5)
    question = f"{q}\n\n{opts}"
    if isinstance(answer, int) and 0 <= answer < len(choices):
        a = f"{CHOICE_LABELS[answer]}. {choices[answer]}"
    elif isinstance(answer, str):
        lbl = answer.strip()
        if lbl in CHOICE_LABELS:
            idx = CHOICE_LABELS.index(lbl)
            a = f"{lbl}. {choices[idx]}" if idx < len(choices) else lbl
        else:
            a = lbl
    else:
        return None
    return mc(question, a, domain)


def auto(row, domain="general"):
    for field in ("messages", "conversations", "conversation"):
        msgs = row.get(field)
        if not isinstance(msgs, list):
            continue
        q = a = ""
        for m in msgs:
            role = m.get("role", m.get("from", "")).lower()
            content = m.get("content", m.get("value", "")).strip()
            if role in ("user", "human") and not q:
                q = content
            elif role in ("assistant", "gpt") and q and not a:
                a = content
                break
        if q and a:
            return mc(q, a, domain)
    for qf in ("instruction", "question", "input", "prompt", "title", "query"):
        for af in ("output", "answer", "response", "reference_answer", "body", "text"):
            q = str(row.get(qf, "") or "").strip()
            a = str(row.get(af, "") or "").strip()
            if q and a and q != a and len(a) > 20:
                return mc(q, a, domain)
    for tf in ("text", "content", "description"):
        t = str(row.get(tf, "") or "").strip()
        if len(t) > 200:
            first = t.splitlines()[0][:120]
            return mc(f"Explain: {first}", t[:3500], domain)
    return None


def conv_law_stackexchange(row):
    """law-stack-exchange: title (question) + body (answer)."""
    title = str(row.get("title", "") or "").strip()
    body = str(row.get("body", "") or "").strip()
    if not title or not body or len(body) < 50:
        return None
    # Strip HTML tags roughly
    import re
    body = re.sub(r'<[^>]+>', ' ', body).strip()
    body = re.sub(r'\s+', ' ', body)
    if len(body) < 50:
        return None
    return mc(title, body[:3500], "law")


# ── MMLU configs mapped to domains ────────────────────────────────────────────

MMLU_CONFIGS = [
    # Philosophy & Ethics
    ("philosophy",                      "philosophy"),
    ("moral_scenarios",                 "philosophy"),
    ("moral_disputes",                  "philosophy"),
    ("logical_fallacies",               "philosophy"),
    ("formal_logic",                    "philosophy"),
    ("business_ethics",                 "philosophy"),
    ("world_religions",                 "arts"),

    # Economics
    ("high_school_macroeconomics",      "economics"),
    ("high_school_microeconomics",      "economics"),
    ("econometrics",                    "economics"),

    # Business & Accounting
    ("marketing",                       "business"),
    ("management",                      "business"),
    ("professional_accounting",         "business"),

    # Law & Government
    ("international_law",               "law"),
    ("jurisprudence",                   "law"),
    ("professional_law",                "law"),
    ("high_school_government_and_politics", "law"),

    # Psychology & Sociology
    ("high_school_psychology",          "psychology"),
    ("professional_psychology",         "psychology"),
    ("sociology",                       "psychology"),
    ("human_sexuality",                 "psychology"),

    # History & Culture (filling gaps)
    ("high_school_us_history",          "arts"),
    ("high_school_european_history",    "arts"),
    ("prehistory",                      "arts"),
    ("human_aging",                     "psychology"),

    # Geography
    ("high_school_geography",           "geography"),
    ("global_facts",                    "geography"),

    # Medical gaps
    ("nutrition",                       "general"),
    ("clinical_knowledge",              "general"),
    ("medical_genetics",                "general"),
    ("professional_medicine",           "general"),
    ("virology",                        "general"),
    ("anatomy",                         "general"),
    ("miscellaneous",                   "general"),
    ("global_facts",                    "geography"),
    ("security_studies",                "law"),
    ("public_relations",                "business"),
]

# ── Non-MMLU datasets ─────────────────────────────────────────────────────────

DATASETS = [
    # Personal Finance
    {
        "name": "finance_alpaca",
        "id": "gbharti/finance-alpaca",
        "split": "train", "max": 5000, "streaming": True,
        "conv": lambda r: auto(r, "finance"),
    },
    {
        "name": "fingpt_fiqa",
        "id": "FinGPT/fingpt-fiqa_qa",
        "split": "train", "max": 5000, "streaming": True,
        "conv": lambda r: auto(r, "finance"),
    },

    # Law Q&A
    {
        "name": "law_stack_exchange",
        "id": "jonathanli/law-stack-exchange",
        "split": "train", "max": 5000, "streaming": True,
        "conv": conv_law_stackexchange,
    },

    # Everyday / Civic knowledge
    {
        "name": "open_platypus",
        "id": "garage-bAInd/Open-Platypus",
        "split": "train", "max": 5000, "streaming": True,
        "conv": lambda r: auto(r, "general"),
    },
    {
        "name": "dolly_15k",
        "id": "databricks/databricks-dolly-15k",
        "split": "train", "max": 5000, "streaming": True,
        "conv": lambda r: auto(r, "general"),
    },
    {
        "name": "flan_v2",
        "id": "Muennighoff/flan",
        "split": "train", "max": 5000, "streaming": True,
        "conv": lambda r: auto(r, "general"),
    },

    # Geography & World
    {
        "name": "world_knowledge",
        "id": "Lots-of-LoRAs/task1156_country_capital",
        "split": "train", "max": 3000, "streaming": True,
        "conv": lambda r: auto(r, "geography"),
    },

    # Philosophy
    {
        "name": "philosophy_se",
        "id": "nicholasKluge/philosophyQA",
        "split": "train", "max": 5000, "streaming": False,
        "conv": lambda r: auto(r, "philosophy"),
    },

    # Arts & Culture
    {
        "name": "music_theory",
        "id": "nielsr/music-theory-qa",
        "split": "train", "max": 3000, "streaming": True,
        "conv": lambda r: auto(r, "arts"),
    },
]


def ingest_mmlu_batch(all_pairs, max_per_config=2000):
    """Stream all MMLU configs through lighteval/mmlu."""
    seen = set()
    for config_name, domain in MMLU_CONFIGS:
        if config_name in seen:
            continue
        seen.add(config_name)
        print(f"\n  [mmlu/{config_name}]")
        try:
            ds = load_dataset("lighteval/mmlu", config_name, split="test", streaming=False)
            pairs = []
            for row in ds:
                if len(pairs) >= max_per_config:
                    break
                p = conv_mmlu(row, domain)
                if p:
                    pairs.append(p)
            print(f"    {len(pairs):,} pairs")
            all_pairs.extend(pairs)
        except Exception as e:
            # Try 'validation' split
            try:
                ds = load_dataset("lighteval/mmlu", config_name, split="validation", streaming=False)
                pairs = []
                for row in ds:
                    if len(pairs) >= max_per_config:
                        break
                    p = conv_mmlu(row, domain)
                    if p:
                        pairs.append(p)
                print(f"    {len(pairs):,} pairs (validation split)")
                all_pairs.extend(pairs)
            except Exception as e2:
                print(f"    SKIP — {str(e2)[:70]}")


def main():
    all_pairs = []

    # ── MMLU batch ─────────────────────────────────────────────────────────────
    print("\n=== MMLU configs (lighteval/mmlu) ===")
    ingest_mmlu_batch(all_pairs, max_per_config=2000)

    # ── Non-MMLU datasets ──────────────────────────────────────────────────────
    print("\n=== Targeted datasets ===")
    for cfg in DATASETS:
        print(f"\n[{cfg['name']}]  {cfg['id']}")
        try:
            ds = load_dataset(cfg["id"], split=cfg["split"], streaming=cfg["streaming"])
            pairs = []
            skipped = 0
            for row in ds:
                if len(pairs) >= cfg["max"]:
                    break
                try:
                    p = cfg["conv"](row)
                    if p:
                        pairs.append(p)
                    else:
                        skipped += 1
                except Exception:
                    skipped += 1
            print(f"  {len(pairs):,} pairs  ({skipped} skipped)")
            all_pairs.extend(pairs)
        except Exception as e:
            print(f"  SKIP — {str(e)[:100]}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        for p in all_pairs:
            f.write(json.dumps(p) + "\n")

    print(f"\n{'═'*60}")
    print(f"Total: {len(all_pairs):,} pairs → {OUTPUT}")
    print(f"\nMerge:")
    print(f"  cat data/sft_master.jsonl {OUTPUT} > /tmp/m.jsonl && mv /tmp/m.jsonl data/sft_master.jsonl")


if __name__ == "__main__":
    main()
