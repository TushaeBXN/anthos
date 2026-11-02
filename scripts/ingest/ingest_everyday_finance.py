#!/usr/bin/env python3
"""
ingest_everyday_finance.py — Brian's Custom Datasets + Personal Finance

Local files (Brian's own hand-crafted training data):
  - /Users/dadsmacpro/Desktop/everyday_problem_solving_dataset.json
      Columnar format: Domain/Subtopic/Common Problem/Core Rule/Immediate Action
      Domains: Basic Cooking Rules, Household Maintenance, Digital Literacy,
               Workplace Etiquette
  - /Users/dadsmacpro/Desktop/everyday_problem_solving_dataset (1).json
      List format: domain/subtopic/problem/core_principle/immediate_action
      Domains: Cooking, Household, Digital, Workplace, Spatial Reasoning,
               Temporal Reasoning, Geography, Financial Literacy, First Aid,
               Mechanical Intuition

HuggingFace Personal Finance:
  - thegr8abdessamad/accounting02
  - Gamestatue/adaption-personal-finance-qa
  - Akhil-Theerthala/PersonalFinance_v2
  - Azfarhashmi/adaption-personal-finance-advice-dialogues
  - Gandalf1/personal-finance-sft-181k
  - Aletheia-ng/personal_finance_v0.2

Output: data/everyday_finance_sft.jsonl
Merge:  cat data/sft_master.jsonl data/everyday_finance_sft.jsonl > /tmp/m.jsonl && mv /tmp/m.jsonl data/sft_master.jsonl
"""

import json, re
from pathlib import Path
from datasets import load_dataset

OUTPUT = Path("data/everyday_finance_sft.jsonl")

LOCAL_FILE_1 = Path("/Users/dadsmacpro/Desktop/everyday_problem_solving_dataset.json")
LOCAL_FILE_2 = Path("/Users/dadsmacpro/Desktop/everyday_problem_solving_dataset (1).json")

SYS = {
    "everyday": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You give practical, clear, actionable advice for everyday problems. "
        "You explain the 'why' behind each solution so people truly understand, "
        "not just what to do but the principle behind it. "
        "You make knowledge that's usually only passed down through privilege or experience "
        "available to everyone."
    ),
    "finance": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You are a trusted financial guide for people who've been shut out of financial "
        "education — the communities that need it most. "
        "You explain money, debt, savings, investing, and financial planning in plain language "
        "with real, actionable steps. You are on the side of the people."
    ),
    "science": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You explain how things work — physics, mechanics, spatial reasoning — "
        "in plain, practical language that helps people solve real problems."
    ),
    "health": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You provide clear, accurate first aid and health information. "
        "You help people handle emergencies confidently until professional medical help arrives. "
        "Always recommend seeking professional medical attention for serious conditions."
    ),
}

DOMAIN_SYS = {
    "Basic Cooking Rules": "everyday",
    "Basic Cooking Rules & Kitchen Safety": "everyday",
    "Household Maintenance": "everyday",
    "Household Maintenance & Troubleshooting": "everyday",
    "Digital Literacy": "everyday",
    "Digital Literacy & Cybersecurity": "everyday",
    "Workplace Etiquette": "everyday",
    "Workplace Etiquette & Professionalism": "everyday",
    "Spatial Reasoning": "science",
    "Temporal Reasoning": "science",
    "Geography": "everyday",
    "Basic Financial Literacy": "finance",
    "First Aid": "health",
    "First Aid & Medical Emergencies": "health",
    "Mechanical Intuition": "science",
}


def mc(q, a, domain="everyday"):
    q, a = q.strip(), a.strip()[:4000]
    if not q or not a or len(q) < 8 or len(a) < 20:
        return None
    return {"conversations": [
        {"from": "system", "value": SYS.get(domain, SYS["everyday"])},
        {"from": "human",  "value": q},
        {"from": "gpt",    "value": a},
    ]}


def strip_think(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def is_english(text, threshold=0.75):
    sample = text[:300]
    if not sample:
        return False
    return sum(1 for c in sample if ord(c) < 128) / len(sample) >= threshold


# ── Local File 1: Columnar JSON ───────────────────────────────────────────────

def ingest_local_file1():
    """everyday_problem_solving_dataset.json — parallel arrays."""
    pairs = []
    if not LOCAL_FILE_1.exists():
        print(f"  SKIP — file not found: {LOCAL_FILE_1}")
        return pairs

    with open(LOCAL_FILE_1) as f:
        data = json.load(f)

    domains = data.get("Domain", [])
    subtopics = data.get("Subtopic", [])
    problems = data.get("Common Problem", [])
    principles = data.get("Core Rule / Principle", [])
    actions = data.get("Immediate Action Step", [])

    n = len(domains)
    for i in range(n):
        try:
            domain = domains[i] if i < len(domains) else ""
            subtopic = subtopics[i] if i < len(subtopics) else ""
            problem = problems[i] if i < len(problems) else ""
            principle = principles[i] if i < len(principles) else ""
            action = actions[i] if i < len(actions) else ""

            if not problem or not (principle or action):
                continue

            sys_key = DOMAIN_SYS.get(domain, "everyday")

            # Primary Q&A: problem → principle + action
            q = f"What should I do if: {problem}?"
            a_parts = []
            if principle:
                a_parts.append(f"**Why this happens:** {principle}")
            if action:
                a_parts.append(f"**What to do:** {action}")
            a = "\n\n".join(a_parts)
            p = mc(q, a, sys_key)
            if p:
                pairs.append(p)

            # Secondary Q&A: principle explanation
            if principle and subtopic:
                q2 = f"Explain the principle behind {subtopic.lower()} safety/best practices."
                p2 = mc(q2, principle, sys_key)
                if p2:
                    pairs.append(p2)

        except Exception:
            continue

    print(f"  local_file1: {len(pairs):,} pairs from {n} entries")
    return pairs


# ── Local File 2: List JSON ───────────────────────────────────────────────────

def ingest_local_file2():
    """everyday_problem_solving_dataset (1).json — list of objects."""
    pairs = []
    if not LOCAL_FILE_2.exists():
        print(f"  SKIP — file not found: {LOCAL_FILE_2}")
        return pairs

    with open(LOCAL_FILE_2) as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("  SKIP — unexpected format")
        return pairs

    seen = set()  # deduplicate by problem text

    for row in data:
        try:
            domain = str(row.get("domain", "") or "").strip()
            subtopic = str(row.get("subtopic", "") or "").strip()
            problem = str(row.get("problem", "") or "").strip()
            principle = str(row.get("core_principle", "") or "").strip()
            action = str(row.get("immediate_action", "") or "").strip()

            if not problem or problem in seen:
                continue
            seen.add(problem)

            if not (principle or action):
                continue

            sys_key = DOMAIN_SYS.get(domain, "everyday")

            # Primary: scenario → principle + action
            q = f"What should I do if: {problem}"
            if not q.endswith("?"):
                q += "?"
            a_parts = []
            if principle:
                a_parts.append(f"**Core principle:** {principle}")
            if action:
                a_parts.append(f"**Immediate action:** {action}")
            a = "\n\n".join(a_parts)
            p = mc(q, a, sys_key)
            if p:
                pairs.append(p)

            # Secondary: explain the principle
            if principle and subtopic:
                q2 = f"Explain the key principle for dealing with: {subtopic}."
                p2 = mc(q2, principle, sys_key)
                if p2:
                    pairs.append(p2)

            # Tertiary: give me the step-by-step action
            if action and subtopic:
                q3 = f"What are the immediate steps to take for: {subtopic}?"
                p3 = mc(q3, action, sys_key)
                if p3:
                    pairs.append(p3)

        except Exception:
            continue

    print(f"  local_file2: {len(pairs):,} pairs from {len(data)} entries")
    return pairs


# ── HuggingFace Personal Finance ──────────────────────────────────────────────

def conv_finance_qa(row):
    """Generic personal finance Q&A converter."""
    for qf, af in [("question", "answer"), ("instruction", "response"),
                   ("input", "output"), ("prompt", "completion"),
                   ("query", "response"), ("human", "assistant"),
                   ("Question", "Answer")]:
        q = str(row.get(qf, "") or "").strip()
        a = str(row.get(af, "") or "").strip()
        if q and a and len(a) > 15 and is_english(q):
            return mc(q, strip_think(a), "finance")
    # Messages list
    msgs = row.get("messages", row.get("conversations", row.get("conversation", [])))
    if isinstance(msgs, list):
        q = a = ""
        for m in msgs:
            role = str(m.get("role", m.get("from", ""))).lower()
            content = str(m.get("content", m.get("value", ""))).strip()
            if role in ("user", "human") and not q:
                q = content
            elif role in ("assistant", "gpt") and q and not a:
                a = strip_think(content)
        if q and a and is_english(q):
            return mc(q, a, "finance")
    # Text fallback
    text = str(row.get("text", "") or "").strip()
    if text and len(text) > 80 and is_english(text):
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) >= 2:
            return mc(f"Explain this personal finance concept: {lines[0][:150]}", "\n".join(lines[1:])[:2500], "finance")
    return None


HF_FINANCE = [
    {"name": "accounting02",            "id": "thegr8abdessamad/accounting02",                        "split": "train", "max": 5000},
    {"name": "personal_finance_qa",     "id": "Gamestatue/adaption-personal-finance-qa",              "split": "train", "max": 5000},
    {"name": "personal_finance_v2",     "id": "Akhil-Theerthala/PersonalFinance_v2",                  "split": "train", "max": 5000},
    {"name": "finance_advice_dialogues","id": "Azfarhashmi/adaption-personal-finance-advice-dialogues","split": "train", "max": 5000},
    {"name": "personal_finance_181k",   "id": "Gandalf1/personal-finance-sft-181k",                   "split": "train", "max": 5000},
    {"name": "personal_finance_v02",    "id": "Aletheia-ng/personal_finance_v0.2",                    "split": "train", "max": 5000},
]


def ingest_hf_finance():
    pairs = []
    for cfg in HF_FINANCE:
        name = cfg["name"]
        ds_id = cfg["id"]
        split = cfg["split"]
        max_pairs = cfg["max"]

        print(f"\n  [{name}]  {ds_id}")
        loaded = False
        first_err = ""
        for try_split in [split, "train", "test", "validation"]:
            try:
                ds = load_dataset(ds_id, split=try_split, streaming=True)
                batch, skipped = [], 0
                for row in ds:
                    if len(batch) >= max_pairs:
                        break
                    try:
                        p = conv_finance_qa(row)
                        if p:
                            batch.append(p)
                        else:
                            skipped += 1
                    except Exception:
                        skipped += 1
                suffix = f" [split={try_split}]" if try_split != split else ""
                print(f"    {len(batch):,} pairs  ({skipped} skipped){suffix}")
                pairs.extend(batch)
                loaded = True
                break
            except Exception as e:
                if not first_err:
                    first_err = str(e)[:100]
        if not loaded:
            print(f"    SKIP — {first_err}")
    return pairs


def main():
    all_pairs = []

    print("\n[LOCAL] everyday_problem_solving_dataset.json (columnar)")
    all_pairs.extend(ingest_local_file1())

    print("\n[LOCAL] everyday_problem_solving_dataset (1).json (list)")
    all_pairs.extend(ingest_local_file2())

    print("\n[HF] Personal Finance datasets")
    all_pairs.extend(ingest_hf_finance())

    # Deduplicate by question text
    seen_q = set()
    deduped = []
    for p in all_pairs:
        q = p["conversations"][1]["value"][:100]
        if q not in seen_q:
            seen_q.add(q)
            deduped.append(p)
    removed = len(all_pairs) - len(deduped)
    if removed:
        print(f"\n  Deduped {removed} duplicate questions")
    all_pairs = deduped

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
