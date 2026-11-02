#!/usr/bin/env python3
"""
ingest_medicine_health.py — Medicine, health, mental health, and human biology datasets.

Cap at 5,000 per dataset. Auto-detects format with custom converters for known schemas.

Output: data/medicine_health_sft.jsonl
Merge:  cat data/sft_master.jsonl data/medicine_health_sft.jsonl > /tmp/m.jsonl && mv /tmp/m.jsonl data/sft_master.jsonl
"""

import json
import re
from pathlib import Path
from datasets import load_dataset

SYS_MED = (
    "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
    "You explain medical and pharmaceutical concepts accurately. "
    "You connect clinical knowledge to patient care and real-world outcomes."
)

SYS_HEALTH = (
    "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
    "You give clear, evidence-based health advice. "
    "You explain anatomy, physiology, and wellness with precision and empathy."
)

SYS_MENTAL = (
    "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
    "You discuss mental health with compassion and clinical accuracy. "
    "You normalize mental health struggles and explain evidence-based approaches to care."
)

OUTPUT = Path("data/medicine_health_sft.jsonl")
CHOICE_LABELS = list("ABCDE")


def mc(q, a, system=SYS_MED):
    q, a = q.strip(), a.strip()[:4000]
    if not q or not a or len(q) < 8 or len(a) < 12:
        return None
    return {"conversations": [
        {"from": "system", "value": system},
        {"from": "human",  "value": q},
        {"from": "gpt",    "value": a},
    ]}


def auto(row, system=SYS_MED):
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
            return mc(q, a, system)

    for qf in ("instruction", "question", "input", "prompt", "query", "title"):
        for af in ("output", "answer", "response", "reference_answer", "solution", "content", "body"):
            q = str(row.get(qf, "") or "").strip()
            a = str(row.get(af, "") or "").strip()
            if q and a and q != a and len(a) > 20:
                return mc(q, a, system)

    for tf in ("text", "content", "description", "body"):
        t = str(row.get(tf, "") or "").strip()
        if len(t) > 200:
            first = t.splitlines()[0][:120]
            return mc(f"Explain this medical concept: {first}", t[:3500], system)

    return None


# ─── Custom converters ────────────────────────────────────────────────────────

def conv_mmlu(row, system=SYS_MED):
    q = str(row.get("question", "") or "").strip()
    choices = row.get("choices", [])
    answer = row.get("answer")
    if not q or not isinstance(choices, list) or not choices:
        return None
    opts = "\n".join(f"{CHOICE_LABELS[i]}. {c}" for i, c in enumerate(choices) if i < 5)
    question = f"{q}\n\n{opts}"
    if isinstance(answer, int) and 0 <= answer < len(choices):
        a = f"{CHOICE_LABELS[answer]}. {choices[answer]}"
    elif isinstance(answer, str) and answer in CHOICE_LABELS:
        idx = CHOICE_LABELS.index(answer)
        a = f"{answer}. {choices[idx]}" if idx < len(choices) else answer
    else:
        return None
    return mc(question, a, system)


def conv_rar(row):
    """RaR reasoning: question + reference_answer."""
    q = str(row.get("question", "") or row.get("prompt", "") or "").strip()
    a = str(row.get("reference_answer", "") or row.get("answer", "") or row.get("output", "") or "").strip()
    if not q or not a:
        return None
    return mc(q, a)


def conv_medicines_catalog(row):
    """kaysss/Medicines: disease_name + med_name + drug_content etc."""
    disease = str(row.get("disease_name", "") or "").strip()
    med = str(row.get("med_name", "") or "").strip()
    generic = str(row.get("generic_name", "") or "").strip()
    content = str(row.get("drug_content", "") or "").strip()
    manufacturer = str(row.get("drug_manufacturer", "") or "").strip()
    if not disease or not med:
        return None
    q = f"What medicines are used to treat {disease}?"
    parts = [f"**{med}**"]
    if generic:
        parts.append(f"Generic name: {generic}")
    if content:
        parts.append(f"Active ingredients: {content}")
    if manufacturer:
        parts.append(f"Manufacturer: {manufacturer}")
    return mc(q, "\n".join(parts))


def conv_medicine_dataset2(row):
    """zenitsu09/medicine-dataset-2: name + use0-4 + sideEffect0-41."""
    name = str(row.get("name", "") or "").strip()
    if not name:
        return None
    uses = [str(row.get(f"use{i}", "") or "").strip() for i in range(5)]
    uses = [u for u in uses if u and u.lower() not in ("nan", "none", "")]
    sides = [str(row.get(f"sideEffect{i}", "") or "").strip() for i in range(42)]
    sides = [s for s in sides if s and s.lower() not in ("nan", "none", "")]
    subs = [str(row.get(f"substitute{i}", "") or "").strip() for i in range(5)]
    subs = [s for s in subs if s and s.lower() not in ("nan", "none", "")]
    if not uses and not sides:
        return None
    q = f"What are the uses and side effects of {name}?"
    parts = []
    if uses:
        parts.append("**Uses:**\n" + "\n".join(f"- {u}" for u in uses))
    if sides:
        parts.append("**Side Effects:**\n" + "\n".join(f"- {s}" for s in sides[:15]))
    if subs:
        parts.append("**Substitutes:**\n" + "\n".join(f"- {s}" for s in subs))
    tc = str(row.get("Therapeutic Class", "") or "").strip()
    if tc and tc.lower() not in ("nan", "none"):
        parts.append(f"**Therapeutic Class:** {tc}")
    return mc(q, "\n\n".join(parts))


def conv_medicine_ansh(row):
    """Ansh99/medicine-dataset: Medicine_Name + structured fields."""
    name = str(row.get("Medicine_Name", "") or "").strip()
    ingredients = str(row.get("Active_Ingredients", "") or "").strip()
    uses = str(row.get("Primary_Use_Cases", "") or "").strip()
    dosage = str(row.get("Dosage_Instructions", "") or "").strip()
    sides = str(row.get("Side_Effects", "") or "").strip()
    if not name:
        return None
    q = f"Explain the medicine {name}: its uses, dosage, and side effects."
    parts = []
    if ingredients:
        parts.append(f"**Active Ingredients:** {ingredients}")
    if uses:
        parts.append(f"**Primary Uses:** {uses}")
    if dosage:
        parts.append(f"**Dosage:** {dosage}")
    if sides:
        parts.append(f"**Side Effects:** {sides}")
    if not parts:
        return None
    return mc(q, "\n\n".join(parts))


def conv_chatdoctor(row):
    """ChatDoctor: instruction (system context) + input (patient Q) + output (doctor A)."""
    instruction = str(row.get("instruction", "") or "").strip()
    patient_q = str(row.get("input", "") or "").strip()
    answer = str(row.get("output", "") or "").strip()
    if not answer or len(answer) < 20:
        return None
    q = patient_q if patient_q else instruction
    if not q:
        return None
    return mc(q, answer, SYS_HEALTH)


def conv_mental_chatbot(row):
    """mental_health_chatbot_dataset: text with <HUMAN>:/<ASSISTANT>: markers."""
    text = str(row.get("text", "") or "").strip()
    if not text:
        return None
    # Try to split on <HUMAN>: and <ASSISTANT>: tags
    human_match = re.search(r'<HUMAN>:\s*(.*?)(?=<ASSISTANT>:|$)', text, re.DOTALL)
    asst_match = re.search(r'<ASSISTANT>:\s*(.*?)(?=<HUMAN>:|$)', text, re.DOTALL)
    if human_match and asst_match:
        q = human_match.group(1).strip()
        a = asst_match.group(1).strip()
        return mc(q, a, SYS_MENTAL)
    # Fallback
    if len(text) > 100:
        return mc("Mental health question:", text[:3000], SYS_MENTAL)
    return None


def conv_reddit_mental(row):
    """reddit_mental_health_posts / mental_health_dataset_1: title + body."""
    title = str(row.get("title", "") or "").strip()
    body = str(row.get("body", "") or row.get("selftext", "") or "").strip()
    if not body or len(body) < 80:
        return None
    if body.lower() in ("[deleted]", "[removed]", ""):
        return None
    q_text = title if title else body.splitlines()[0][:120]
    sub = str(row.get("subreddit", "") or "").strip()
    context = f"This was posted in r/{sub}: {q_text}" if sub else q_text
    return mc(
        f"Someone shared this about their mental health: {context}\n\nHow should we understand this?",
        body[:3000], SYS_MENTAL
    )


def conv_anatomy(row):
    q = str(row.get("question", "") or "").strip()
    a = str(row.get("answer", "") or "").strip()
    diff = str(row.get("difficulty", "") or "").strip()
    if not q or not a:
        return None
    if diff:
        q = f"[{diff.capitalize()}] {q}"
    return mc(q, a, SYS_HEALTH)


def conv_human_rights(row):
    sys_law = (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You explain legal concepts, court decisions, and human rights law clearly and accurately."
    )
    for tf in ("text", "content", "decision", "judgment", "body", "full_text"):
        t = str(row.get(tf, "") or "").strip()
        if len(t) > 200:
            first = t.splitlines()[0][:120]
            return mc(f"Explain this European Court of Human Rights decision: {first}", t[:3500], sys_law)
    for qf in ("title", "case_name", "name"):
        for af in ("summary", "description", "text", "content"):
            q = str(row.get(qf, "") or "").strip()
            a = str(row.get(af, "") or "").strip()
            if q and a and len(a) > 30:
                return mc(f"Summarize this ECHR case: {q}", a, sys_law)
    return auto(row, sys_law)


# ─── Dataset registry ──────────────────────────────────────────────────────────

DATASETS = [
    # ── Medicine ──────────────────────────────────────────────────────────────
    {"name": "medicines_catalog",    "id": "kaysss/Medicines",
     "split": "train", "max": 5000, "streaming": True,  "conv": conv_medicines_catalog},
    {"name": "rar_medicine_20k",     "id": "anisha2102/RaR-Medicine-20k-o3-mini",
     "split": "train", "max": 5000, "streaming": True,  "conv": conv_rar},
    {"name": "mmlu_college_medicine","id": "joey234/mmlu-college_medicine-neg",
     "split": "test",  "max": 5000, "streaming": False, "conv": lambda r: conv_mmlu(r, SYS_MED)},
    {"name": "medicine_dataset2",    "id": "zenitsu09/medicine-dataset-2",
     "split": "train", "max": 5000, "streaming": True,  "conv": conv_medicine_dataset2},
    {"name": "baai_health_medicine", "id": "BAAI/IndustryInstruction_Health-Medicine",
     "split": "train", "max": 5000, "streaming": True,  "conv": lambda r: auto(r, SYS_MED)},
    {"name": "herbal_medicine_africa","id": "electricsheepafrica/africa-synth-herbal-traditional-medicine-safety-all",
     "split": "validation", "max": 5000, "streaming": True, "conv": lambda r: auto(r, SYS_MED)},
    {"name": "medicine_ansh",        "id": "Ansh99/medicine-dataset",
     "split": "train", "max": 5000, "streaming": True,  "conv": conv_medicine_ansh},

    # ── Health & Medical Q&A ───────────────────────────────────────────────
    {"name": "medical_meadow_health_advice", "id": "medalpaca/medical_meadow_health_advice",
     "split": "train", "max": 5000, "streaming": True,  "conv": lambda r: auto(r, SYS_HEALTH)},
    {"name": "chatdoctor_100k",       "id": "lavita/ChatDoctor-HealthCareMagic-100k",
     "split": "train", "max": 5000, "streaming": True,  "conv": conv_chatdoctor},
    {"name": "anatomy_qa",            "id": "ekplatebiryani/human_anatomy_qa_with_difficulty",
     "split": "train", "max": 5000, "streaming": True,  "conv": conv_anatomy},
    {"name": "medical_eval_humanity", "id": "meoconxinhxan/Medical-Eval-HumanityLastExam",
     "split": "train", "max": 5000, "streaming": True,  "conv": lambda r: auto(r, SYS_MED)},
    {"name": "human_assistant_medical","id": "erkamd/human_assistant_medical",
     "split": "train", "max": 5000, "streaming": True,  "conv": lambda r: auto(r, SYS_MED)},
    {"name": "nicolybgs_healthcare",  "id": "Nicolybgs/healthcare_data",
     "split": "train", "max": 5000, "streaming": True,  "conv": lambda r: auto(r, SYS_HEALTH)},

    # ── Mental Health ──────────────────────────────────────────────────────
    {"name": "reddit_mental_health",  "id": "solomonk/reddit_mental_health_posts",
     "split": "train", "max": 5000, "streaming": True,  "conv": conv_reddit_mental},
    {"name": "synthetic_mental_convos","id": "hllzmz/synthetic-mental-health-convos",
     "split": "train", "max": 5000, "streaming": True,  "conv": lambda r: auto(r, SYS_MENTAL)},
    {"name": "mental_health_1",       "id": "quocanh34/mental_health_dataset_1",
     "split": "train", "max": 5000, "streaming": True,  "conv": conv_reddit_mental},
    {"name": "mental_health_chatbot", "id": "heliosbrahma/mental_health_chatbot_dataset",
     "split": "train", "max": 5000, "streaming": True,  "conv": conv_mental_chatbot},

    # ── Human (Q&A + Legal) ────────────────────────────────────────────────
    {"name": "instruct_human_assistant","id": "Dahoas/instruct-human-assistant-prompt",
     "split": "train", "max": 5000, "streaming": True,  "conv": lambda r: auto(r)},
    {"name": "human_assistant_convo", "id": "Isotonic/human_assistant_conversation",
     "split": "train", "max": 5000, "streaming": True,  "conv": lambda r: auto(r)},
    {"name": "eu_human_rights",       "id": "roslein/EU_Court_Human_Rights_Decisions",
     "split": "train", "max": 5000, "streaming": True,  "conv": conv_human_rights},
    {"name": "human_vs_machine",      "id": "NicolaiSivesind/human-vs-machine",
     "split": "train", "max": 3000, "streaming": True,  "conv": lambda r: auto(r)},
    {"name": "humaneval_multilingual", "id": "ellamind/humaneval-multilingual",
     "split": "train", "max": 3000, "streaming": True,  "conv": lambda r: auto(r)},
    {"name": "multi_domain_ai_human", "id": "acmc/multi_domain_ai_human_text",
     "split": "train", "max": 3000, "streaming": True,  "conv": lambda r: auto(r)},
    {"name": "arena_human_preference","id": "lmarena-ai/arena-human-preference-140k",
     "split": "train", "max": 5000, "streaming": True,  "conv": lambda r: auto(r)},
]


def main():
    all_pairs = []

    for cfg in DATASETS:
        name = cfg["name"]
        ds_id = cfg["id"]
        print(f"\n[{name}]  {ds_id}")
        try:
            ds = load_dataset(ds_id, split=cfg["split"], streaming=cfg["streaming"])
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
