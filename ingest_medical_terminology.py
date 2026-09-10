#!/usr/bin/env python3
"""
ingest_medical_terminology.py — Medical terminology and public health datasets.

Output: data/medical_terminology_sft.jsonl
Merge:  cat data/sft_master.jsonl data/medical_terminology_sft.jsonl > /tmp/m.jsonl && mv /tmp/m.jsonl data/sft_master.jsonl
"""

import json
import re
from pathlib import Path
from datasets import load_dataset

SYS = (
    "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
    "You explain medical terminology, clinical concepts, and public health topics clearly and accurately. "
    "You connect medical language to its practical meaning for patients and practitioners."
)

OUTPUT = Path("data/medical_terminology_sft.jsonl")


def mc(q, a):
    q, a = q.strip(), a.strip()[:3500]
    if not q or not a or len(q) < 8 or len(a) < 12:
        return None
    return {"conversations": [
        {"from": "system", "value": SYS},
        {"from": "human",  "value": q},
        {"from": "gpt",    "value": a},
    ]}


def parse_llama2_text(text):
    """Extract instruction and response from Llama2 format."""
    # <s>[INST] <<SYS>>\nsystem\n<</SYS>>\n\nhuman[/INST] response</s>
    inst_match = re.search(r'\[INST\].*?\[/INST\]\s*(.*?)(?:</s>|$)', text, re.DOTALL)
    if not inst_match:
        return None, None
    response = inst_match.group(1).strip()
    # Get the human part (after <<SYS>>...</SYS>> or just after [INST])
    human_match = re.search(r'<</SYS>>\s*\n+(.*?)\[/INST\]', text, re.DOTALL)
    if human_match:
        human = human_match.group(1).strip()
    else:
        human_match = re.search(r'\[INST\](.*?)\[/INST\]', text, re.DOTALL)
        human = human_match.group(1).strip() if human_match else ""
        human = re.sub(r'<<SYS>>.*?<</SYS>>', '', human, flags=re.DOTALL).strip()
    return human, response


def try_messages(row):
    msgs = row.get("messages", [])
    if not isinstance(msgs, list):
        return None
    q = a = ""
    for m in msgs:
        role = m.get("role", m.get("from", "")).lower()
        content = m.get("content", m.get("value", "")).strip()
        if role in ("user", "human") and not q:
            q = content
        elif role in ("assistant", "gpt") and q and not a:
            a = content
            break
    return mc(q, a) if q and a else None


# ─── Dataset-specific converters ────────────────────────────────────────────

def conv_autotrain_text(row):
    """autotrain_text = medical definition snippet — wrap as 'explain this term'."""
    text = str(row.get("autotrain_text", "") or "").strip()
    if len(text) < 30:
        return None
    title = str(row.get("title", "") or "").strip()
    if title:
        q = f"Explain the medical term or concept: {title}"
    else:
        first_line = text.splitlines()[0][:100]
        q = f"Explain this medical concept: {first_line}"
    return mc(q, text)


def conv_mcq(row):
    """medical-terminology-mcq: question + choices list + answer_text + rationale."""
    q = str(row.get("question", "") or "").strip()
    choices = row.get("choices", [])
    answer_text = str(row.get("answer_text", "") or "").strip()
    rationale = str(row.get("rationale", "") or "").strip()

    if not q:
        return None

    if isinstance(choices, list) and choices:
        opts = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
        question = f"{q}\n\n{opts}"
    else:
        question = q

    if answer_text and rationale:
        answer = f"{answer_text}\n\n{rationale}"
    elif answer_text:
        answer = answer_text
    elif rationale:
        answer = rationale
    else:
        return None

    return mc(question, answer)


def conv_llama2_text(row):
    """wiki_medical_terms_llama2: parse Llama2 format text."""
    text = str(row.get("text", "") or "").strip()
    human, response = parse_llama2_text(text)
    if not human or not response:
        return None
    return mc(human, response)


def conv_wiki_term(row):
    """wiki-medical-terms: medical_term + wiki_description."""
    term = str(row.get("medical_term", "") or "").strip()
    desc = str(row.get("wiki_description", "") or "").strip()
    if not term or not desc or len(desc) < 30:
        return None
    return mc(f"What is {term}? Explain this medical term.", desc)


def conv_inout(row):
    """wiki_medical_terms_inout: instruction + input + output."""
    instruction = str(row.get("instruction", "") or "").strip()
    inp = str(row.get("input", "") or "").strip()
    output = str(row.get("output", "") or "").strip()
    if not output:
        return None
    q = f"{instruction}\n\n{inp}".strip() if inp else instruction
    return mc(q, output)


def conv_pub_health_instruct(row):
    """Instruction-public-health-dataset: instructions + outputs."""
    q = str(row.get("instructions", "") or "").strip()
    a = str(row.get("outputs", "") or "").strip()
    return mc(q, a) if q and a else None


# ─── Dataset registry ────────────────────────────────────────────────────────

DATASETS = [
    # Medical terminology — autotrain Zephyr format (medical glossary entries)
    {
        "name": "med_term_zephyr1",
        "id": "pseudolab/autotrain-data-Medical_Terminology_Zephyr",
        "split": "train", "max": 5000, "streaming": True,
        "conv": conv_autotrain_text,
    },
    {
        "name": "med_term_zephyr2",
        "id": "pseudolab/autotrain-data-Medical_Terminology_Zephyr_2",
        "split": "train", "max": 5000, "streaming": True,
        "conv": conv_autotrain_text,
    },
    # MCQ with rationale — high quality
    {
        "name": "med_term_mcq_test",
        "id": "chongpangnasilemak/medical-terminology-mcq",
        "split": "test", "max": 5000, "streaming": False,
        "conv": conv_mcq,
    },
    {
        "name": "med_term_mcq_dev",
        "id": "chongpangnasilemak/medical-terminology-mcq",
        "split": "dev", "max": 5000, "streaming": False,
        "conv": conv_mcq,
    },
    # Llama2-formatted medical term explanations
    {
        "name": "wiki_med_llama2",
        "id": "mychen76/wiki_medical_terms_llama2",
        "split": "train", "max": 5000, "streaming": True,
        "conv": conv_llama2_text,
    },
    # Dialogue-format medical term Q&A
    {
        "name": "med_terms_dialogue",
        "id": "lamhieu/medical_terms_dialogue_en",
        "split": "train", "max": 5000, "streaming": True,
        "conv": try_messages,
    },
    # Wikipedia medical term + description
    {
        "name": "wiki_med_terms",
        "id": "dmedhi/wiki-medical-terms",
        "split": "train", "max": 5000, "streaming": True,
        "conv": conv_wiki_term,
    },
    # Instruction + input + output medical term explanations
    {
        "name": "wiki_med_inout",
        "id": "HoangHa/wiki_medical_terms_inout",
        "split": "train", "max": 5000, "streaming": True,
        "conv": conv_inout,
    },
    # Public health instruction dataset
    {
        "name": "pub_health_instruct",
        "id": "sambanankhu/Instruction-public-health-dataset",
        "split": "train", "max": 5000, "streaming": True,
        "conv": conv_pub_health_instruct,
    },
    # Public health intelligence — messages format
    {
        "name": "pub_health_intel",
        "id": "introvoyz041/public-health-intelligence-datasetpublic-health-intelligence-dataset",
        "split": "train", "max": 5000, "streaming": True,
        "conv": try_messages,
    },
]

# Skipped:
# MarieDeVox/english-vocal-medical-terminology-mini — audio (needs torchcodec)
# jacekduszenko/rare-medical-terms — just term names, no definitions
# FrancophonIA/Collection_of_public_health — just language code labels
# gvic-unb/public-health-news-DF-qa — Portuguese only
# introvoyz041/model-based-geostatistics-for-global-public-health — image dataset


def main():
    all_pairs = []

    for cfg in DATASETS:
        name = cfg["name"]
        ds_id = cfg["id"]
        print(f"\n[{name}]  {ds_id}  (split={cfg['split']})")
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
    print(f"\nMerge into sft_master:")
    print(f"  cat data/sft_master.jsonl {OUTPUT} > /tmp/m.jsonl && mv /tmp/m.jsonl data/sft_master.jsonl")


if __name__ == "__main__":
    main()
