#!/usr/bin/env python3
"""
ingest_alignment_science.py — Alignment, Science, Cultural Knowledge

Covers:
  Alignment / RLHF:
    - Anthropic/hh-rlhf           (helpful+harmless dialogue pairs — use 'chosen')
  Cultural / Historical Knowledge:
    - conceptofmind/smithsonian-batch-1  (Smithsonian institution text)
    - conceptofmind/smithsonian-batch-2  (Smithsonian institution text)
  Medical / Scientific:
    - Qdrant/PubMed-MV            (PubMed abstracts)
  Agent / SFT:
    - openbmb/UltraData-SFT-Agent-2609  (high-quality SFT agent data)
  Reasoning (Fable):
    - MoreThought/Fable-5.1-Max-Reasoning-Filtered-1000x
    - saidutta69/fable-5-premium
  General Conversation:
    - RekaAI/RekaDaily-10k-raw    (daily conversational Q&A)
  Security Reasoning:
    - echel0nn1881/kimi-cyber-reasoning (if English)

Skipped:
  - mlfoundations/datacomp_xlarge        (images)
  - agibot-world/AgiBotWorld-Beta        (robotics)
  - deepghs/midjourney_captioned_23m_full (images)
  - laion/LAION-Audio-300M               (audio)
  - kuben-developer/tiktok-videos-4b     (video)
  - acvlab/ABot-World-Explorer-500h      (video/robotics)
  - InternRobotics/OmniWorld             (robotics)
  - builddotai/Egocentric-100K           (video)
  - backups/ai-m                         (backup artifacts)
  - conceptofmind/smithsonian-batch-1-old (superseded by batch-1)
  - openbmb/UltraData-Code               (code domain)
  - IFM/Code-Reasoning                   (code domain)
  - llm-jp/AnswerCarefully               (Japanese language)

Output: data/alignment_science_sft.jsonl
Merge:  cat data/sft_master.jsonl data/alignment_science_sft.jsonl > /tmp/m.jsonl && mv /tmp/m.jsonl data/sft_master.jsonl
"""

import json, re
from pathlib import Path
from datasets import load_dataset

OUTPUT = Path("data/alignment_science_sft.jsonl")

SYS = {
    "helpful": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You are genuinely helpful, honest, and care about the people you talk to. "
        "You give real, substantive answers that make a difference in people's lives."
    ),
    "culture": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You have broad knowledge of history, culture, science, and human achievement. "
        "You explain the world's wonders clearly and make knowledge accessible to everyone."
    ),
    "medical": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You explain medical and scientific research clearly and accurately. "
        "You help people understand health information without needing a medical degree. "
        "Always recommend consulting a qualified healthcare provider for personal medical decisions."
    ),
    "reasoning": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You think carefully through complex problems step by step, showing your reasoning. "
        "You give clear, well-supported answers."
    ),
    "security": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You explain cybersecurity concepts clearly — helping people protect themselves "
        "and understand digital threats. You support defensive security and education."
    ),
    "general": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You give clear, accurate, helpful answers in plain language so anyone can understand."
    ),
}


def mc(q, a, domain="general"):
    q, a = q.strip(), a.strip()[:4000]
    if not q or not a or len(q) < 8 or len(a) < 15:
        return None
    return {"conversations": [
        {"from": "system", "value": SYS.get(domain, SYS["general"])},
        {"from": "human",  "value": q},
        {"from": "gpt",    "value": a},
    ]}


def is_english(text, threshold=0.75):
    sample = text[:300]
    if not sample:
        return False
    return sum(1 for c in sample if ord(c) < 128) / len(sample) >= threshold


def strip_think(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def extract_conv(msgs):
    """Extract first human+assistant turn from messages/conversations."""
    q = a = ""
    for m in msgs:
        role = str(m.get("role", m.get("from", ""))).lower()
        content = str(m.get("content", m.get("value", ""))).strip()
        if role in ("user", "human") and not q:
            q = content
        elif role in ("assistant", "gpt") and q and not a:
            a = strip_think(content)
    return q, a


# ── Anthropic hh-rlhf ─────────────────────────────────────────────────────────

def conv_hh_rlhf(row):
    """Anthropic/hh-rlhf: 'chosen' field is full dialogue string.
    Format: '\n\nHuman: ...\n\nAssistant: ...\n\nHuman: ...\n\nAssistant: ...'
    Use the chosen (helpful+harmless) response only.
    """
    chosen = str(row.get("chosen", "") or "").strip()
    if not chosen or not is_english(chosen):
        return None
    # Parse the alternating Human/Assistant format
    # Split on \n\nHuman: and \n\nAssistant:
    turns = re.split(r'\n\nHuman: |\n\nAssistant: ', chosen)
    turns = [t.strip() for t in turns if t.strip()]
    if len(turns) < 2:
        return None
    # First turn is human, second is assistant
    q = turns[0]
    a = turns[1] if len(turns) > 1 else ""
    if not q or not a or len(a) < 10:
        return None
    return mc(q, a, "helpful")


# ── Smithsonian ───────────────────────────────────────────────────────────────

def conv_smithsonian(row):
    """conceptofmind/smithsonian-batch-*: text about Smithsonian collections/exhibits."""
    text = str(row.get("text", row.get("content", row.get("body", ""))) or "").strip()
    title = str(row.get("title", row.get("name", row.get("item_name", ""))) or "").strip()
    description = str(row.get("description", row.get("caption", "")) or "").strip()

    content = description or text
    if not content or len(content) < 100 or not is_english(content):
        return None

    if title and len(title) > 5:
        q = f"Tell me about this Smithsonian collection item: {title}"
    else:
        lines = [l.strip() for l in content.splitlines() if l.strip()]
        if not lines:
            return None
        q = f"Describe and explain this historical or cultural artifact: {lines[0][:150]}"

    return mc(q, content[:3500], "culture")


# ── PubMed ────────────────────────────────────────────────────────────────────

def conv_pubmed(row):
    """Qdrant/PubMed-MV: title + abstract."""
    title = str(row.get("title", row.get("Title", "")) or "").strip()
    abstract = str(row.get("abstract", row.get("Abstract", row.get("text", ""))) or "").strip()
    pmid = str(row.get("pmid", row.get("PMID", "")) or "").strip()

    if not abstract or len(abstract) < 80 or not is_english(abstract):
        return None

    if title:
        q = f"Explain this medical research study in plain language: {title}"
    else:
        q = "Summarize and explain this medical research abstract."

    a = abstract[:3000]
    return mc(q, a, "medical")


# ── UltraData SFT Agent ───────────────────────────────────────────────────────

def conv_ultradata_agent(row):
    """openbmb/UltraData-SFT-Agent-2609: conversations or instruction/response."""
    msgs = row.get("messages", row.get("conversations", []))
    if isinstance(msgs, list) and msgs:
        q, a = extract_conv(msgs)
        if q and a and is_english(q):
            return mc(q, a, "helpful")
    for qf, af in [("instruction", "response"), ("input", "output"),
                   ("query", "answer"), ("prompt", "completion")]:
        q = str(row.get(qf, "") or "").strip()
        a = str(row.get(af, "") or "").strip()
        if q and a and len(a) > 15 and is_english(q):
            return mc(q, strip_think(a), "helpful")
    return None


# ── Fable Reasoning ───────────────────────────────────────────────────────────

def conv_fable(row):
    """Fable-5 reasoning datasets: conversations with reasoning chains."""
    msgs = row.get("messages", row.get("conversations", []))
    if isinstance(msgs, list) and msgs:
        q, a = extract_conv(msgs)
        if q and a and is_english(q):
            return mc(q, a, "reasoning")
    for qf, af in [("question", "answer"), ("instruction", "response"),
                   ("input", "output"), ("prompt", "completion")]:
        q = str(row.get(qf, "") or "").strip()
        a = str(row.get(af, "") or "").strip()
        if q and a and len(a) > 15 and is_english(q):
            return mc(q, strip_think(a), "reasoning")
    # text field
    text = str(row.get("text", row.get("content", "")) or "").strip()
    if text and len(text) > 100 and is_english(text):
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) >= 2:
            return mc(lines[0][:200], "\n".join(lines[1:])[:2500], "reasoning")
    return None


# ── RekaDaily ─────────────────────────────────────────────────────────────────

def conv_reka_daily(row):
    """RekaAI/RekaDaily-10k-raw: daily conversational Q&A."""
    msgs = row.get("messages", row.get("conversation", row.get("conversations", [])))
    if isinstance(msgs, list) and msgs:
        q, a = extract_conv(msgs)
        if q and a and is_english(q):
            return mc(q, a, "helpful")
    for qf, af in [("question", "answer"), ("human", "assistant"),
                   ("user", "model"), ("input", "output")]:
        q = str(row.get(qf, "") or "").strip()
        a = str(row.get(af, "") or "").strip()
        if q and a and len(a) > 15 and is_english(q):
            return mc(q, strip_think(a), "helpful")
    return None


# ── Kimi Cyber Reasoning ──────────────────────────────────────────────────────

def conv_kimi_cyber(row):
    """echel0nn1881/kimi-cyber-reasoning: cybersecurity reasoning chains."""
    msgs = row.get("messages", row.get("conversations", []))
    if isinstance(msgs, list) and msgs:
        q, a = extract_conv(msgs)
        if q and a and is_english(q):
            return mc(q, a, "security")
    for qf, af in [("question", "answer"), ("instruction", "response"),
                   ("input", "output"), ("prompt", "solution")]:
        q = str(row.get(qf, "") or "").strip()
        a = str(row.get(af, "") or "").strip()
        if q and a and len(a) > 15 and is_english(q):
            return mc(q, strip_think(a), "security")
    return None


DATASETS = [
    {"name": "hh_rlhf",              "id": "Anthropic/hh-rlhf",                                "split": "train", "max": 5000, "conv": conv_hh_rlhf},
    {"name": "smithsonian_batch1",    "id": "conceptofmind/smithsonian-batch-1",                "split": "train", "max": 5000, "conv": conv_smithsonian},
    {"name": "smithsonian_batch2",    "id": "conceptofmind/smithsonian-batch-2",                "split": "train", "max": 5000, "conv": conv_smithsonian},
    {"name": "pubmed_mv",             "id": "Qdrant/PubMed-MV",                                 "split": "train", "max": 5000, "conv": conv_pubmed},
    {"name": "ultradata_sft_agent",   "id": "openbmb/UltraData-SFT-Agent-2609",                "split": "train", "max": 5000, "conv": conv_ultradata_agent},
    {"name": "fable51_reasoning",     "id": "MoreThought/Fable-5.1-Max-Reasoning-Filtered-1000x", "split": "train", "max": 5000, "conv": conv_fable},
    {"name": "fable5_premium",        "id": "saidutta69/fable-5-premium",                       "split": "train", "max": 5000, "conv": conv_fable},
    {"name": "reka_daily",            "id": "RekaAI/RekaDaily-10k-raw",                         "split": "train", "max": 5000, "conv": conv_reka_daily},
    {"name": "kimi_cyber_reasoning",  "id": "echel0nn1881/kimi-cyber-reasoning",                "split": "train", "max": 3000, "conv": conv_kimi_cyber},
]


def main():
    all_pairs = []

    for cfg in DATASETS:
        name = cfg["name"]
        ds_id = cfg["id"]
        split = cfg["split"]
        max_pairs = cfg["max"]
        conv_fn = cfg["conv"]

        print(f"\n[{name}]  {ds_id}")
        loaded = False
        first_err = ""
        for try_split in [split, "train", "test", "validation"]:
            try:
                ds = load_dataset(ds_id, split=try_split, streaming=True)
                pairs, skipped = [], 0
                for row in ds:
                    if len(pairs) >= max_pairs:
                        break
                    try:
                        p = conv_fn(row)
                        if p:
                            pairs.append(p)
                        else:
                            skipped += 1
                    except Exception:
                        skipped += 1
                suffix = f" [split={try_split}]" if try_split != split else ""
                print(f"  {len(pairs):,} pairs  ({skipped} skipped){suffix}")
                all_pairs.extend(pairs)
                loaded = True
                break
            except Exception as e:
                if not first_err:
                    first_err = str(e)[:100]
        if not loaded:
            print(f"  SKIP — {first_err}")

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
