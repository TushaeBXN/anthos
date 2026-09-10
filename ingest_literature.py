#!/usr/bin/env python3
"""
ingest_literature.py — Literature, Arts, African Research, Science Papers

Covers:
  - Svngoku/adaption-african-research-literature-current-events-sft (African geopolitics SFT)
  - ACOSharma/literature (English classic literature text)
  - BEE-spoke-data/fineweb-literature-100k (historical/academic literature)
  - BAAI/IndustryCorpus_literature (literary text, book descriptions)
  - adorkin/olmocr_science_pdfs-literature (science paper text, capped)

Skipped:
  - baka999/Erotic_Literature_Collection (adult content, not appropriate for Anthos's mission)
  - All non-English language datasets (Russian, Polish, Arabic, Thai, Chinese, Spanish/Cuban)
  - ImpulseLeap/retrieval_literature (Russian emotion classification)
  - deekshavijayakumxr/scientific-literature-research-assistant-data (inverted-index metadata only)

Output: data/literature_sft.jsonl
Merge:  cat data/sft_master.jsonl data/literature_sft.jsonl > /tmp/m.jsonl && mv /tmp/m.jsonl data/sft_master.jsonl
"""

import json, re
from pathlib import Path
from datasets import load_dataset

OUTPUT = Path("data/literature_sft.jsonl")

SYS = {
    "arts": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You engage with literature, art, and culture with depth and enthusiasm. "
        "You explain themes, context, and meaning in ways that make great works accessible to everyone."
    ),
    "africa": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You have deep knowledge of African history, politics, culture, and current events. "
        "You explain African perspectives with nuance, respect, and grounding in real facts."
    ),
    "science": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You explain scientific research clearly — what it found, why it matters, and what it means "
        "for real people. You make science accessible without dumbing it down."
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
        {"from": "system", "value": SYS[domain]},
        {"from": "human",  "value": q},
        {"from": "gpt",    "value": a},
    ]}


def auto_messages(row, domain="general"):
    """Extract Q&A from messages/conversations list."""
    for field in ("messages", "conversations", "conversation"):
        msgs = row.get(field)
        if not isinstance(msgs, list):
            continue
        q = a = ""
        for m in msgs:
            role = str(m.get("role", m.get("from", ""))).lower()
            content = str(m.get("content", m.get("value", ""))).strip()
            if role == "system":
                continue
            if role in ("user", "human") and not q:
                q = content
            elif role in ("assistant", "gpt") and q and not a:
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                if content:
                    a = content
                break
        if q and a:
            return mc(q, a, domain)
    return None


def conv_african_sft(row):
    """Svngoku/adaption-african-research-literature-current-events-sft: messages format."""
    return auto_messages(row, "africa")


def conv_literature_text(row, domain="arts"):
    """Generic text-field literary converter — use opening as topic."""
    text = str(row.get("text", "") or "").strip()
    if len(text) < 150:
        return None
    # Skip if primarily non-English (detect by ASCII ratio)
    ascii_ratio = sum(1 for c in text[:200] if ord(c) < 128) / min(len(text), 200)
    if ascii_ratio < 0.7:
        return None
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return None
    # Use first substantive line as the topic/title
    topic = lines[0][:200].lstrip('#').strip()
    if len(topic) < 10:
        topic = " ".join(lines[:2])[:200]
    q = f"Discuss or explain this literary passage or excerpt:\n\n{topic}"
    return mc(q, text[:3500], domain)


def conv_science_pdf(row):
    """adorkin/olmocr_science_pdfs-literature: scientific paper OCR text."""
    text = str(row.get("text", "") or "").strip()
    if len(text) < 200:
        return None
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return None
    # First line is usually paper title + author
    title = lines[0][:200]
    if len(title) < 10:
        return None
    q = f"Summarize and explain this scientific paper: {title}"
    return mc(q, text[:3500], "science")


DATASETS = [
    {
        "name": "african_research_current_events",
        "id": "Svngoku/adaption-african-research-literature-current-events-sft",
        "split": "train", "max": 5000, "streaming": True,
        "conv": conv_african_sft,
    },
    {
        "name": "classic_literature",
        "id": "ACOSharma/literature",
        "split": "train", "max": 5000, "streaming": True,
        "conv": lambda r: conv_literature_text(r, "arts"),
    },
    {
        "name": "fineweb_literature",
        "id": "BEE-spoke-data/fineweb-literature-100k",
        "split": "train", "max": 5000, "streaming": True,
        "conv": lambda r: conv_literature_text(r, "arts"),
    },
    {
        "name": "industry_corpus_literature",
        "id": "BAAI/IndustryCorpus_literature",
        "split": "train", "max": 5000, "streaming": True,
        "conv": lambda r: conv_literature_text(r, "arts"),
    },
    {
        "name": "science_pdf_literature",
        "id": "adorkin/olmocr_science_pdfs-literature",
        "split": "train", "max": 3000, "streaming": True,
        "conv": conv_science_pdf,
    },
]


def main():
    all_pairs = []

    for cfg in DATASETS:
        name = cfg["name"]
        ds_id = cfg["id"]
        split = cfg["split"]
        max_pairs = cfg["max"]
        streaming = cfg["streaming"]
        conv_fn = cfg["conv"]

        print(f"\n[{name}]  {ds_id}")
        loaded = False
        first_err = ""
        for try_split in [split, "train", "test", "validation"]:
            try:
                ds = load_dataset(ds_id, split=try_split, streaming=streaming)
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
