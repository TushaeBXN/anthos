#!/usr/bin/env python3
"""
ingest_poetry_books_patch.py — Fix skipped datasets from poetry/books run

Fixes:
  - ubaada/booksum-complete-cleaned  → need config 'chapters'
  - AlephFunk/storyworld-plays       → need config 'turns' or 'episodes'
  - bestofbothworldsenjoyer/literary-reasoning-filtered → format was messages list
  - Hananguyen12/shakespeare-QA-plays → check actual split names
  - agentlans/literary-reasoning     → zstd not supported, try non-streaming

Output: data/poetry_books_patch_sft.jsonl
"""

import json, re
from pathlib import Path
from datasets import load_dataset

OUTPUT = Path("data/poetry_books_patch_sft.jsonl")

SYS = {
    "literature": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You engage with literature, storytelling, and the written word with depth and enthusiasm. "
        "You explain themes, context, and meaning in ways that make great works accessible to everyone."
    ),
    "reasoning": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You think carefully and reason through complex questions step by step. "
        "You give clear, well-supported answers that help people understand and learn."
    ),
}


def mc(q, a, domain="literature"):
    q, a = q.strip(), a.strip()[:4000]
    if not q or not a or len(q) < 8 or len(a) < 15:
        return None
    return {"conversations": [
        {"from": "system", "value": SYS.get(domain, SYS["literature"])},
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


def try_load(ds_id, config=None, splits=None, streaming=True):
    """Try to load a dataset with multiple split fallbacks."""
    for split in (splits or ["train", "test", "validation"]):
        try:
            kwargs = {"split": split, "streaming": streaming}
            if config:
                kwargs["name"] = config
            return load_dataset(ds_id, **kwargs), split
        except Exception:
            continue
    return None, None


# ── booksum chapters ──────────────────────────────────────────────────────────

def ingest_booksum():
    pairs = []
    ds, split = try_load("ubaada/booksum-complete-cleaned", config="chapters")
    if ds is None:
        ds, split = try_load("ubaada/booksum-complete-cleaned", config="books")
    if ds is None:
        print("  booksum: SKIP — could not load any config")
        return pairs
    skipped = 0
    for row in ds:
        if len(pairs) >= 5000:
            break
        try:
            summary = str(row.get("summary", row.get("Summary", "")) or "").strip()
            chapter = str(row.get("chapter", row.get("text", "")) or "").strip()
            title = str(row.get("title", row.get("book_id", "")) or "").strip()
            if not summary or len(summary) < 30 or not is_english(summary):
                skipped += 1
                continue
            if title:
                q = f'Summarize this chapter from "{title}".'
            elif chapter:
                q = f"Summarize the following passage:\n\n{chapter[:400]}"
            else:
                q = "Provide a summary of this literary passage."
            p = mc(q, summary[:3000], "literature")
            if p:
                pairs.append(p)
            else:
                skipped += 1
        except Exception:
            skipped += 1
    print(f"  booksum/chapters [{split}]: {len(pairs):,} pairs ({skipped} skipped)")
    return pairs


# ── storyworld plays ──────────────────────────────────────────────────────────

def ingest_storyworld():
    pairs = []
    for config in ["turns", "episodes"]:
        ds, split = try_load("AlephFunk/storyworld-plays", config=config)
        if ds is None:
            continue
        skipped = 0
        batch = []
        for row in ds:
            if len(batch) >= 3000:
                break
            try:
                # turns format: speaker+text or role+content
                text = str(row.get("text", row.get("content", row.get("line", ""))) or "").strip()
                speaker = str(row.get("speaker", row.get("character", row.get("role", ""))) or "").strip()
                title = str(row.get("title", row.get("play", row.get("episode", ""))) or "").strip()
                if not text or len(text) < 20 or not is_english(text):
                    skipped += 1
                    continue
                if speaker and title:
                    q = f'In the play "{title}", what does {speaker} say?'
                elif speaker:
                    q = f"What does {speaker} say in this scene?"
                else:
                    q = "What happens in this scene from the play?"
                p = mc(q, text[:2000], "literature")
                if p:
                    batch.append(p)
                else:
                    skipped += 1
            except Exception:
                skipped += 1
        if batch:
            print(f"  storyworld/{config} [{split}]: {len(batch):,} pairs ({skipped} skipped)")
            pairs.extend(batch)
            break  # one config is enough
    if not pairs:
        print("  storyworld: SKIP — no usable pairs found")
    return pairs


# ── literary reasoning (non-streaming for zstd) ───────────────────────────────

def ingest_literary_reasoning():
    pairs = []
    for ds_id in [
        "agentlans/literary-reasoning",
        "bestofbothworldsenjoyer/literary-reasoning-filtered",
    ]:
        batch = []
        skipped = 0
        try:
            # Try non-streaming to avoid zstd issue
            ds = load_dataset(ds_id, split="train", streaming=False)
            for row in ds:
                if len(batch) >= 5000:
                    break
                try:
                    # Try standard Q&A fields
                    for qf, af in [("question", "answer"), ("instruction", "response"),
                                   ("input", "output"), ("prompt", "completion")]:
                        q = str(row.get(qf, "") or "").strip()
                        a = str(row.get(af, "") or "").strip()
                        if q and a and len(a) > 20 and is_english(q):
                            p = mc(q, a, "reasoning")
                            if p:
                                batch.append(p)
                            else:
                                skipped += 1
                            break
                    else:
                        # Try messages list
                        msgs = row.get("messages", row.get("conversations", []))
                        if isinstance(msgs, list):
                            q = a = ""
                            for m in msgs:
                                role = str(m.get("role", m.get("from", ""))).lower()
                                content = str(m.get("content", m.get("value", ""))).strip()
                                if role in ("user", "human") and not q:
                                    q = content
                                elif role in ("assistant", "gpt") and q and not a:
                                    a = strip_think(content)
                            if q and a:
                                p = mc(q, a, "reasoning")
                                if p:
                                    batch.append(p)
                                    continue
                        skipped += 1
                except Exception:
                    skipped += 1
            print(f"  {ds_id.split('/')[-1]}: {len(batch):,} pairs ({skipped} skipped)")
            pairs.extend(batch)
        except Exception as e:
            print(f"  {ds_id.split('/')[-1]}: SKIP — {str(e)[:80]}")
    return pairs


# ── Shakespeare Q&A ───────────────────────────────────────────────────────────

def ingest_shakespeare():
    pairs = []
    try:
        ds = load_dataset("Hananguyen12/shakespeare-QA-plays", streaming=False)
        split_name = list(ds.keys())[0]
        ds = ds[split_name]
        skipped = 0
        for row in ds:
            if len(pairs) >= 5000:
                break
            try:
                q = str(row.get("question", row.get("Question", row.get("query", ""))) or "").strip()
                a = str(row.get("answer", row.get("Answer", row.get("response", ""))) or "").strip()
                context = str(row.get("context", row.get("Context", row.get("passage", ""))) or "").strip()
                if not q or not a or len(a) < 10:
                    skipped += 1
                    continue
                if context and len(context) > 30:
                    full_q = f"Context from Shakespeare:\n\n{context[:500]}\n\nQuestion: {q}"
                else:
                    full_q = q
                p = mc(full_q, a, "literature")
                if p:
                    pairs.append(p)
                else:
                    skipped += 1
            except Exception:
                skipped += 1
        print(f"  shakespeare_qa [{split_name}]: {len(pairs):,} pairs ({skipped} skipped)")
    except Exception as e:
        print(f"  shakespeare_qa: SKIP — {str(e)[:80]}")
    return pairs


def main():
    all_pairs = []

    print("\n[booksum_chapters]")
    all_pairs.extend(ingest_booksum())

    print("\n[storyworld_plays]")
    all_pairs.extend(ingest_storyworld())

    print("\n[literary_reasoning_datasets]")
    all_pairs.extend(ingest_literary_reasoning())

    print("\n[shakespeare_qa]")
    all_pairs.extend(ingest_shakespeare())

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
