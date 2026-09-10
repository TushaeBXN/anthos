#!/usr/bin/env python3
"""
ingest_world_history.py — World history, geopolitics, and ancient civilizations.

Output: data/world_history_sft.jsonl
Merge:  cat data/sft_master.jsonl data/world_history_sft.jsonl > /tmp/m.jsonl && mv /tmp/m.jsonl data/sft_master.jsonl
"""

import json
import re
from pathlib import Path
from datasets import load_dataset

SYS_HISTORY = (
    "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
    "You explain world history accurately — causes, consequences, and context. "
    "You present historical events honestly, including uncomfortable truths, "
    "because understanding the past is how we avoid repeating it."
)

SYS_GEOPOLITICS = (
    "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
    "You analyze geopolitical events, conflicts, and international relations with depth and balance. "
    "You present facts clearly and acknowledge multiple perspectives."
)

SYS_CIVILIZATION = (
    "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
    "You explain ancient civilizations, their cultures, politics, and legacies with scholarly accuracy. "
    "You connect ancient patterns to modern parallels."
)

OUTPUT = Path("data/world_history_sft.jsonl")
CHOICE_LABELS = list("ABCDE")


def mc(q, a, system=SYS_HISTORY):
    q, a = q.strip(), a.strip()[:4000]
    if not q or not a or len(q) < 8 or len(a) < 12:
        return None
    return {"conversations": [
        {"from": "system", "value": system},
        {"from": "human",  "value": q},
        {"from": "gpt",    "value": a},
    ]}


def auto_messages(row, system=SYS_HISTORY):
    """Try message/conversation lists."""
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
    return None


# ─── Dataset-specific converters ─────────────────────────────────────────────

def conv_mmlu_history(row):
    """MMLU world history MCQ."""
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
    return mc(question, a)


def conv_prosodia(row):
    """prosodia-world-history: question_en + answer_en."""
    q = str(row.get("question_en", "") or "").strip()
    a = str(row.get("answer_en", "") or "").strip()
    return mc(q, a) if q and a else None


def conv_quotes(row):
    """english_historical_quotes: quote + author + category."""
    quote = str(row.get("quote", "") or "").strip()
    author = str(row.get("author", "") or "").strip()
    category = str(row.get("category", "") or "").strip()
    if not quote or not author or len(quote) < 20:
        return None
    q = f'Who said this quote and what does it mean?\n\n"{quote}"'
    a = f'This quote is by {author}.'
    if category:
        a += f' It falls under the theme of {category}.'
    a += f'\n\n"{quote}" — {author}'
    return mc(q, a)


def conv_geopolitical_bias(row):
    """deepseek-geopolitical-bias: use Claude/O1 answers (DeepSeek often censored)."""
    country = str(row.get("Country", "") or "").strip()
    incident = str(row.get("Incident", "") or "").strip()
    subtopic = str(row.get("Sub Topic", "") or "").strip()
    question = str(row.get("Question", "") or "").strip()
    if not question:
        return None
    # Prefer Opus > Sonnet > O1 — DeepSeek answers are often refused
    answer = ""
    for field in ("Opus Answer", "Sonnet Answer", "O1 Answer"):
        v = str(row.get(field, "") or "").strip()
        if v and len(v) > 30 and "cannot" not in v.lower()[:50] and "sorry" not in v.lower()[:30]:
            answer = v
            break
    if not answer:
        return None
    context = ""
    if incident:
        context = f"Context: {incident}"
        if subtopic:
            context += f" — {subtopic}"
        context += "\n\n"
    return mc(context + question, answer, SYS_GEOPOLITICS)


def conv_rag_conflict(row):
    """geopolitical_conflict_rag_v2: text_chunks as geopolitical analysis."""
    text = str(row.get("text_chunks", "") or "").strip()
    metadata = str(row.get("metadata", "") or "").strip()
    if len(text) < 100:
        return None
    # First sentence as context clue
    first = text.split('.')[0].strip()[:120]
    source = metadata.replace(".pdf", "").replace("-", " ").replace("_", " ")
    q = f"Analyze the following geopolitical situation: {first}..."
    return mc(q, text[:3500], SYS_GEOPOLITICS)


def conv_ancient_civ(row):
    """Ancient_Civilization_25k: prompt + answer."""
    q = str(row.get("prompt", "") or "").strip()
    a = str(row.get("answer", "") or "").strip()
    return mc(q, a, SYS_CIVILIZATION) if q and a else None


def conv_un_corpus(row):
    """UN_Historical_PDF: use English text field."""
    text = str(row.get("en", "") or "").strip()
    if len(text) < 150:
        return None
    # Strip header boilerplate, grab substantive content
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    # Skip lines that are just codes/dates/formatting
    content_lines = [l for l in lines if len(l) > 40]
    if not content_lines:
        return None
    content = " ".join(content_lines[:50])[:2500]
    first = content_lines[0][:120]
    q = f"Summarize or explain this United Nations document: {first}"
    return mc(q, content)


def conv_auto(row, system=SYS_HISTORY):
    """Generic auto-converter for unknown formats."""
    p = auto_messages(row, system)
    if p:
        return p
    for qf in ("instruction", "question", "prompt", "title", "query"):
        for af in ("output", "answer", "response", "content", "text", "body"):
            q = str(row.get(qf, "") or "").strip()
            a = str(row.get(af, "") or "").strip()
            if q and a and q != a and len(a) > 20:
                return mc(q, a, system)
    for tf in ("text", "content", "body", "description"):
        t = str(row.get(tf, "") or "").strip()
        if len(t) > 200:
            first = t.splitlines()[0][:120]
            return mc(f"Explain this historical event or document: {first}", t[:3000], system)
    return None


# ─── Dataset registry ─────────────────────────────────────────────────────────

DATASETS = [
    {
        "name": "mmlu_world_history",
        "id": "joey234/mmlu-high_school_world_history-neg-prepend-fix",
        "split": "test", "max": 3000, "streaming": False,
        "conv": conv_mmlu_history, "system": SYS_HISTORY,
    },
    {
        "name": "prosodia_world_history",
        "id": "dequeirozrodriguez/prosodia-world-history-1500-qa-pt",
        "split": "train", "max": 5000, "streaming": True,
        "conv": conv_prosodia, "system": SYS_HISTORY,
    },
    {
        "name": "historical_quotes",
        "id": "m-ric/english_historical_quotes",
        "split": "train", "max": 5000, "streaming": True,
        "conv": conv_quotes, "system": SYS_HISTORY,
    },
    {
        "name": "geopolitical_bias",
        "id": "enkryptai/deepseek-geopolitical-bias-dataset",
        "split": "train", "max": 5000, "streaming": False,
        "conv": conv_geopolitical_bias, "system": SYS_GEOPOLITICS,
    },
    {
        "name": "geopolitical_conflict_rag",
        "id": "sHeHrYaR11/geopolitical_conflict_rag_v2",
        "split": "train", "max": 5000, "streaming": True,
        "conv": conv_rag_conflict, "system": SYS_GEOPOLITICS,
    },
    {
        "name": "ancient_civilization",
        "id": "11-47/Ancient_Civilization_25k",
        "split": "train", "max": 5000, "streaming": True,
        "conv": conv_ancient_civ, "system": SYS_CIVILIZATION,
    },
    {
        "name": "un_historical_corpus",
        "id": "ranWang/UN_Historical_PDF_Article_Text_Corpus",
        "split": "train", "max": 5000, "streaming": True,
        "conv": conv_un_corpus, "system": SYS_HISTORY,
    },
    # Banned historical archives — important for "don't repeat history"
    {
        "name": "banned_historical_archives",
        "id": "banned-historical-archives/banned-historical-archives",
        "split": "train", "max": 5000, "streaming": True,
        "conv": lambda r: conv_auto(r, SYS_HISTORY), "system": SYS_HISTORY,
    },
    {
        "name": "banned_historical_archives2",
        "id": "Dragonegg2026/banned-historical-archives",
        "split": "train", "max": 5000, "streaming": True,
        "conv": lambda r: conv_auto(r, SYS_HISTORY), "system": SYS_HISTORY,
    },
]

# Skipped (not suitable for English instruction tuning):
# if001/world_history_textbook_phi4 — Japanese language
# Llamqak/historical_eth_drbs — ETH crypto price bars
# VoiceOfML/A-Historical-Learning-Data — empty/garbage
# freQuensy23/russian_rebellion_historical_data — numerical only
# QingYuYunTu/Chinese_Historical_Figures_Dialogue — Chinese language
# RevolutionCrossroads/si_us_revolutionary_era_collections — museum catalog, no text
# alerterra/* — gated
# electricsheepasia/asia-owid-* — tabular GDP data
# dsfefvx/finance-historical-data — financial tabular data
# biglam/icdar2021-historical-document-dating — OCR benchmark
# ahmedheakl/arocrbench_historicalbooks — Arabic OCR
# prg-unibe/dodis-historical-documents — diplomatic archive (complex format)
# bigscience-historical-texts/Open_Medieval_French — medieval French
# Mediocreatmybest/Miscellany_of_Australian_Historical_Photography — photos


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
