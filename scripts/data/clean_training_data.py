"""
clean_training_data.py — Scrub AI-isms and bloat from teacher_conversations.jsonl

Filters and trims Claude's default politeness patterns before GPU training.
Run this once after generation, before uploading to RunPod.

Usage:
    python3 clean_training_data.py --in data/teacher_conversations.jsonl
    python3 clean_training_data.py --in data/teacher_conversations.jsonl --max-words 180
"""

import json
import re
import argparse
from pathlib import Path

# ── Phrases that pollute Anthos's personality ────────────────────────────────
REJECT_PHRASES = [
    "as an ai",
    "as a language model",
    "i'm an ai",
    "i am an ai",
    "i'm just an ai",
    "i cannot provide",
    "i can't provide",
    "i'm not able to",
    "i am not able to",
    "i don't have personal",
    "i don't have feelings",
    "i want to emphasize",
    "it's important to note that",
    "it's worth noting that",
    "please note that",
    "i must clarify",
    "i need to clarify",
    "as always, consult",
    "consult a professional",
    "seek professional advice",
    "i'm happy to help",
    "i'd be happy to",
    "great question",
    "excellent question",
    "certainly!",
    "of course!",
    "absolutely!",
    "sure thing",
]

# ── Opening filler to strip from the start of responses ──────────────────────
FILLER_OPENERS = re.compile(
    r"^(certainly[!,]?\s*|of course[!,]?\s*|absolutely[!,]?\s*|"
    r"great question[!,]?\s*|excellent question[!,]?\s*|"
    r"sure[!,]?\s*|happy to help[!,]?\s*|"
    r"i('d| would) be happy to [^.!?]*[.!?]\s*)",
    re.IGNORECASE,
)


def word_count(text: str) -> int:
    return len(text.split())


def is_contaminated(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in REJECT_PHRASES)


def trim_opener(text: str) -> str:
    return FILLER_OPENERS.sub("", text).strip()


def clean_response(text: str, max_words: int) -> str | None:
    """Return cleaned response, or None if it should be dropped entirely."""
    if is_contaminated(text):
        return None

    text = trim_opener(text)

    # Trim to max_words at sentence boundary
    if word_count(text) > max_words:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        kept, wc = [], 0
        for s in sentences:
            sw = word_count(s)
            if wc + sw > max_words:
                break
            kept.append(s)
            wc += sw
        text = " ".join(kept).strip()

    if word_count(text) < 20:   # too short to be useful after trimming
        return None

    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in",        dest="input",     default="data/teacher_conversations.jsonl")
    parser.add_argument("--out",       dest="output",    default=None,
                        help="Output path (default: overwrites input)")
    parser.add_argument("--max-words", dest="max_words", type=int, default=200,
                        help="Max words per response (default: 200)")
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output) if args.output else src

    total = kept = dropped_contaminated = dropped_short = trimmed = 0

    cleaned_lines = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1

            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue

            convs = ex.get("conversations", [])
            gpt_turn = next((c for c in convs if c["from"] == "gpt"), None)
            if not gpt_turn:
                continue

            original_words = word_count(gpt_turn["value"])
            cleaned = clean_response(gpt_turn["value"], args.max_words)

            if cleaned is None:
                if original_words < 20:
                    dropped_short += 1
                else:
                    dropped_contaminated += 1
                continue

            if word_count(cleaned) < original_words:
                trimmed += 1

            gpt_turn["value"] = cleaned
            cleaned_lines.append(json.dumps(ex, ensure_ascii=False) + "\n")
            kept += 1

    with open(dst, "w", encoding="utf-8") as f:
        f.writelines(cleaned_lines)

    print(f"\n── Clean Report ─────────────────────────────────────")
    print(f"  Total read:          {total:,}")
    print(f"  Kept:                {kept:,}  ({kept/total*100:.1f}%)")
    print(f"  Dropped (AI-isms):   {dropped_contaminated:,}")
    print(f"  Dropped (too short): {dropped_short:,}")
    print(f"  Trimmed (wordcount): {trimmed:,}")
    print(f"  Saved to:            {dst}")
    print(f"─────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
