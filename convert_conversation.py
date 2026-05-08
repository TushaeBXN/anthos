"""
convert_conversation.py — Convert Anthos_conversation.txt to ShareGPT JSONL

Parses the Ollama conversation format where:
  - User turns start with ">>> "
  - Anthos responses follow immediately after

Appends to data/teacher_conversations.jsonl
"""

import json
import re
from pathlib import Path

INPUT_FILE  = "data/Anthos_conversation.txt"
OUTPUT_FILE = "data/teacher_conversations.jsonl"
SYSTEM      = "You are Anthos, an AI assistant and language model created by Tushae Thomas. You are a Thought-Token Bifurcated Recurrent Transformer — a custom architecture with dual processing streams: a non-causal thought stream for full-context working memory, and a causal sequence stream for output generation."

def parse_conversation(text: str) -> list[dict]:
    """Parse Ollama-format conversation into (human, assistant) pairs."""
    pairs = []

    # Split on >>> markers — each block is one user turn + response
    # Handle multi-line >>> inputs (lines starting with "...")
    lines = text.split('\n')

    i = 0
    while i < len(lines):
        line = lines[i]

        # Detect start of a user turn
        if line.startswith('>>> '):
            # Collect the full user message (may span multiple lines with "...")
            user_lines = [line[4:].strip()]  # strip ">>> "
            i += 1
            while i < len(lines) and lines[i].startswith('... '):
                user_lines.append(lines[i][4:].strip())
                i += 1

            user_message = ' '.join(user_lines).strip()
            if not user_message:
                continue

            # Collect the assistant response (everything until next ">>>")
            response_lines = []
            while i < len(lines) and not lines[i].startswith('>>> '):
                response_lines.append(lines[i])
                i += 1

            # Clean up the response
            response = '\n'.join(response_lines).strip()

            # Skip empty responses or system messages
            if not response or response.startswith('Use') or len(response) < 20:
                continue

            pairs.append((user_message, response))
        else:
            i += 1

    return pairs


def to_sharegpt(pairs: list[tuple]) -> list[dict]:
    records = []
    for human, gpt in pairs:
        records.append({
            "conversations": [
                {"from": "system", "value": SYSTEM},
                {"from": "human",  "value": human},
                {"from": "gpt",    "value": gpt},
            ],
            "source": "anthos_conversation"
        })
    return records


def main():
    input_path  = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)

    if not input_path.exists():
        print(f"ERROR: {INPUT_FILE} not found")
        return

    print(f"Reading {INPUT_FILE}...")
    text = input_path.read_text(encoding="utf-8")

    print("Parsing conversation...")
    pairs = parse_conversation(text)
    print(f"  Found {len(pairs)} conversation turns")

    if not pairs:
        print("No pairs found — check the file format")
        return

    # Preview first 3
    print("\nPreview (first 3 pairs):")
    for i, (h, g) in enumerate(pairs[:3]):
        print(f"\n  [{i+1}] Human: {h[:80]}...")
        print(f"       Anthos: {g[:80]}...")

    records = to_sharegpt(pairs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n  Appended {len(records)} examples to {OUTPUT_FILE}")

    # Count total
    total = sum(1 for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip())
    print(f"  Total examples in file: {total}")


if __name__ == "__main__":
    main()
