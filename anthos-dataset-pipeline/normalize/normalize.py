"""
normalize/normalize.py
Merges github_raw.jsonl + claude_ai_raw.jsonl + claude_code_raw.jsonl
into a single clean ShareGPT JSONL file.

Applies basic quality checks:
- Must have at least one code block in an assistant turn
- Assistant turns must have >= MIN_CODE_LINES lines of code
- Must start with a human turn
- Removes exact duplicates by conversation hash
"""

import json
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    GITHUB_RAW_OUT, CLAUDE_AI_RAW_OUT, CLAUDE_CODE_RAW_OUT,
    MERGED_OUT, MIN_CODE_LINES, REQUIRE_CODE_BLOCK,
    ensure_output_dirs,
)

CODE_FENCE = "```"


def has_code(text):
    return CODE_FENCE in (text or "")


def count_code_lines(text):
    """Count lines inside all ``` blocks in text."""
    total = 0
    inside = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(CODE_FENCE):
            inside = not inside
            continue
        if inside:
            total += 1
    return total


def passes_quality(example):
    turns = example.get("conversations", [])
    if len(turns) < 2:
        return False
    if turns[0].get("from") != "human":
        return False

    assistant_turns = [t for t in turns if t.get("from") == "gpt"]
    if not assistant_turns:
        return False

    has_any_code = any(has_code(t["value"]) for t in assistant_turns)
    if REQUIRE_CODE_BLOCK and not has_any_code:
        return False

    if has_any_code:
        max_code_lines = max(count_code_lines(t["value"]) for t in assistant_turns)
        if max_code_lines < MIN_CODE_LINES:
            return False

    return True


def conversation_hash(example):
    """Stable hash of the conversation content for dedup."""
    turns = example.get("conversations", [])
    key = json.dumps(turns, sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()


def load_jsonl(path):
    path = Path(path)
    if not path.exists():
        print(f"  ⚠ Not found, skipping: {path}")
        return []
    examples = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return examples


def run():
    ensure_output_dirs()

    sources = [
        ("github", GITHUB_RAW_OUT),
        ("claude_ai", CLAUDE_AI_RAW_OUT),
        ("claude_code", CLAUDE_CODE_RAW_OUT),
    ]

    seen_hashes = set()
    counts = {}
    total_kept = 0
    total_dropped_quality = 0
    total_dropped_dupe = 0

    with MERGED_OUT.open("w") as out:
        for source_name, path in sources:
            examples = load_jsonl(path)
            kept = 0
            dropped_q = 0
            dropped_d = 0

            for ex in examples:
                if not passes_quality(ex):
                    dropped_q += 1
                    continue
                h = conversation_hash(ex)
                if h in seen_hashes:
                    dropped_d += 1
                    continue
                seen_hashes.add(h)
                out.write(json.dumps(ex) + "\n")
                kept += 1

            counts[source_name] = kept
            total_kept += kept
            total_dropped_quality += dropped_q
            total_dropped_dupe += dropped_d
            print(f"  {source_name}: {kept} kept, {dropped_q} quality-dropped, {dropped_d} dupes")

    print(f"\n✓ Normalize complete")
    print(f"  Total kept:    {total_kept}")
    print(f"  Quality drops: {total_dropped_quality}")
    print(f"  Dupe drops:    {total_dropped_dupe}")
    print(f"  Output:        {MERGED_OUT}")
    return counts


if __name__ == "__main__":
    run()
