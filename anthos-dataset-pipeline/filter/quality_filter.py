"""
filter/quality_filter.py — Drop low-quality training examples.
"""

import hashlib
import json
import re
from pathlib import Path


def _has_code_block(text: str) -> bool:
    return bool(re.search(r"```[\w]*\n.+?```", text, re.DOTALL))


def _count_code_lines(text: str) -> int:
    total = 0
    for block in re.findall(r"```[\w]*\n(.+?)```", text, re.DOTALL):
        total += len([l for l in block.splitlines() if l.strip()])
    return total


def _is_refusal(text: str) -> bool:
    refusals = [
        "i can't help with that",
        "i cannot help with that",
        "i'm not able to",
        "i am not able to",
        "i won't be able to",
        "as an ai language model",
        "i don't have the ability",
    ]
    lower = text.lower()
    return any(r in lower for r in refusals) and len(text) < 300


def _fingerprint(record: dict) -> str:
    turns = record.get("conversations", [])
    text = " ".join(t.get("value", "") for t in turns)
    return hashlib.md5(text.encode()).hexdigest()


def filter_file(
    in_path: str,
    out_path: str,
    min_code_lines: int = 10,
    require_code_block: bool = True,
) -> dict:
    in_path  = Path(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen_hashes: set[str] = set()
    stats = {"total": 0, "kept": 0, "dropped_no_code": 0,
             "dropped_short": 0, "dropped_refusal": 0, "dropped_duplicate": 0}

    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            stats["total"] += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Find the assistant turn
            turns = record.get("conversations", [])
            assistant_turns = [t["value"] for t in turns if t.get("from") in ("gpt", "assistant")]
            if not assistant_turns:
                continue
            response = assistant_turns[-1]

            # Filters
            if require_code_block and not _has_code_block(response):
                stats["dropped_no_code"] += 1
                continue

            if require_code_block and _count_code_lines(response) < min_code_lines:
                stats["dropped_short"] += 1
                continue

            if _is_refusal(response):
                stats["dropped_refusal"] += 1
                continue

            fp = _fingerprint(record)
            if fp in seen_hashes:
                stats["dropped_duplicate"] += 1
                continue
            seen_hashes.add(fp)

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["kept"] += 1

    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in",  required=True, dest="inp")
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-code-lines", type=int, default=10)
    parser.add_argument("--no-require-code", action="store_true")
    args = parser.parse_args()

    stats = filter_file(
        args.inp, args.out,
        min_code_lines=args.min_code_lines,
        require_code_block=not args.no_require_code,
    )
    print(f"Total:     {stats['total']:,}")
    print(f"Kept:      {stats['kept']:,}")
    print(f"No code:   {stats['dropped_no_code']:,}")
    print(f"Too short: {stats['dropped_short']:,}")
    print(f"Refusal:   {stats['dropped_refusal']:,}")
    print(f"Duplicate: {stats['dropped_duplicate']:,}")
