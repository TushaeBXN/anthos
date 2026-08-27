"""
augment/thought_token_wrap.py — Prepend <|thought|> blocks to coding responses.

This is the core Anthos-specific augmentation. For each assistant turn that
contains code, we ask Claude to reconstruct the internal reasoning that would
have produced it — what tradeoffs were weighed, what was rejected, why this
approach. That reasoning becomes the thought-token training signal for Anthos's
bifurcated thought/sequence stream architecture.

Usage:
    python augment/thought_token_wrap.py \
        --in  output/merged/anthos-code-training-v1.jsonl \
        --out output/merged/anthos-code-training-v1-thought.jsonl \
        --max 5000
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

THOUGHT_OPEN  = "<|thought|>"
THOUGHT_CLOSE = "<|/thought|>"

SYSTEM_PROMPT = """\
You are helping build training data for Anthos, a custom AI with a bifurcated
thought/sequence architecture. For each coding response you receive, write a
concise internal reasoning block (3-6 sentences) that explains:
- What the core problem requires
- Key design tradeoffs considered
- Why this specific approach was chosen over alternatives
- Any edge cases or gotchas accounted for

Output ONLY this format — nothing else:
<|thought|>
[your reasoning here]
<|/thought|>
[original response verbatim]"""


def _has_code(text: str) -> bool:
    return "```" in text


def _wrap_with_thought(client, response: str, model: str, max_tokens: int) -> str:
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": response}],
    )
    return message.content[0].text


def augment_file(
    in_path: str,
    out_path: str,
    max_examples: int = 0,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 512,
    delay: float = 0.5,
) -> dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set — thought wrapping requires the API")

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    in_path  = Path(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "wrapped": 0, "skipped_no_code": 0, "errors": 0}

    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            if max_examples and stats["total"] >= max_examples:
                break
            stats["total"] += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            turns = record.get("conversations", [])
            modified = False

            new_turns = []
            for turn in turns:
                if turn.get("from") in ("gpt", "assistant") and _has_code(turn.get("value", "")):
                    try:
                        wrapped = _wrap_with_thought(client, turn["value"], model, max_tokens)
                        new_turn = dict(turn)
                        new_turn["value"] = wrapped
                        new_turns.append(new_turn)
                        modified = True
                        time.sleep(delay)
                    except Exception as e:
                        print(f"  [error] {e}")
                        stats["errors"] += 1
                        new_turns.append(turn)
                else:
                    new_turns.append(turn)

            if modified:
                record["conversations"] = new_turns
                stats["wrapped"] += 1
            else:
                stats["skipped_no_code"] += 1

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            if stats["total"] % 100 == 0:
                print(f"  {stats['total']:,} processed | {stats['wrapped']:,} wrapped | {stats['errors']} errors")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in",  required=True, dest="inp")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max", type=int, default=0, help="Max examples (0=all)")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    stats = augment_file(args.inp, args.out, args.max, args.model, args.max_tokens)
    print(f"\nDone. Wrapped {stats['wrapped']:,} / {stats['total']:,} examples")
    print(f"Skipped (no code): {stats['skipped_no_code']:,} | Errors: {stats['errors']}")
