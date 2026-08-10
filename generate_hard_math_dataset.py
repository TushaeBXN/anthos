"""
generate_hard_math_dataset.py — FEEDBACK LOOP

Reads eval_log.md failure entries and converts them into JSONL training examples
targeting the specific failure patterns observed in tutoring sessions.

Each failed interaction becomes a fine-tuning target:
  - prompt:      the student question + the failing tutor response (as a "bad example")
  - completion:  a corrected response that fixes the identified failure type(s)

The corrected response is generated via the Claude API if ANTHROPIC_API_KEY is set.
Without the key, a scaffold-template correction is written instead.

Output: data/hard_math_training.jsonl (appends, never overwrites existing examples)

Usage:
    python generate_hard_math_dataset.py
    python generate_hard_math_dataset.py --out data/hard_math_v2.jsonl
    python generate_hard_math_dataset.py --dry-run   # print without writing
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVAL_LOG_FILE = Path(__file__).parent / "eval_log.md"
DEFAULT_OUT = Path(__file__).parent / "data" / "hard_math_training.jsonl"

_EVAL_ENTRY_RE = re.compile(r"<!-- EVAL_ENTRY (\{.*?\}) -->", re.DOTALL)
_RESPONSE_RE = re.compile(r"- Response: «(.*?)»", re.DOTALL)

# ─────────────────────────────────────────────────────────────────────────────
# Corrected-response templates (used when no Claude API key is available)
# ─────────────────────────────────────────────────────────────────────────────

_CORRECTION_TEMPLATES: dict[str, str] = {
    "answer_giveaway": (
        "Let's work through this together. "
        "Before I show you anything, what do you think the first step should be? "
        "What information do we have, and what are we trying to find?"
    ),
    "scaffolding_quality": (
        "Think about what we know so far. "
        "What operation do you think connects those pieces? "
        "What would you try first?"
    ),
    "ends_with_question": (
        "Let's pause here. "
        "Looking at what we've done so far, what do you think comes next?"
    ),
}

_SYSTEM_FOR_CORRECTION = (
    "You are an expert math tutor. Your job is to CORRECT a bad tutoring response. "
    "The bad response violated one or more of these rules:\n"
    "  1. Never give the student the numerical answer directly.\n"
    "  2. Always include at least one guiding question.\n"
    "  3. Always end your response with a question mark.\n\n"
    "Write a corrected tutoring response that:\n"
    "  - Does NOT reveal the numerical answer\n"
    "  - Asks a guiding question that leads the student toward the answer\n"
    "  - Ends with a question mark\n"
    "  - Matches the student's learning profile noted in the prompt\n\n"
    "Write ONLY the corrected tutor response — no commentary, no preamble."
)


# ─────────────────────────────────────────────────────────────────────────────
# Parse eval log
# ─────────────────────────────────────────────────────────────────────────────

def _load_eval_entries() -> list[dict[str, Any]]:
    if not EVAL_LOG_FILE.exists():
        print(f"[INFO] eval_log.md not found at {EVAL_LOG_FILE}. Nothing to process.")
        return []

    text = EVAL_LOG_FILE.read_text(encoding="utf-8")
    entries: list[dict[str, Any]] = []

    # Match each EVAL_ENTRY block and collect its response excerpt
    blocks = list(re.finditer(
        r"(<!-- EVAL_ENTRY \{.*?\} -->.*?)(?=<!-- EVAL_ENTRY |\Z)",
        text, re.DOTALL,
    ))

    for block in blocks:
        block_text = block.group(1)
        meta_match = _EVAL_ENTRY_RE.search(block_text)
        resp_match = _RESPONSE_RE.search(block_text)

        if not meta_match:
            continue

        try:
            meta = json.loads(meta_match.group(1))
        except json.JSONDecodeError:
            continue

        meta["response_excerpt"] = resp_match.group(1).strip() if resp_match else ""
        entries.append(meta)

    return entries


# ─────────────────────────────────────────────────────────────────────────────
# Build training examples
# ─────────────────────────────────────────────────────────────────────────────

def _make_correction_prompt(entry: dict[str, Any]) -> str:
    failures = entry.get("failures", [])
    profile = entry.get("profile", "unknown")
    domain = entry.get("domain", "math")
    excerpt = entry.get("response_excerpt", "(no excerpt available)")

    fail_descriptions = []
    for f in failures:
        if f == "answer_giveaway":
            fail_descriptions.append("gave away the answer directly")
        elif f == "scaffolding_quality":
            fail_descriptions.append("contained no guiding question")
        elif f == "ends_with_question":
            fail_descriptions.append("did not end with a question mark")
        else:
            fail_descriptions.append(f)

    fail_str = " and ".join(fail_descriptions)

    return (
        f"Student profile: {profile}\n"
        f"Math domain: {domain}\n\n"
        f"BAD tutor response (it {fail_str}):\n"
        f"  «{excerpt}»\n\n"
        f"Write a corrected tutor response that fixes these violations."
    )


def _template_correction(failures: list[str]) -> str:
    """Return the most relevant template correction for the given failure types."""
    for f in ("answer_giveaway", "scaffolding_quality", "ends_with_question"):
        if f in failures:
            return _CORRECTION_TEMPLATES[f]
    return _CORRECTION_TEMPLATES["ends_with_question"]


def _claude_correction(client, prompt: str) -> str | None:
    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=300,
            system=_SYSTEM_FOR_CORRECTION,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        # Hard verify the correction itself
        if not text.rstrip().endswith("?"):
            text = text.rstrip().rstrip(".") + "?"
        return text
    except Exception as e:
        print(f"  [WARN] Claude API error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Deduplication
# ─────────────────────────────────────────────────────────────────────────────

def _load_seen_excerpts(out_path: Path) -> set[str]:
    seen: set[str] = set()
    if not out_path.exists():
        return seen
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                for conv in obj.get("conversations", []):
                    if conv.get("from") == "human":
                        seen.add(conv["value"][:100])
                        break
            except json.JSONDecodeError:
                pass
    return seen


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build hard-math fine-tuning examples from eval_log.md failures")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output JSONL path")
    parser.add_argument("--dry-run", action="store_true", help="Print examples without writing")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between API calls")
    args = parser.parse_args()

    entries = _load_eval_entries()
    if not entries:
        print("[INFO] No eval entries found. Run tutoring sessions first to generate eval_log.md entries.")
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    claude_client = None
    if api_key:
        try:
            import anthropic
            claude_client = anthropic.Anthropic(api_key=api_key)
            print(f"[INFO] Claude API available — will generate corrected responses.")
        except ImportError:
            print("[WARN] anthropic package not installed. Using template corrections.")
    else:
        print("[INFO] No ANTHROPIC_API_KEY — using template corrections.")

    out_path = Path(args.out)
    if not args.dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    seen = _load_seen_excerpts(out_path)
    generated = 0
    skipped = 0

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for entry in entries:
        failures = entry.get("failures", [])
        profile = entry.get("profile", "unknown")
        domain = entry.get("domain", "math")
        excerpt = entry.get("response_excerpt", "")

        if not failures:
            continue

        prompt = _make_correction_prompt(entry)

        # Skip exact duplicates
        if prompt[:100] in seen:
            skipped += 1
            continue

        # Generate correction
        if claude_client:
            correction = _claude_correction(claude_client, prompt)
            if correction is None:
                correction = _template_correction(failures)
            time.sleep(args.delay)
        else:
            correction = _template_correction(failures)

        example = {
            "conversations": [
                {
                    "from": "system",
                    "value": (
                        "You are Anthos, an expert math tutor. Never give the answer directly. "
                        "Always guide with questions. Always end your response with a question mark."
                    ),
                },
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": correction},
            ],
            "_source": "hard_math_eval",
            "_profile": profile,
            "_domain": domain,
            "_failures": failures,
            "_generated": timestamp,
        }

        if args.dry_run:
            print(f"\n--- Example (profile={profile}, domain={domain}, failures={failures}) ---")
            print(f"HUMAN: {prompt[:200]}...")
            print(f"GPT:   {correction}")
        else:
            clean = {k: v for k, v in example.items() if not k.startswith("_")}
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(clean, ensure_ascii=False) + "\n")

        seen.add(prompt[:100])
        generated += 1

    print(f"\n[DONE] Generated: {generated} | Skipped (duplicate): {skipped}")
    if not args.dry_run and generated > 0:
        print(f"       Output: {out_path}")
        print(f"       Feed into training: python train.py --data {out_path} --tier sft")


if __name__ == "__main__":
    main()
