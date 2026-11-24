#!/usr/bin/env python3
"""
generate_dpo_data.py — Anthos DPO preference pairs.

Two axes:
  depth:     full <think> trace + answer  >  shallow answer
  tool_call: valid JSON tool_call         >  malformed/incomplete tool_call

Output: JSONL with {"prompt": ..., "chosen": ..., "rejected": ...}
for Unsloth DPOTrainer.

NOTE: DPO beta should be 0.5 for instruction-tuned models.
      beta=0.1 is aggressive and risks capability loss on this model class.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

LOG = logging.getLogger("anthos.dpo_data")

SYSTEM = (
    "You are Anthos. You reason before acting. You plan, call tools, "
    "observe results, and self-correct. You do not give shallow answers."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run tests and return output",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        },
    }
]


def prompt_text(task: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{task}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def deep_trace(task: str) -> str:
    return (
        "<think>\n"
        "1. Restate the objective and constraints.\n"
        "2. Identify the failing component.\n"
        "3. Plan: inspect, patch, verify.\n"
        "4. Anticipate a boundary-case regression.\n"
        "</think>\n"
        f"<answer>\nCompleted the task: {task}\n</answer>"
    )


def shallow_trace(task: str) -> str:
    return f"<answer>\nDone: {task}\n</answer>"


def valid_tool_call() -> str:
    return (
        '<tool_call>\n'
        '{"name": "run_tests", "arguments": {"pattern": "test_boundary"}}\n'
        '</tool_call>'
    )


def malformed_tool_call() -> str:
    return '<tool_call>\n{"name": "run_tests", "arguments": {}}\n</tool_call>'


DEPTH_TASKS = [
    "Fix the off-by-one in binary_search and add a regression test.",
    "Refactor parse_config to reject malformed YAML without crashing.",
    "Make the pagination endpoint stable when the cursor is absent.",
    "Diagnose why the cache layer is returning stale results after a write.",
    "Make the retry logic back off exponentially and not exceed 3 attempts.",
]

TOOL_TASKS = [
    "Run the boundary tests and report the first failure.",
    "Run the pagination test suite.",
    "Run the cache integration tests and show any failures.",
]


def build_pairs(rng: random.Random) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    for task in DEPTH_TASKS:
        pairs.append({
            "prompt": prompt_text(task),
            "chosen": deep_trace(task),
            "rejected": shallow_trace(task),
        })
    for task in TOOL_TASKS:
        pairs.append({
            "prompt": prompt_text(task),
            "chosen": valid_tool_call(),
            "rejected": malformed_tool_call(),
        })
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Anthos DPO data")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    rng = random.Random(args.seed)

    pairs = build_pairs(rng)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    LOG.info("wrote %d DPO pairs to %s", len(pairs), args.out)
    LOG.info("note: use beta=0.5 with DPOTrainer for instruction-tuned models")
    return 0


if __name__ == "__main__":
    sys.exit(main())
