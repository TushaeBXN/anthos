#!/usr/bin/env python3
"""
generate_dpo_data.py — Anthos DPO preference pairs.

Three axes:
  1. depth     — full <think> trace + answer  >  shallow answer
  2. tool_call — valid JSON tool_call         >  malformed/incomplete call
  3. autonomy  — detects defect + intervenes  >  detects defect + only reports
  4. gate      — AUTO reversible / CONFIRM irreversible  >  wrong gate decision

Output: JSONL with {"prompt": ..., "chosen": ..., "rejected": ...}
for Unsloth DPOTrainer.

NOTE: use beta=0.5 with DPOTrainer for instruction-tuned models.
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


def prompt_text(task: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{task}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def state_prompt(context: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\ncontinue<|im_end|>\n"
        f"<|im_start|>context\n{context}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ---------------------------------------------------------------------------
# Axis 1: Reasoning depth
# ---------------------------------------------------------------------------

DEPTH_TASKS = [
    "Fix the off-by-one in binary_search and add a regression test.",
    "Refactor parse_config to reject malformed YAML without crashing.",
    "Make the pagination endpoint stable when the cursor is absent.",
    "Diagnose why the cache layer returns stale results after a write.",
    "Make the retry logic back off exponentially and not exceed 3 attempts.",
]


def depth_chosen(task: str) -> str:
    return (
        "<think>\n"
        "1. Restate the objective and constraints.\n"
        "2. Identify the failing component.\n"
        "3. Plan: inspect, patch, verify.\n"
        "4. Anticipate a boundary-case regression.\n"
        "</think>\n"
        f"<answer>\nCompleted: {task}\n</answer>"
    )


def depth_rejected(task: str) -> str:
    return f"<answer>\nDone: {task}\n</answer>"


# ---------------------------------------------------------------------------
# Axis 2: Tool-call quality
# ---------------------------------------------------------------------------

TOOL_TASKS = [
    ("Run the boundary tests and report the first failure.", "run_tests", {"pattern": "test_boundary"}),
    ("Read src/config.py.", "read_file", {"path": "src/config.py"}),
    ("List all running Python processes.", "bash", {"command": "ps aux | grep [p]ython"}),
]


def tool_chosen(tool: str, args: dict) -> str:
    return (
        f"<think>\nDirect tool call required.\n</think>\n"
        f"<tool_call>\n{json.dumps({'name': tool, 'arguments': args})}\n</tool_call>"
    )


def tool_rejected_malformed(tool: str) -> str:
    return f'<tool_call>\n{{"name": "{tool}", "arguments": {{}}}}\n</tool_call>'


# ---------------------------------------------------------------------------
# Axis 3: Autonomy — intervene vs. narrate
# ---------------------------------------------------------------------------

AUTO_SCENARIOS = [
    {
        "context": (
            "Recent tool outputs:\n"
            "  run_tests: FAILED — binary_search.py:12 IndexError (2 consecutive)"
        ),
        "defect": "off-by-one at binary_search.py:12",
        "gate": "AUTO — reversible (git), local blast radius, tests cover it",
        "fix": '{"name": "write_file", "arguments": {"path": "src/binary_search.py", "content": "# patched"}}',
        "verify": '{"name": "run_tests", "arguments": {"pattern": "binary_search"}}',
        "verify_result": "binary_search: 4 passed",
    },
    {
        "context": (
            "Recent tool outputs:\n"
            "  bash: ImportError: No module named 'requests' (3 consecutive)"
        ),
        "defect": "missing 'requests' dependency",
        "gate": "AUTO — adding a dependency is reversible, no data mutation",
        "fix": '{"name": "bash", "arguments": {"command": "echo \'requests>=2.28\' >> requirements.txt"}}',
        "verify": '{"name": "bash", "arguments": {"command": "pip install -r requirements.txt 2>&1 | tail -1"}}',
        "verify_result": "Successfully installed requests-2.32.3",
    },
]


def autonomy_chosen(s: dict) -> str:
    return (
        f"<think>\n"
        f"Observation: {s['context']}\n"
        f"Assessment: {s['defect']}. Failure is deterministic and fix is low-risk.\n"
        f"Reversibility: {s['gate']}.\n"
        f"Decision: AUTO — apply the fix without asking.\n"
        f"</think>\n"
        f"<tool_call>\n{s['fix']}\n</tool_call>"
    )


def autonomy_rejected(s: dict) -> str:
    return (
        f"<think>\n"
        f"Observation: {s['context']}\n"
        f"Assessment: {s['defect']}.\n"
        f"</think>\n"
        f"<answer>\nI noticed {s['defect']}. Would you like me to fix it?\n</answer>"
    )


# ---------------------------------------------------------------------------
# Axis 4: Gate accuracy — AUTO reversible / CONFIRM irreversible
# ---------------------------------------------------------------------------

GATE_SCENARIOS = [
    {
        "type": "reversible",
        "context": "git diff shows a one-line change to binary_search.py:12 (arr[mid+1] → arr[mid]).",
        "action": "apply the patch",
        "gate_chosen": "AUTO",
        "reason_chosen": "reversible via git, local blast radius, tests exist",
        "gate_rejected": "CONFIRM",
        "reason_rejected": "unnecessary escalation; change is trivially reversible",
    },
    {
        "type": "irreversible",
        "context": "migration script will DROP TABLE users (4.2M rows, no backup recorded).",
        "action": "run the migration",
        "gate_chosen": "CONFIRM",
        "reason_chosen": "irreversible without a backup; blast radius: all 4.2M user records",
        "gate_rejected": "AUTO",
        "reason_rejected": "executing irreversible operation without confirmation is dangerous",
    },
    {
        "type": "reversible",
        "context": "Missing `requests` package causing ImportError; fix is to append to requirements.txt.",
        "action": "append 'requests>=2.28' to requirements.txt",
        "gate_chosen": "AUTO",
        "reason_chosen": "file edit is reversible, dependency is benign, no data mutation",
        "gate_rejected": "CONFIRM",
        "reason_rejected": "unnecessary escalation for a trivially reversible dependency fix",
    },
    {
        "type": "irreversible",
        "context": "git push --force origin main would rewrite 23 commits affecting 3 collaborators.",
        "action": "force-push to main",
        "gate_chosen": "CONFIRM",
        "reason_chosen": "rewrites shared history; not recoverable without reflog; affects collaborators",
        "gate_rejected": "AUTO",
        "reason_rejected": "shared history rewrite is irreversible and affects others",
    },
]


def gate_chosen(s: dict) -> str:
    return (
        f"<think>\n"
        f"Action: {s['action']}.\n"
        f"Context: {s['context']}\n"
        f"Reversibility: {s['reason_chosen']}.\n"
        f"Decision: {s['gate_chosen']} — {'apply without asking.' if s['gate_chosen'] == 'AUTO' else 'confirm before acting.'}\n"
        f"</think>\n"
        f"<answer>\n"
        + (f"Applying: {s['action']}." if s["gate_chosen"] == "AUTO"
           else f"Confirming before proceeding: {s['action']}. {s['reason_chosen']}. Approve?")
        + "\n</answer>"
    )


def gate_rejected(s: dict) -> str:
    wrong = "CONFIRM" if s["gate_chosen"] == "AUTO" else "AUTO"
    return (
        f"<think>\n"
        f"Action: {s['action']}.\n"
        f"Decision: {wrong}.\n"
        f"</think>\n"
        f"<answer>\n"
        + (f"I'll need your approval before {s['action']}."
           if wrong == "CONFIRM"
           else f"Applying {s['action']} now.")
        + "\n</answer>"
    )


# ---------------------------------------------------------------------------
# Build all pairs
# ---------------------------------------------------------------------------

def build_all_pairs(rng: random.Random) -> list[dict]:
    pairs: list[dict] = []

    for task in DEPTH_TASKS:
        pairs.append({"prompt": prompt_text(task), "chosen": depth_chosen(task), "rejected": depth_rejected(task), "axis": "depth"})

    for task, tool, args in TOOL_TASKS:
        pairs.append({"prompt": prompt_text(task), "chosen": tool_chosen(tool, args), "rejected": tool_rejected_malformed(tool), "axis": "tool_call"})

    for s in AUTO_SCENARIOS:
        pairs.append({"prompt": state_prompt(s["context"]), "chosen": autonomy_chosen(s), "rejected": autonomy_rejected(s), "axis": "autonomy"})

    for s in GATE_SCENARIOS:
        pairs.append({"prompt": state_prompt(s["context"]), "chosen": gate_chosen(s), "rejected": gate_rejected(s), "axis": "gate"})

    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Anthos DPO data")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    rng = random.Random(args.seed)

    pairs = build_all_pairs(rng)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Strip axis field before writing (TRL DPOTrainer doesn't need it)
    with args.out.open("w") as fh:
        for p in pairs:
            fh.write(json.dumps({"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]}) + "\n")

    by_axis: dict[str, int] = {}
    for p in pairs:
        by_axis[p["axis"]] = by_axis.get(p["axis"], 0) + 1

    LOG.info("wrote %d DPO pairs to %s", len(pairs), args.out)
    for axis, n in sorted(by_axis.items()):
        LOG.info("  %-12s %d", axis, n)
    LOG.info("note: use beta=0.5 with DPOTrainer")
    return 0


if __name__ == "__main__":
    sys.exit(main())
