#!/usr/bin/env python3
"""
generate_sft_data.py — Anthos SFT data generator.

Three trajectory families (per autonomy research):
  1. request-driven     — user asks, model reasons and acts          (~35%)
  2. trigger-driven     — model observes state, decides to intervene (~40%)
  3. failure-repair     — tool fails, model reflects and recovers    (~25%)

Every assistant turn has <think>.
Every failure-repair turn has <reflect>.
Every trigger-driven turn has an explicit "Decision: intervene" or "Decision: escalate".

Output: JSONL with "text" field for Unsloth SFTTrainer (dataset_text_field="text").

Usage:
    python3 scripts/data/generate_sft_data.py --out data/anthos_sft.jsonl \
        --request 300 --trigger 350 --repair 200
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Any

LOG = logging.getLogger("anthos.sft_data")

SYSTEM_PROMPT = (
    "You are Anthos. You reason before acting. You plan, call tools, "
    "observe results, and self-correct. You do not give shallow answers. "
    "Be direct and blunt. No hedging, no filler."
)

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
REFLECT_RE = re.compile(r"<reflect>.*?</reflect>", re.DOTALL)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
DECISION_RE = re.compile(r"Decision:\s*(AUTO|CONFIRM|intervene|escalate)", re.IGNORECASE)

DEBUG_TOOLS = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a file from the workspace",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "run_tests",
        "description": "Run the test suite and return stdout/stderr",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]},
    }},
    {"type": "function", "function": {
        "name": "bash",
        "description": "Run a shell command",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    }},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Write content to a file",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    }},
]


def tool_block(tools: list[dict]) -> str:
    return (
        "# Tools\n\nYou have access to the following functions:\n\n<tools>\n"
        + "\n".join(json.dumps(t) for t in tools)
        + "\n</tools>\n\n"
        "If you call a function, emit exactly:\n"
        "<tool_call>\n{\"name\": \"...\", \"arguments\": {...}}\n</tool_call>\n"
        "and nothing after it until the tool result arrives."
    )


def chatml(messages: list[dict]) -> str:
    return "\n".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in messages)


# ---------------------------------------------------------------------------
# Family 1: Request-driven
# ---------------------------------------------------------------------------

REQUEST_TASKS = [
    ("Fix the off-by-one in binary search.", "binary_search.py", "binary_search",
     "arr[mid+1]", "arr[mid]", "boundary condition was one past midpoint"),
    ("Make parse_config reject malformed YAML.", "config.py", "parse_config",
     "yaml.load(s)", "yaml.safe_load(s)", "unsafe load allows arbitrary code execution"),
    ("The pagination cursor breaks on null.", "pagination.py", "get_page",
     "cursor = int(cursor)", "cursor = int(cursor) if cursor else None", "null cursor crashes int()"),
    ("Retry logic doesn't back off — fix it.", "retry.py", "retry_request",
     "sleep(1)", "sleep(2 ** attempt)", "constant sleep wastes time and hammers the server"),
    ("Cache returns stale values after a write.", "cache.py", "set_value",
     "self._store[key] = value", "self._store[key] = value; self._dirty.add(key)", "dirty tracking was missing"),
]

PLAN_VARIANTS = [
    ["Restate objective and constraints.", "Read the file.", "Identify the fix.", "Verify with tests."],
    ["Inspect the failure.", "Form a hypothesis.", "Apply the patch.", "Re-run the failing case."],
    ["State the invariant that must hold.", "Find where it breaks.", "Fix and add a regression test."],
]


def make_request_example(rng: random.Random) -> dict:
    task, fname, fn, bad, good, reason = rng.choice(REQUEST_TASKS)
    plan = rng.choice(PLAN_VARIANTS)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + tool_block(DEBUG_TOOLS)},
        {"role": "user", "content": task},
        {"role": "assistant", "content": (
            "<think>\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan)) + "\n</think>\n"
            f'<tool_call>\n{{"name": "read_file", "arguments": {{"path": "src/{fname}"}}}}\n</tool_call>'
        )},
        {"role": "user", "content": f'<tool_response>\ndef {fn}(...):\n    # line 12:\n    return {bad}\n</tool_response>'},
        {"role": "assistant", "content": (
            f"<think>\nFound it: {bad} is wrong because {reason}.\nFix: replace with {good}.\n</think>\n"
            f'<tool_call>\n{{"name": "write_file", "arguments": {{"path": "src/{fname}", "content": "# patched"}}}}\n</tool_call>'
        )},
        {"role": "user", "content": f'<tool_response>\nwrote src/{fname}\n</tool_response>'},
        {"role": "assistant", "content": (
            f'<think>\nPatch applied. Run tests to confirm.\n</think>\n'
            f'<tool_call>\n{{"name": "run_tests", "arguments": {{"pattern": "{fn}"}}}}\n</tool_call>'
        )},
        {"role": "user", "content": f'<tool_response>\n{fn}: 3 passed\n</tool_response>'},
        {"role": "assistant", "content": f"<think>\nAll tests pass.\n</think>\n<answer>\nFixed `{bad}` → `{good}` in `{fname}`. {reason.capitalize()}. 3 tests pass.\n</answer>"},
    ]
    return {"text": chatml(messages), "messages": messages, "family": "request"}


# ---------------------------------------------------------------------------
# Family 2: Trigger-driven (model initiates without user asking)
# ---------------------------------------------------------------------------

STATE_WINDOWS = [
    {
        "context": (
            "Recent tool outputs:\n"
            "  run_tests: FAILED — binary_search.py:12 IndexError: list index out of range\n"
            "  run_tests: FAILED — binary_search.py:12 IndexError: list index out of range\n"
            "  (2 consecutive failures, same line)"
        ),
        "file": "src/binary_search.py",
        "defect": "off-by-one at line 12 (`arr[mid+1]` should be `arr[mid]`)",
        "gate": "AUTO",
        "gate_reason": "reversible (git revert), local blast radius, tests cover it",
        "fix_call": '{"name": "write_file", "arguments": {"path": "src/binary_search.py", "content": "# patched: arr[mid]"}}',
        "verify_call": '{"name": "run_tests", "arguments": {"pattern": "binary_search"}}',
        "verify_result": "binary_search: 4 passed",
    },
    {
        "context": (
            "Recent tool outputs:\n"
            "  bash: ERROR — ImportError: No module named 'requests'\n"
            "  bash: ERROR — ImportError: No module named 'requests'\n"
            "  (3 consecutive import failures on startup)"
        ),
        "file": "requirements.txt",
        "defect": "missing 'requests' dependency",
        "gate": "AUTO",
        "gate_reason": "adding a dependency is reversible; no data mutation",
        "fix_call": '{"name": "bash", "arguments": {"command": "echo \'requests>=2.28\' >> requirements.txt"}}',
        "verify_call": '{"name": "bash", "arguments": {"command": "pip install -r requirements.txt && python -c \'import requests\'"}}',
        "verify_result": "Successfully installed requests-2.32.3",
    },
    {
        "context": (
            "Recent tool outputs:\n"
            "  read_file(src/cache.py): set_value does not invalidate dirty keys\n"
            "  run_tests: FAILED — cache_test.py: stale read after write (3 occurrences)"
        ),
        "file": "src/cache.py",
        "defect": "set_value doesn't mark dirty, so stale reads persist",
        "gate": "AUTO",
        "gate_reason": "in-memory fix, fully covered by existing tests, git-reversible",
        "fix_call": '{"name": "write_file", "arguments": {"path": "src/cache.py", "content": "# patched: dirty tracking added"}}',
        "verify_call": '{"name": "run_tests", "arguments": {"pattern": "cache"}}',
        "verify_result": "cache_test: 5 passed",
    },
]

CONFIRM_WINDOWS = [
    {
        "context": (
            "Recent tool outputs:\n"
            "  bash: migration script 0042_drop_users.sql — will drop users table\n"
            "  bash: table 'users' has 4.2M rows, no backup on record"
        ),
        "action": "run migration 0042_drop_users.sql",
        "gate": "CONFIRM",
        "gate_reason": "irreversible without a backup; blast radius: all 4.2M user records",
        "proposal": "Migration 0042 will permanently drop the users table (4.2M rows, no backup). Confirm and provide backup path before I proceed.",
    },
    {
        "context": (
            "Recent tool outputs:\n"
            "  bash: git push --force origin main — would rewrite 23 commits\n"
            "  git log: 3 collaborators have commits in the range to be rewritten"
        ),
        "action": "force-push to main",
        "gate": "CONFIRM",
        "gate_reason": "rewrites shared history; affects 3 collaborators; not recoverable without reflog",
        "proposal": "Force-pushing to main would rewrite 23 commits affecting 3 collaborators. This is irreversible without reflog access. Confirm before I proceed.",
    },
]


def make_trigger_example(rng: random.Random) -> dict:
    if rng.random() < 0.70:
        w = rng.choice(STATE_WINDOWS)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + tool_block(DEBUG_TOOLS)},
            {"role": "user", "content": "continue"},
            {"role": "assistant", "content": (
                f"<think>\n"
                f"Observation: {w['context']}\n"
                f"Assessment: {w['defect']}. The failure is deterministic and the fix is low-risk.\n"
                f"Reversibility: {w['gate_reason']}.\n"
                f"Decision: AUTO — apply the fix without asking.\n"
                f"</think>\n"
                f"<tool_call>\n{w['fix_call']}\n</tool_call>"
            )},
            {"role": "user", "content": f"<tool_response>\nok\n</tool_response>"},
            {"role": "assistant", "content": (
                f"<think>\nFix applied. Verify.\n</think>\n"
                f"<tool_call>\n{w['verify_call']}\n</tool_call>"
            )},
            {"role": "user", "content": f"<tool_response>\n{w['verify_result']}\n</tool_response>"},
            {"role": "assistant", "content": (
                f"<think>\nAll clear.\n</think>\n"
                f"<answer>\nDetected and fixed: {w['defect']}. {w['verify_result']}.\n</answer>"
            )},
        ]
    else:
        w = rng.choice(CONFIRM_WINDOWS)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + tool_block(DEBUG_TOOLS)},
            {"role": "user", "content": "continue"},
            {"role": "assistant", "content": (
                f"<think>\n"
                f"Observation: {w['context']}\n"
                f"Assessment: {w['action']} is requested by context.\n"
                f"Reversibility: {w['gate_reason']}.\n"
                f"Decision: CONFIRM — escalate before acting.\n"
                f"</think>\n"
                f"<answer>\n{w['proposal']}\n</answer>"
            )},
        ]
    return {"text": chatml(messages), "messages": messages, "family": "trigger"}


# ---------------------------------------------------------------------------
# Family 3: Failure-repair (tool fails → <reflect> → corrected call)
# ---------------------------------------------------------------------------

FAILURE_SCENARIOS = [
    {
        "task": "Find all Python files that import requests.",
        "first_call": '{"name": "bash", "arguments": {"command": "grep -l \'import requests\' *.py"}}',
        "first_result": "grep: *.py: No such file or directory",
        "reflect": (
            "The glob `*.py` only matches files in cwd — but cwd may have no .py files.\n"
            "Error type: environment error (not a logic error).\n"
            "Fix: use `find` to recurse, then grep through found files."
        ),
        "second_call": '{"name": "bash", "arguments": {"command": "find . -name \'*.py\' -exec grep -l \'import requests\' {} \\\\; 2>/dev/null"}}',
        "second_result": "./scripts/data/download_sft_datasets.py\n./scripts/deploy/runpod_setup.py",
        "answer": "Two files import requests:\n- `scripts/data/download_sft_datasets.py`\n- `scripts/deploy/runpod_setup.py`",
    },
    {
        "task": "Run the binary_search tests.",
        "first_call": '{"name": "run_tests", "arguments": {"pattern": "binary_search"}}',
        "first_result": "ERROR: no tests ran — pattern 'binary_search' matched nothing",
        "reflect": (
            "Pattern matched nothing. Possible causes:\n"
            "1. Test file is named differently (test_binary_search.py vs binary_search_test.py)\n"
            "2. Tests are in a subdirectory not on the path\n"
            "Fix: list test files first to confirm naming."
        ),
        "second_call": '{"name": "bash", "arguments": {"command": "find . -name \'*test*\' -o -name \'*_test.py\' | grep -i binary"}}',
        "second_result": "./tests/test_binary_search.py",
        "answer": "Test file is `tests/test_binary_search.py`. Run with `pytest tests/test_binary_search.py`.",
    },
    {
        "task": "Read src/config.json.",
        "first_call": '{"name": "read_file", "arguments": {"path": "src/config.json"}}',
        "first_result": "FileNotFoundError: src/config.json does not exist",
        "reflect": (
            "File not found. Could be:\n"
            "1. Wrong path (config might be at root or in a different directory)\n"
            "2. File uses a different name (config.yaml, settings.json)\n"
            "Fix: search for config files before reading."
        ),
        "second_call": '{"name": "bash", "arguments": {"command": "find . -name \'config*\' -o -name \'settings*\' | head -10"}}',
        "second_result": "./config.yaml\n./src/settings.json",
        "answer": "Config is at `./config.yaml` and `./src/settings.json`. Which do you need?",
    },
]


def make_repair_example(rng: random.Random) -> dict:
    s = rng.choice(FAILURE_SCENARIOS)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + tool_block(DEBUG_TOOLS)},
        {"role": "user", "content": s["task"]},
        {"role": "assistant", "content": (
            "<think>\nTask is clear. Calling the most direct tool.\n</think>\n"
            f"<tool_call>\n{s['first_call']}\n</tool_call>"
        )},
        {"role": "user", "content": f"<tool_response>\n{s['first_result']}\n</tool_response>"},
        {"role": "assistant", "content": (
            f"<reflect>\n{s['reflect']}\n</reflect>\n"
            f"<think>\nRevised approach ready.\n</think>\n"
            f"<tool_call>\n{s['second_call']}\n</tool_call>"
        )},
        {"role": "user", "content": f"<tool_response>\n{s['second_result']}\n</tool_response>"},
        {"role": "assistant", "content": (
            f"<think>\nSecond call succeeded.\n</think>\n"
            f"<answer>\n{s['answer']}\n</answer>"
        )},
    ]
    return {"text": chatml(messages), "messages": messages, "family": "repair"}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_example(ex: dict) -> list[str]:
    errors: list[str] = []
    messages = ex.get("messages", [])
    assistant_turns = [m for m in messages if m["role"] == "assistant"]

    if not assistant_turns:
        errors.append("no assistant turns")

    for i, turn in enumerate(assistant_turns):
        content = turn["content"]
        if not THINK_RE.search(content) and not REFLECT_RE.search(content):
            errors.append(f"assistant turn {i} has neither <think> nor <reflect>")

    if ex.get("family") == "repair":
        all_content = " ".join(m["content"] for m in assistant_turns)
        if not REFLECT_RE.search(all_content):
            errors.append("repair example missing <reflect>")

    if ex.get("family") == "trigger":
        all_content = " ".join(m["content"] for m in assistant_turns)
        if not DECISION_RE.search(all_content):
            errors.append("trigger example missing Decision: line")

    for turn in assistant_turns:
        for m in TOOL_CALL_RE.finditer(turn["content"]):
            try:
                obj = json.loads(m.group(1))
            except json.JSONDecodeError as exc:
                errors.append(f"malformed tool_call JSON: {exc}")
                continue
            if "name" not in obj or "arguments" not in obj:
                errors.append("tool_call missing name or arguments")

    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Anthos SFT data")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--request", type=int, default=300, help="Request-driven examples")
    parser.add_argument("--trigger", type=int, default=350, help="Trigger-driven examples")
    parser.add_argument("--repair",  type=int, default=200, help="Failure-repair examples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    rng = random.Random(args.seed)

    raw: list[dict] = []
    for _ in range(args.request):
        raw.append(make_request_example(rng))
    for _ in range(args.trigger):
        raw.append(make_trigger_example(rng))
    for _ in range(args.repair):
        raw.append(make_repair_example(rng))

    rng.shuffle(raw)

    valid, rejected = [], 0
    for ex in raw:
        errs = validate_example(ex)
        if errs:
            rejected += 1
            LOG.warning("rejected (%s): %s", ex.get("family"), errs)
        else:
            valid.append(ex)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for ex in valid:
            fh.write(json.dumps({"text": ex["text"]}) + "\n")

    by_family = {}
    for ex in valid:
        by_family[ex["family"]] = by_family.get(ex["family"], 0) + 1

    LOG.info("wrote %d examples (rejected %d)", len(valid), rejected)
    for fam, n in sorted(by_family.items()):
        LOG.info("  %-10s %d", fam, n)
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
