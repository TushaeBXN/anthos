#!/usr/bin/env python3
"""
generate_sft_data.py — Anthos SFT data generator.

Produces ChatML-formatted reasoning-trace and tool-call examples.
Every assistant turn carries a <think> block; shallow answers are rejected.

Format: Qwen2.5 ChatML with Hermes-style tags
  <think>...</think>              — mandatory reasoning block
  <tool_call>{...}</tool_call>    — structured JSON tool invocation
  <tool_response>...</tool_response> — tool result (user turn)
  <answer>...</answer>            — final response

Output: JSONL with "text" field, ready for Unsloth SFTTrainer
  (dataset_text_field="text").

Usage:
    python3 scripts/data/generate_sft_data.py --out data/anthos_sft.jsonl --coding 800
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
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

DEBUG_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the workspace",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run the test suite and return stdout/stderr",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
]


def tool_block(tools: list[dict[str, Any]]) -> str:
    return (
        "# Tools\n\nYou have access to the following functions:\n\n"
        "<tools>\n"
        + "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)
        + "\n</tools>\n\n"
        "If you call a function, emit exactly:\n"
        "<tool_call>\n{\"name\": \"...\", \"arguments\": {...}}\n</tool_call>\n"
        "and nothing after it until the tool result arrives."
    )


def chatml(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
    return "\n".join(parts)


def build_example(
    task: str,
    tools: list[dict[str, Any]],
    plan_steps: list[str],
    tool_name: str | None,
    tool_args: dict[str, Any] | None,
    tool_result: str | None,
    correction: str | None,
    final_answer: str,
) -> dict[str, Any]:
    system_content = SYSTEM_PROMPT
    if tools:
        system_content += "\n\n" + tool_block(tools)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": task},
    ]

    think_lines = ["<think>"] + [f"{i + 1}. {s}" for i, s in enumerate(plan_steps)] + ["</think>"]
    assistant_first = "\n".join(think_lines)

    if tool_name and tool_args is not None:
        assistant_first += (
            "\n<tool_call>\n"
            + json.dumps({"name": tool_name, "arguments": tool_args}, ensure_ascii=False)
            + "\n</tool_call>"
        )

    messages.append({"role": "assistant", "content": assistant_first})

    if tool_result is not None:
        messages.append({"role": "user", "content": f"<tool_response>\n{tool_result}\n</tool_response>"})

        second_think = "<think>\n"
        if correction:
            second_think += f"The previous step produced: {tool_result}\n"
            second_think += f"Correction: {correction}\n"
        else:
            second_think += f"Tool returned: {tool_result}\nProceeding to synthesize.\n"
        second_think += "</think>"
        assistant_second = second_think + f"\n<answer>\n{final_answer}\n</answer>"
        messages.append({"role": "assistant", "content": assistant_second})
    else:
        messages[-1]["content"] += f"\n<answer>\n{final_answer}\n</answer>"

    return {"text": chatml(messages), "messages": messages}


CODING_TASKS = [
    ("Fix the off-by-one in this binary search and explain the failure.", "binary_search_fix"),
    ("Refactor this function to avoid mutating the caller's list.", "list_mutation"),
    ("Make this SQL query safe against injection.", "sql_injection"),
    ("The test passes locally but fails in CI. Diagnose the likely cause.", "ci_flake"),
    ("Convert this recursive Fibonacci to an iterative memoised version.", "fib_memo"),
    ("The cache returns stale values after a write. Find the bug.", "cache_stale"),
    ("The retry logic doesn't back off. Fix it to use exponential backoff.", "retry_backoff"),
    ("Pagination breaks when cursor is null. Make it stable.", "pagination_null"),
]

PLAN_VARIANTS = [
    ["Restate the objective and constraints.", "Read the relevant code.", "Identify the smallest change.", "Verify with a test."],
    ["Inspect the failure.", "Form a hypothesis.", "Apply the fix.", "Re-run the failing case."],
    ["Map inputs to expected outputs.", "Locate the divergence.", "Patch and add a regression test."],
    ["State the invariant that must hold.", "Find where it breaks.", "Fix and verify the boundary case."],
]


def generate_coding_examples(n: int, rng: random.Random) -> list[dict[str, Any]]:
    examples = []
    for _ in range(n):
        task, tag = rng.choice(CODING_TASKS)
        tools = DEBUG_TOOLS if rng.random() < 0.8 else []
        plan = rng.choice(PLAN_VARIANTS)
        tool = rng.choice(["read_file", "run_tests"]) if tools else None
        args = {"path": f"src/{tag}.py"} if tool == "read_file" else {"pattern": tag}
        result = (
            f"{tag}.py:12: AssertionError: boundary case failed"
            if tool == "run_tests"
            else f"def {tag}(...):\n    # line 12: suspicious index"
        )
        correction = f"Line 12 is the culprit; adjust the boundary condition."
        final = f"Patched {tag}. Added a regression test for the boundary case."
        examples.append(
            build_example(task, tools, plan, tool, args, result if tool else None, correction if tool else None, final)
        )
    return examples


def validate_example(ex: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    messages = ex.get("messages", [])
    assistant_turns = [m for m in messages if m["role"] == "assistant"]
    if not assistant_turns:
        errors.append("no assistant turns")
    for i, turn in enumerate(assistant_turns):
        if not THINK_RE.search(turn["content"]):
            errors.append(f"assistant turn {i} lacks <think>")
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Anthos SFT data")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--coding", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    rng = random.Random(args.seed)

    examples = generate_coding_examples(args.coding, rng)

    valid: list[dict[str, Any]] = []
    rejected = 0
    for ex in examples:
        errs = validate_example(ex)
        if errs:
            rejected += 1
            LOG.warning("rejected example: %s", errs)
            continue
        valid.append(ex)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for ex in valid:
            fh.write(json.dumps({"text": ex["text"]}, ensure_ascii=False) + "\n")

    LOG.info("wrote %d examples, rejected %d", len(valid), rejected)
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
