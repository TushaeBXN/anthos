"""
harness.py — Anthos harness constants.

This module is the single source of truth for the observation-action interface
that all training data generators must use. Changing these values after training
incurs the harness coupling penalty: the model will behave incorrectly in a
harness with different formatting, block tags, or vocabulary.

Rule: every SFT example, DPO pair, and RSI trace must use ONLY the constants
defined here. Never hardcode format strings in generator scripts.

Deployment target: Qwen2.5-9B (merged) → Ollama (GGUF Q4_K_M).
Harness: local Ollama runtime with the tool schemas defined in TOOLS below.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Block tags (the "language" the model is trained to speak)
# ---------------------------------------------------------------------------

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

REFLECT_OPEN = "<reflect>"       # mandatory in failure-repair examples
REFLECT_CLOSE = "</reflect>"

ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"

TOOL_CALL_OPEN = "<tool_call>"   # always followed by exactly one JSON object
TOOL_CALL_CLOSE = "</tool_call>"

TOOL_RESPONSE_OPEN = "<tool_response>"   # wraps tool result in user turn
TOOL_RESPONSE_CLOSE = "</tool_response>"

# ---------------------------------------------------------------------------
# Gate vocabulary
# ---------------------------------------------------------------------------

GATE_AUTO    = "AUTO"     # apply without asking: reversible, local, low blast-radius
GATE_CONFIRM = "CONFIRM"  # escalate first: irreversible, shared, or high blast-radius

DECISION_LINE_AUTO    = f"Decision: {GATE_AUTO} — apply the fix without asking."
DECISION_LINE_CONFIRM = f"Decision: {GATE_CONFIRM} — confirm before acting."

# ---------------------------------------------------------------------------
# Trigger-driven null turn (the user turn that starts an observation episode)
# ---------------------------------------------------------------------------

NULL_USER_TURN = "continue"

# ---------------------------------------------------------------------------
# System prompt (identical across all training examples)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are Anthos. You reason before acting. You plan, call tools, "
    "observe results, and self-correct. You do not give shallow answers. "
    "Be direct and blunt. No hedging, no filler."
)

# ---------------------------------------------------------------------------
# Tool definitions (what Anthos can call in the Ollama runtime)
# If you add or remove tools here, retrain — the coupling is in the weights.
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a file from the workspace",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
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
        "name": "search_docs",
        "description": "Search technical documentation",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "source": {"type": "string"}},
            "required": ["query"],
        },
    }},
]

TOOL_NAMES: set[str] = {t["function"]["name"] for t in TOOLS}

# ---------------------------------------------------------------------------
# ChatML rendering helpers
# ---------------------------------------------------------------------------

import json as _json


def tool_block(tools: list[dict] | None = None) -> str:
    if tools is None:
        tools = TOOLS
    return (
        "# Tools\n\nYou have access to the following functions:\n\n<tools>\n"
        + "\n".join(_json.dumps(t) for t in tools)
        + "\n</tools>\n\n"
        f"If you call a function, emit exactly:\n"
        f"{TOOL_CALL_OPEN}\n{{\"name\": \"...\", \"arguments\": {{...}}}}\n{TOOL_CALL_CLOSE}\n"
        f"and nothing after it until the tool result arrives."
    )


def chatml(messages: list[dict[str, str]]) -> str:
    return "\n".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in messages)


def system_message(include_tools: bool = True) -> dict[str, str]:
    content = SYSTEM_PROMPT
    if include_tools:
        content += "\n\n" + tool_block()
    return {"role": "system", "content": content}


def think(content: str) -> str:
    return f"{THINK_OPEN}\n{content.strip()}\n{THINK_CLOSE}"


def reflect(content: str) -> str:
    return f"{REFLECT_OPEN}\n{content.strip()}\n{REFLECT_CLOSE}"


def answer(content: str) -> str:
    return f"{ANSWER_OPEN}\n{content.strip()}\n{ANSWER_CLOSE}"


def tool_call(name: str, arguments: dict) -> str:
    return f"{TOOL_CALL_OPEN}\n{_json.dumps({'name': name, 'arguments': arguments})}\n{TOOL_CALL_CLOSE}"


def tool_response(content: str) -> str:
    return f"{TOOL_RESPONSE_OPEN}\n{content.strip()}\n{TOOL_RESPONSE_CLOSE}"


def decision_auto(reasoning: str) -> str:
    return f"Reversibility: {reasoning}.\n{DECISION_LINE_AUTO}"


def decision_confirm(reasoning: str) -> str:
    return f"Reversibility: {reasoning}.\n{DECISION_LINE_CONFIRM}"
