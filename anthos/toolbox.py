"""
anthos/toolbox.py — Tool dispatch for Anthos

Tools available to the chat layer (not model-side tool calling —
these run in Python and inject results into context before generation):

  search(query)     — DuckDuckGo web search
  wiki(topic)       — Wikipedia summary
  calc(expr)        — Safe math evaluation
  time()            — Current date and time
  define(word)      — Dictionary-style definition via Wikipedia

Usage:
    from anthos.toolbox import detect_tool, run_tool
    tool, arg = detect_tool(user_message)
    if tool:
        result = run_tool(tool, arg)
        # inject result into context
"""

from __future__ import annotations

import re
import math
from datetime import datetime
from typing import Optional


# ── Tool detection ────────────────────────────────────────────────────────────

_TOOL_PATTERNS = [
    ("calc",   r'\b(calc(ulate)?|compute|math|what is)\s+([0-9\+\-\*\/\^\(\)\.\s]+[0-9\)])\s*[=?]?$'),
    ("time",   r'\b(what time|what date|today\'s date|current time|what day)\b'),
    ("search", r'\b(search for|look up|find me|google|what\'s happening with|latest on)\b'),
    ("wiki",   r'\b(who is|who was|what is a|define|tell me about|explain what)\b'),
]

def detect_tool(query: str) -> tuple[Optional[str], Optional[str]]:
    """Returns (tool_name, argument) or (None, None)."""
    q = query.strip()
    ql = q.lower()

    for tool, pattern in _TOOL_PATTERNS:
        m = re.search(pattern, ql)
        if m:
            if tool == "calc":
                # Extract the math expression
                expr = re.search(r'([0-9\+\-\*\/\^\(\)\.\s]+[0-9\)])', q)
                return "calc", (expr.group(1).strip() if expr else q)
            elif tool == "time":
                return "time", None
            elif tool == "search":
                # Extract what comes after the trigger
                arg = re.sub(r'^.*(search for|look up|find me|google|latest on)\s*', '', ql).strip()
                return "search", arg or q
            elif tool == "wiki":
                arg = re.sub(r'^(who is|who was|what is a|define|tell me about|explain what)\s*', '', ql).strip().rstrip("?.")
                return "wiki", arg or q
    return None, None


# ── Tool runners ──────────────────────────────────────────────────────────────

def _run_calc(expr: str) -> str:
    try:
        # Safe eval: only allow math symbols
        safe = re.sub(r'[^0-9\+\-\*\/\^\(\)\.\s]', '', expr)
        safe = safe.replace('^', '**')
        if not safe.strip():
            return "I couldn't parse that expression."
        result = eval(safe, {"__builtins__": {}, "math": math})  # noqa: S307
        return f"{expr.strip()} = {result}"
    except Exception as e:
        return f"Calculation error: {e}"


def _run_time() -> str:
    now = datetime.now()
    return f"Current date and time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}"


def _run_search(query: str) -> str:
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=4))
        if not results:
            return "No results found."
        lines = []
        for r in results:
            title = r.get("title", "")
            body  = r.get("body", "")[:250]
            href  = r.get("href", "")
            lines.append(f"• {title}\n  {body}\n  {href}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"Search unavailable: {e}"


def _run_wiki(topic: str) -> str:
    try:
        import wikipedia
        wikipedia.set_lang("en")
        summary = wikipedia.summary(topic, sentences=5, auto_suggest=True)
        return summary.strip()
    except Exception as e:
        return f"Wikipedia lookup failed: {e}"


def run_tool(tool: str, arg: Optional[str]) -> str:
    if tool == "calc":
        return _run_calc(arg or "")
    elif tool == "time":
        return _run_time()
    elif tool == "search":
        return _run_search(arg or "")
    elif tool == "wiki":
        return _run_wiki(arg or "")
    return "Unknown tool."


def format_tool_context(tool: str, result: str) -> str:
    """Wrap tool result for injection into system prompt."""
    labels = {
        "calc":   "Calculator",
        "time":   "Current Time",
        "search": "Web Search Results",
        "wiki":   "Wikipedia",
    }
    label = labels.get(tool, "Tool")
    return f"[{label}]\n{result}"
