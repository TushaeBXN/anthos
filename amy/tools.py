"""
Tool definitions available to Amy during a session.

Each tool is a callable that accepts typed arguments and returns a string
result.  companion.py dispatches tool calls by name.
"""

from __future__ import annotations

import json
from typing import Callable

from amy import web, memory, moltbook

# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Callable] = {}


def register(name: str):
    def decorator(fn: Callable) -> Callable:
        _REGISTRY[name] = fn
        return fn
    return decorator


def call(name: str, **kwargs) -> str:
    if name not in _REGISTRY:
        return f"[tool error] unknown tool: {name}"
    try:
        return _REGISTRY[name](**kwargs)
    except Exception as exc:
        return f"[tool error] {name} failed: {exc}"


def available_tools() -> list[dict]:
    return [
        {"name": "web_search", "description": "Search the web for current information.", "params": ["query"]},
        {"name": "recall", "description": "Search Amy's episodic memory.", "params": ["query", "n"]},
        {"name": "journal_read", "description": "Read entries from Amy's personal journal.", "params": ["n_entries"]},
        {"name": "journal_write", "description": "Write a new entry to Amy's journal.", "params": ["content"]},
        {"name": "web_fetch", "description": "Fetch and read the text content of a URL.", "params": ["url"]},
    ]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

@register("web_search")
def tool_web_search(query: str) -> str:
    results = web.search(query)
    if not results:
        return "No results found."
    return "\n".join(f"- {r['title']}: {r['snippet']}" for r in results[:5])


@register("web_fetch")
def tool_web_fetch(url: str) -> str:
    return web.fetch_text(url, max_chars=3000)


@register("recall")
def tool_recall(query: str, n: int = 5) -> str:
    results = memory.query_memory(query, n=int(n))
    if not results:
        return "No relevant memories found."
    lines = [f"[{r['session_id']} / {r['role']}] {r['content'][:200]}" for r in results]
    return "\n".join(lines)


@register("journal_read")
def tool_journal_read(n_entries: int = 5) -> str:
    entries = moltbook.read_entries(n=int(n_entries))
    if not entries:
        return "Journal is empty."
    return "\n\n".join(entries)


@register("journal_write")
def tool_journal_write(content: str) -> str:
    moltbook.write_entry(content)
    return "Journal entry saved."
