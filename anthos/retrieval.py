"""
anthos/retrieval.py — Real-time retrieval for Anthos (RAG layer)

Pulls live context from:
  - DuckDuckGo web search (no API key)
  - Wikipedia summary lookup

Inject the returned string into the system prompt before generation.
"""

from __future__ import annotations

import re
import textwrap
from typing import Optional


def _ddg_search(query: str, max_results: int = 3) -> list[dict]:
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    except Exception:
        return []


def _wiki_summary(topic: str, sentences: int = 4) -> str:
    try:
        import wikipedia
        wikipedia.set_lang("en")
        page = wikipedia.summary(topic, sentences=sentences, auto_suggest=True)
        return page.strip()
    except Exception:
        return ""


def retrieve(query: str, use_web: bool = True, use_wiki: bool = True,
             max_web: int = 3) -> str:
    """
    Return a context string to inject into the system prompt.
    Empty string if nothing useful found.
    """
    parts: list[str] = []

    if use_wiki:
        # Extract the core noun phrase for Wikipedia
        topic = re.sub(r'^(what is|who is|explain|tell me about|how does|define)\s+', '',
                       query, flags=re.IGNORECASE).strip().rstrip("?.")
        summary = _wiki_summary(topic)
        if summary:
            parts.append(f"[Wikipedia]\n{textwrap.fill(summary, 120)}")

    if use_web:
        results = _ddg_search(query, max_results=max_web)
        if results:
            snippets = []
            for r in results:
                title = r.get("title", "")
                body  = r.get("body", "")
                if body:
                    snippets.append(f"• {title}: {body[:300]}")
            if snippets:
                parts.append("[Web]\n" + "\n".join(snippets))

    if not parts:
        return ""

    return "\n\n".join(parts)


def needs_retrieval(query: str) -> bool:
    """
    Heuristic: does this query need live information?
    Returns True for questions about current events, prices, news, people, etc.
    """
    triggers = [
        r'\b(today|tonight|now|current(ly)?|latest|recent|this (week|month|year))\b',
        r'\b(price|cost|rate|stock|crypto|weather|news|election|score|result)\b',
        r'\b(who (is|are|was|were)|what (is|are) (the )?current)\b',
        r'\b(happened|happening|going on|update|announce|release)\b',
        r'\b\d{4}\b',  # years suggest factual/historical queries worth checking
    ]
    q = query.lower()
    return any(re.search(p, q) for p in triggers)
