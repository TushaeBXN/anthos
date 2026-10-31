"""
Web search and fetch utilities for Amy's tool layer.

Uses DuckDuckGo's lite HTML endpoint for search (no API key required)
and requests + html2text for page fetching.
"""

from __future__ import annotations

import re
import urllib.parse

_REQUESTS_OK = False
try:
    import requests
    _REQUESTS_OK = True
except ImportError:
    pass


_DDG_URL = "https://html.duckduckgo.com/html/"
_HEADERS = {"User-Agent": "Amy-Companion/1.0 (educational AI assistant)"}


def search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search DuckDuckGo and return a list of {title, url, snippet} dicts.
    Returns [] if requests is unavailable or the request fails.
    """
    if not _REQUESTS_OK:
        return []
    try:
        resp = requests.post(
            _DDG_URL,
            data={"q": query, "kl": "us-en"},
            headers=_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
    except Exception:
        return []

    results = []
    pattern = re.compile(
        r'class="result__title".*?href="([^"]+)"[^>]*>(.*?)</a>.*?'
        r'class="result__snippet"[^>]*>(.*?)</span>',
        re.DOTALL,
    )
    for m in pattern.finditer(resp.text):
        url = urllib.parse.unquote(m.group(1))
        title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
        snippet = re.sub(r"<[^>]+>", "", m.group(3)).strip()
        if url.startswith("http"):
            results.append({"title": title, "url": url, "snippet": snippet})
        if len(results) >= max_results:
            break
    return results


def fetch_text(url: str, max_chars: int = 4000) -> str:
    """
    Fetch a URL and return its readable text content (tags stripped).
    Returns an error string on failure.
    """
    if not _REQUESTS_OK:
        return "[web fetch unavailable: requests not installed]"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as exc:
        return f"[fetch error: {exc}]"

    text = re.sub(r"<script[^>]*>.*?</script>", "", resp.text, flags=re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]
