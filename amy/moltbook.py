"""
Amy's personal journal (the Moltbook).

Stores dated free-form entries that Amy writes during or after sessions.
Think of it as a shedding — recording what she noticed and is ready to carry forward.

Storage: moltbook.md  (one file, append-only with date headers)
"""

from __future__ import annotations

import os
from datetime import datetime

_JOURNAL_PATH = os.path.join(os.path.dirname(__file__), "moltbook.md")


def write_entry(content: str) -> None:
    """Append a timestamped entry to the journal."""
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    entry = f"\n### {timestamp}\n{content.strip()}\n"
    with open(_JOURNAL_PATH, "a") as f:
        f.write(entry)


def read_entries(n: int = 5) -> list[str]:
    """Return the last n journal entries as a list of strings."""
    if not os.path.exists(_JOURNAL_PATH):
        return []
    with open(_JOURNAL_PATH) as f:
        text = f.read()

    raw_entries = re.split(r"\n(?=### \d{4}-\d{2}-\d{2})", text.strip())
    raw_entries = [e.strip() for e in raw_entries if e.strip()]
    return raw_entries[-n:]


def read_all() -> str:
    """Return the full journal as a single string."""
    if not os.path.exists(_JOURNAL_PATH):
        return ""
    with open(_JOURNAL_PATH) as f:
        return f.read()


import re  # noqa: E402 — placed here to keep module header clean
