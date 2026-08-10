"""
context_engineering/learnings_manager.py

WRITE LAYER  — append a structured 4-line entry to learnings.md after each session.
SELECT LAYER — load the relevant profile + last N learnings for that profile at session start.
COMPRESS LAYER — when a profile has >50 entries, summarize the oldest 25 into a dated
                 archive block and delete the raw entries.

Entry format (4 lines, preceded by a machine-readable JSON comment):
    <!-- ENTRY {"profile":..., "date":..., "domain":..., "scaffold":...,
                "ended_with_question":..., "answer_given":..., "note":...} -->
    **{date}** | `{profile}` | domain: {domain} | scaffold: {scaffold}
    - Ended ?: {yes/no} | Given away: {yes/no}
    - Note: {note}
"""

from __future__ import annotations

import json
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LEARNINGS_FILE = Path(__file__).parent.parent / "learnings.md"
COMPRESS_THRESHOLD = 50
COMPRESS_KEEP_OLDEST = 25

_ENTRY_RE = re.compile(
    r"<!-- ENTRY (\{.*?\}) -->\n(.*?)\n(.*?)\n(.*?)(?=\n<!-- ENTRY |\n<!-- ARCHIVE |\Z)",
    re.DOTALL,
)
_ARCHIVE_RE = re.compile(
    r"<!-- ARCHIVE_START profile:(\S+) date:(\S+) count:(\d+) -->\n(.*?)<!-- ARCHIVE_END -->",
    re.DOTALL,
)


# ─────────────────────────────────────────────────────────────────────────────
# WRITE LAYER
# ─────────────────────────────────────────────────────────────────────────────

def append_learning(
    profile_id: str,
    domain: str,
    scaffold: str,
    ended_with_question: bool,
    answer_given: bool,
    note: str,
    date: str | None = None,
) -> None:
    """Append one 4-line structured learning entry to learnings.md."""
    if date is None:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    note_short = note[:120].replace("\n", " ")
    scaffold_slug = scaffold.replace(" ", "_")[:40]

    meta: dict[str, Any] = {
        "profile": profile_id,
        "date": date,
        "domain": domain,
        "scaffold": scaffold_slug,
        "ended_with_question": ended_with_question,
        "answer_given": answer_given,
        "note": note_short,
    }

    eq = "yes" if ended_with_question else "no"
    ag = "yes" if answer_given else "no"

    entry = (
        f"<!-- ENTRY {json.dumps(meta, separators=(',', ':'))} -->\n"
        f"**{date}** | `{profile_id}` | domain: {domain} | scaffold: {scaffold_slug}\n"
        f"- Ended ?: {eq} | Given away: {ag}\n"
        f"- Note: {note_short}\n"
    )

    LEARNINGS_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not LEARNINGS_FILE.exists():
        LEARNINGS_FILE.write_text("# Tutor Learnings Log\n\n", encoding="utf-8")

    with LEARNINGS_FILE.open("a", encoding="utf-8") as f:
        f.write(entry + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# SELECT LAYER
# ─────────────────────────────────────────────────────────────────────────────

def load_learnings(profile_id: str, n: int = 10) -> list[dict[str, Any]]:
    """
    Return the last n learning entries for profile_id.
    Other profiles' entries are never loaded into the same context.
    """
    if not LEARNINGS_FILE.exists():
        return []

    text = LEARNINGS_FILE.read_text(encoding="utf-8")
    entries = _parse_entries(text)
    profile_entries = [e for e in entries if e.get("profile") == profile_id]
    return profile_entries[-n:]


def format_learnings_for_context(learnings: list[dict[str, Any]]) -> str:
    """Convert loaded learnings into a compact context block for the system prompt."""
    if not learnings:
        return ""

    lines = ["[RECENT LEARNINGS FOR THIS PROFILE]"]
    for e in learnings:
        eq = "yes" if e.get("ended_with_question") else "no"
        ag = "yes" if e.get("answer_given") else "no"
        lines.append(
            f"  {e.get('date','')} | {e.get('domain','')} | scaffold:{e.get('scaffold','')} "
            f"| ended?:{eq} | given_away:{ag}"
        )
        if e.get("note"):
            note = textwrap.shorten(e["note"], width=80, placeholder="...")
            lines.append(f"    note: {note}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# COMPRESS LAYER
# ─────────────────────────────────────────────────────────────────────────────

def count_raw_entries(profile_id: str) -> int:
    if not LEARNINGS_FILE.exists():
        return 0
    text = LEARNINGS_FILE.read_text(encoding="utf-8")
    return sum(1 for e in _parse_entries(text) if e.get("profile") == profile_id)


def compress_if_needed(profile_id: str) -> bool:
    """
    If a profile has >50 raw entries, summarize the oldest 25 into an archive block.
    Returns True if compression was performed.
    """
    if not LEARNINGS_FILE.exists():
        return False

    text = LEARNINGS_FILE.read_text(encoding="utf-8")
    all_entries = _parse_entries(text)
    profile_raw = [e for e in all_entries if e.get("profile") == profile_id]

    if len(profile_raw) <= COMPRESS_THRESHOLD:
        return False

    oldest = profile_raw[:COMPRESS_KEEP_OLDEST]
    _archive_entries(profile_id, oldest, text)
    return True


def _archive_entries(
    profile_id: str,
    to_archive: list[dict[str, Any]],
    original_text: str,
) -> None:
    dates = [e.get("date", "") for e in to_archive if e.get("date")]
    date_range = f"{min(dates)} to {max(dates)}" if dates else "unknown"
    archive_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    count = len(to_archive)

    # Build summary stats
    domains = {}
    scaffolds: dict[str, int] = {}
    ended_q = sum(1 for e in to_archive if e.get("ended_with_question"))
    given_away = sum(1 for e in to_archive if e.get("answer_given"))
    notes = [e.get("note", "") for e in to_archive if e.get("note")]

    for e in to_archive:
        d = e.get("domain", "unknown")
        domains[d] = domains.get(d, 0) + 1
        s = e.get("scaffold", "unknown")
        scaffolds[s] = scaffolds.get(s, 0) + 1

    top_domain = max(domains, key=lambda k: domains[k]) if domains else "n/a"
    top_scaffold = max(scaffolds, key=lambda k: scaffolds[k]) if scaffolds else "n/a"
    recent_notes = "; ".join(notes[-3:])

    archive_block = (
        f"<!-- ARCHIVE_START profile:{profile_id} date:{archive_date} count:{count} -->\n"
        f"**Archive** ({date_range}) | {count} sessions | `{profile_id}`\n"
        f"- Top domain: {top_domain} | Top scaffold: {top_scaffold} "
        f"| Ended-with-?: {ended_q}/{count} | Given-away: {given_away}/{count}\n"
        f"- Notes sample: {textwrap.shorten(recent_notes, width=200, placeholder='...')}\n"
        f"<!-- ARCHIVE_END -->\n\n"
    )

    # Remove the raw entries from the file, insert archive block at the top of entries
    new_text = _remove_raw_entries(original_text, to_archive)

    # Insert archive block after the header line (first \n\n boundary)
    split_pos = new_text.find("\n\n") + 2
    new_text = new_text[:split_pos] + archive_block + new_text[split_pos:]

    LEARNINGS_FILE.write_text(new_text, encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_entries(text: str) -> list[dict[str, Any]]:
    """Parse all <!-- ENTRY {...} --> blocks from the file text."""
    entries: list[dict[str, Any]] = []
    for m in re.finditer(r"<!-- ENTRY (\{[^>]*?\}) -->", text):
        try:
            meta = json.loads(m.group(1))
            entries.append(meta)
        except json.JSONDecodeError:
            continue
    return entries


def _remove_raw_entries(text: str, to_remove: list[dict[str, Any]]) -> str:
    """Remove specific ENTRY blocks from file text, identified by their JSON."""
    remove_jsons = {json.dumps(e, separators=(",", ":"), sort_keys=True) for e in to_remove}

    def should_remove(m: re.Match) -> bool:
        try:
            meta = json.loads(m.group(1))
            return json.dumps(meta, separators=(",", ":"), sort_keys=True) in remove_jsons
        except json.JSONDecodeError:
            return False

    # Match each full entry block (comment line + 3 content lines + blank line)
    pattern = re.compile(
        r"<!-- ENTRY \{[^>]*?\} -->\n[^\n]*\n[^\n]*\n[^\n]*\n\n?",
    )

    def replacer(m: re.Match) -> str:
        json_match = re.search(r"<!-- ENTRY (\{[^>]*?\}) -->", m.group(0))
        if json_match:
            try:
                meta = json.loads(json_match.group(1))
                if json.dumps(meta, separators=(",", ":"), sort_keys=True) in remove_jsons:
                    return ""
            except json.JSONDecodeError:
                pass
        return m.group(0)

    return pattern.sub(replacer, text)
