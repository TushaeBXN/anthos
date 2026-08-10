"""
compliance_learnings.py — WRITE + ESCALATION + COMPRESS layers

WRITE   : append a 2-line entry to compliance_learnings.md after every
          verifier gate run (pass or fail).
ESCALATE: when the same claim type fails 3+ times, write the offending
          pattern to blocked_claims.md so the verifier rejects it first.
COMPRESS: when compliance_learnings.md exceeds 50 entries, summarize
          the oldest 25 into a dated archive block and rewrite the file.
"""

from __future__ import annotations

import re
import threading
from collections import defaultdict
from datetime import date
from pathlib import Path

_DATA_DIR = Path(__file__).parent.parent / "data"
LEARNINGS_FILE = _DATA_DIR / "compliance_learnings.md"
BLOCKED_FILE   = _DATA_DIR / "blocked_claims.md"

_ENTRY_RE = re.compile(
    r"^## (\d{4}-\d{2}-\d{2}) \| type=(\S+) \| check=(\S+) \| result=(PASS|FAIL)$"
)
_COMPRESS_THRESHOLD = 50
_COMPRESS_KEEP      = 25   # oldest N entries to archive when threshold hit


class LearningsLog:
    """Thread-safe append-only learnings ledger with escalation and compression."""

    def __init__(
        self,
        learnings_file: Path = LEARNINGS_FILE,
        blocked_file: Path = BLOCKED_FILE,
    ):
        self._lf = learnings_file
        self._bf = blocked_file
        self._lock = threading.Lock()
        # failure_counts[check_name][pattern_key] = count
        self._failure_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._already_blocked: set[str] = set()
        self._load_existing_blocked()

    # ── public API ────────────────────────────────────────────────────────────

    def append(
        self,
        output_type: str,
        check_name: str,
        result: str,
        note: str,
    ) -> None:
        """Write a 2-line entry and, on failure, trigger escalation + compression."""
        today = date.today().isoformat()
        header = f"## {today} | type={output_type} | check={check_name} | result={result}"
        body   = f"- {note[:200]}"

        with self._lock:
            self._lf.parent.mkdir(parents=True, exist_ok=True)
            with open(self._lf, "a", encoding="utf-8") as fh:
                fh.write(f"{header}\n{body}\n\n")

            if result == "FAIL":
                self._track_failure(check_name, note)

            self._maybe_compress()

    def load_blocked_claims(self) -> list[str]:
        """Return all active blocked-claim patterns from blocked_claims.md."""
        with self._lock:
            return self._read_blocked_lines()

    # ── internals ─────────────────────────────────────────────────────────────

    def _load_existing_blocked(self) -> None:
        for line in self._read_blocked_lines():
            self._already_blocked.add(line)

    def _read_blocked_lines(self) -> list[str]:
        if not self._bf.exists():
            return []
        lines = []
        for raw in self._bf.read_text(encoding="utf-8").splitlines():
            stripped = raw.strip()
            if stripped and not stripped.startswith("#"):
                lines.append(stripped)
        return lines

    def _track_failure(self, check_name: str, note: str) -> None:
        """Count failures; escalate to blocked_claims.md when threshold reached."""
        key = _extract_pattern_key(note)
        self._failure_counts[check_name][key] += 1
        count = self._failure_counts[check_name][key]

        if count >= 3 and key not in self._already_blocked:
            self._escalate(check_name, key, count)

    def _escalate(self, check_name: str, key: str, count: int) -> None:
        today = date.today().isoformat()
        entry = (
            f"# Escalated {today} | check={check_name} | failures={count}\n"
            f"{key}\n"
        )
        self._bf.parent.mkdir(parents=True, exist_ok=True)
        with open(self._bf, "a", encoding="utf-8") as fh:
            fh.write(entry)
        self._already_blocked.add(key)

    def _maybe_compress(self) -> None:
        """Archive the oldest _COMPRESS_KEEP entries when total exceeds threshold."""
        if not self._lf.exists():
            return
        raw = self._lf.read_text(encoding="utf-8")
        entries = _parse_entries(raw)
        if len(entries) <= _COMPRESS_THRESHOLD:
            return

        to_archive  = entries[:_COMPRESS_KEEP]
        to_keep     = entries[_COMPRESS_KEEP:]
        archive_block = _build_archive_block(to_archive)

        header_section = _header_section(raw)
        new_content = header_section + archive_block + _entries_to_text(to_keep)
        self._lf.write_text(new_content, encoding="utf-8")


# ── helpers ───────────────────────────────────────────────────────────────────

def _extract_pattern_key(note: str) -> str:
    """Derive a stable, short key from a failure note for dedup counting."""
    # Strip leading "Matches prohibited pattern: " or "... detected: «...»"
    for prefix in ("Matches prohibited pattern:", "claim detected:", "control referenced:"):
        idx = note.find(prefix)
        if idx != -1:
            return note[idx + len(prefix):].strip()[:120]
    return note[:120].strip()


def _parse_entries(raw: str) -> list[tuple[str, str]]:
    """Return list of (header_line, body_line) pairs from the file."""
    entries: list[tuple[str, str]] = []
    lines = raw.splitlines()
    i = 0
    while i < len(lines):
        if _ENTRY_RE.match(lines[i]):
            header = lines[i]
            body   = lines[i + 1] if (i + 1 < len(lines)) else ""
            entries.append((header, body))
            i += 3   # header + body + blank line
        else:
            i += 1
    return entries


def _entries_to_text(entries: list[tuple[str, str]]) -> str:
    return "".join(f"{h}\n{b}\n\n" for h, b in entries)


def _header_section(raw: str) -> str:
    """
    Return only the non-entry header (comment block) that precedes the first
    entry.  If the file starts directly with an entry line (no comment block),
    return an empty string so no entry is accidentally captured as a header.
    """
    first_line = raw.split("\n")[0] if raw else ""
    # File starts with an actual entry — there is no comment header.
    if _ENTRY_RE.match(first_line):
        return ""

    idx = raw.find("\n## ")
    if idx == -1:
        # No entries at all — the whole file is comment/header text.
        return raw.rstrip() + "\n\n"
    return raw[: idx + 1]


def _build_archive_block(entries: list[tuple[str, str]]) -> str:
    if not entries:
        return ""
    first_date = entries[0][0].split("|")[0].replace("##", "").strip()
    last_date  = entries[-1][0].split("|")[0].replace("##", "").strip()
    counts: dict[str, int] = defaultdict(int)
    pass_count = fail_count = 0
    for header, _ in entries:
        m = _ENTRY_RE.match(header)
        if m:
            counts[m.group(3)] += 1
            if m.group(4) == "PASS":
                pass_count += 1
            else:
                fail_count += 1

    top_checks = ", ".join(
        f"{k}×{v}" for k, v in sorted(counts.items(), key=lambda x: -x[1])
    )
    block = (
        f"## ARCHIVE {first_date}→{last_date} "
        f"| entries={len(entries)} | pass={pass_count} fail={fail_count} "
        f"| checks={top_checks}\n"
        f"- Summarized {len(entries)} oldest entries; "
        f"{fail_count} failures across checks: {top_checks}\n\n"
    )
    return block
