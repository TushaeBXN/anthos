"""
Context engineering loop infrastructure for the Anthos autonomous agent.

Provides six layers:
  WRITE     — append structured entries to learnings.md after each run
  SELECT    — load 10 most-recent entries + all hard-rule entries
  COMPRESS  — archive oldest 25 entries when file exceeds 50 entries
  VERIFY    — run pytest after any file write; block loop advancement on failure
  ESCALATE  — promote repeated failures (≥3 occurrences) to CLAUDE.md hard rules
  ISOLATE   — build clean, history-free context for subagent subtasks

Intentionally has no torch dependency — importable without the full anthos
package so that tests run in environments where torch is not installed.
"""

from __future__ import annotations

import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Optional

# Project root — this file lives at the repo root, so __file__.parent IS root.
_ROOT = Path(__file__).parent

LEARNINGS_PATH = _ROOT / "learnings.md"
CLAUDE_MD_PATH = _ROOT / "CLAUDE.md"

# Thresholds
MAX_ENTRIES = 50          # trigger compression above this count
ARCHIVE_CHUNK = 25        # number of oldest entries to archive at once
ESCALATION_THRESHOLD = 3  # auto-promote a failure to hard rule after N occurrences

# Entry delimiter and hard-rule tag used inside learnings.md
_ENTRY_SEP = "\n---\n"
_HARD_TAG = "[HARD]"


class LearningsManager:
    """
    Manages the learnings.md context engineering loop.

    All public methods are safe to call even if learnings.md or CLAUDE.md do
    not yet exist — they are created on first use.
    """

    def __init__(
        self,
        learnings_path: Path = LEARNINGS_PATH,
        claude_md_path: Path = CLAUDE_MD_PATH,
    ) -> None:
        self.learnings_path = Path(learnings_path)
        self.claude_md_path = Path(claude_md_path)
        self._ensure_files()

    # ── WRITE LAYER ──────────────────────────────────────────────────────────

    def write_entry(
        self,
        task: str,
        succeeded: str,
        failed: str,
        rule: str,
        hard: bool = False,
    ) -> None:
        """
        Append one structured 3-line entry to learnings.md.

        Format (exactly 3 content lines, preceded by the --- separator):
            YYYY-MM-DD | <task>
            OK: <succeeded> | FAIL: <failed>
            RULE: <rule> [HARD]   ← [HARD] appended only when hard=True
        """
        today = date.today().isoformat()
        hard_tag = f" {_HARD_TAG}" if hard else ""
        entry = (
            f"{today} | {_truncate(task, 120)}\n"
            f"OK: {_truncate(succeeded, 100)} | FAIL: {_truncate(failed, 100)}\n"
            f"RULE: {_truncate(rule, 160)}{hard_tag}"
        )
        with self.learnings_path.open("a") as fh:
            fh.write(_ENTRY_SEP + entry + "\n")

        self._maybe_compress()
        self._check_escalation(failed, rule)

    # ── SELECT LAYER ─────────────────────────────────────────────────────────

    def select_context(self) -> str:
        """
        Return a context string containing:
          • all entries flagged [HARD]   (never pruned)
          • the 10 most-recent non-hard entries

        Inject this string after any skill/CLAUDE.md content and before the
        task specification when building an agent's system context.
        """
        entries = self._load_entries()
        hard_entries = [e for e in entries if _HARD_TAG in e]
        recent_entries = [e for e in entries if _HARD_TAG not in e][-10:]

        selected = hard_entries + recent_entries
        if not selected:
            return ""

        blocks = "\n---\n".join(selected)
        return f"## Injected Learnings\n\n{blocks}\n"

    # ── COMPRESS LAYER ───────────────────────────────────────────────────────

    def _maybe_compress(self) -> None:
        """Archive the oldest ARCHIVE_CHUNK entries when total exceeds MAX_ENTRIES."""
        entries = self._load_entries()
        non_hard = [e for e in entries if _HARD_TAG not in e]
        if len(non_hard) <= MAX_ENTRIES:
            return

        hard = [e for e in entries if _HARD_TAG in e]
        to_archive = non_hard[:ARCHIVE_CHUNK]
        to_keep = non_hard[ARCHIVE_CHUNK:]

        archive_block = self._make_archive_block(to_archive)
        # Rebuild the file: header, archive block, hard rules, then remaining entries
        header = self.learnings_path.read_text().split(_ENTRY_SEP)[0]
        parts = [header.rstrip(), archive_block] + hard + to_keep
        self.learnings_path.write_text(_ENTRY_SEP.join(parts) + "\n")

    def _make_archive_block(self, entries: list[str]) -> str:
        tasks, rules = [], []
        for e in entries:
            lines = e.strip().splitlines()
            if lines:
                # Line 0: "YYYY-MM-DD | <task>"
                tasks.append(lines[0].split("|", 1)[-1].strip() if "|" in lines[0] else lines[0])
            if len(lines) >= 3:
                # Line 2: "RULE: <rule>"
                rules.append(lines[2].removeprefix("RULE: ").strip())

        task_list = "\n".join(f"- {t}" for t in tasks)
        rule_list = "\n".join(f"- {r}" for r in rules)
        today = date.today().isoformat()
        return (
            f"## ARCHIVE — {today} ({len(entries)} entries compressed)\n"
            f"### Tasks\n{task_list}\n"
            f"### Rules Captured\n{rule_list}"
        )

    # ── VERIFIER GATE ────────────────────────────────────────────────────────

    def run_tests(self) -> tuple[bool, str]:
        """
        Run the project test suite (pytest tests/).

        Returns (passed: bool, output: str).
        The loop MUST NOT advance to the next iteration if this returns False.
        """
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short"],
            capture_output=True,
            text=True,
            cwd=str(_ROOT),
        )
        passed = result.returncode == 0
        output = (result.stdout + result.stderr).strip()
        return passed, output

    # ── ESCALATION ───────────────────────────────────────────────────────────

    def _check_escalation(self, failed: str, rule: str) -> None:
        """Promote rule to CLAUDE.md hard rules if the same failure has appeared ≥3 times."""
        if not failed or failed.lower() in ("none", "n/a", ""):
            return
        text = self.learnings_path.read_text()
        # Count occurrences of the first 60 chars of the failure string
        fingerprint = re.escape(_truncate(failed, 60))
        occurrences = len(re.findall(fingerprint, text))
        if occurrences >= ESCALATION_THRESHOLD:
            self._promote_to_claude_md(rule, failed)

    def _promote_to_claude_md(self, rule: str, trigger: str) -> None:
        """Insert a hard rule into the ## Hard Rules section of CLAUDE.md."""
        text = self.claude_md_path.read_text()
        # Idempotent: skip if already present
        if rule[:60] in text:
            return
        bullet = (
            f"- {rule}"
            f"  _(auto-promoted — repeated failure: {_truncate(trigger, 80)})_\n"
        )
        marker = "## Hard Rules"
        if marker in text:
            text = text.replace(marker + "\n", marker + "\n" + bullet, 1)
        else:
            text += f"\n{marker}\n{bullet}"
        self.claude_md_path.write_text(text)

    # ── ISOLATION ────────────────────────────────────────────────────────────

    def build_subagent_context(
        self,
        task_spec: str,
        relevant_files: Optional[list[str]] = None,
    ) -> str:
        """
        Build an isolated context for a subagent subtask.

        Includes only:
          • the task specification
          • the list of relevant files
          • hard rules from CLAUDE.md

        No conversation history is passed — each subagent starts clean.
        """
        hard_rules = self._load_hard_rules_from_claude_md()
        files_section = ""
        if relevant_files:
            files_section = (
                "## Relevant Files\n"
                + "\n".join(f"- {f}" for f in relevant_files)
                + "\n\n"
            )
        return (
            f"## Task\n{task_spec}\n\n"
            f"{files_section}"
            f"## Hard Rules\n{hard_rules}\n"
        )

    def _load_hard_rules_from_claude_md(self) -> str:
        text = self.claude_md_path.read_text()
        marker = "## Hard Rules"
        if marker not in text:
            return "(none)"
        section = text[text.index(marker) + len(marker):].strip()
        # Stop at the next ## heading
        next_heading = re.search(r"\n## ", section)
        if next_heading:
            section = section[: next_heading.start()]
        return section.strip() or "(none)"

    # ── INTERNALS ────────────────────────────────────────────────────────────

    def _ensure_files(self) -> None:
        if not self.learnings_path.exists():
            self.learnings_path.write_text(
                "# Learnings\n\n"
                "_Auto-maintained by LearningsManager. Do not hand-edit separators._\n"
            )
        if not self.claude_md_path.exists():
            self.claude_md_path.write_text(
                "# CLAUDE.md\n\n"
                "Project: Anthos — Thought-Token Bifurcated Recurrent Transformer\n\n"
                "## Hard Rules\n\n"
                "_Auto-populated by LearningsManager when failures repeat ≥3 times._\n"
            )

    def _load_entries(self) -> list[str]:
        """Return all individual entry strings from learnings.md."""
        text = self.learnings_path.read_text()
        parts = text.split(_ENTRY_SEP)
        entries = []
        for part in parts:
            stripped = part.strip()
            # An entry starts with a date line: YYYY-MM-DD |
            if re.match(r"^\d{4}-\d{2}-\d{2}\s*\|", stripped):
                entries.append(stripped)
        return entries


def _truncate(s: str, n: int) -> str:
    s = s.strip()
    return s[:n] + "…" if len(s) > n else s
