"""
context_engineering/escalation.py

ESCALATION — after each session, scan eval_log.md for failure counts per profile.
If the same failure type appears 5 or more times for a student profile, promote it
as a hard rule into that profile's hard-rules file.

Hard rules generated here are permanent — they persist across all future sessions.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from profiles.student_profiles import add_hard_rule

EVAL_LOG_FILE = Path(__file__).parent.parent / "eval_log.md"
ESCALATION_THRESHOLD = 5

_EVAL_ENTRY_RE = re.compile(
    r"<!-- EVAL_ENTRY (\{.*?\}) -->",
    re.DOTALL,
)

_RULE_TEMPLATES: dict[str, str] = {
    "answer_giveaway": (
        "ESCALATED: Never state numerical answers directly — "
        "this profile's sessions triggered repeated answer giveaway violations."
    ),
    "scaffolding_quality": (
        "ESCALATED: Every response must contain at least one guiding question — "
        "this profile repeatedly received responses with no '?' character."
    ),
    "ends_with_question": (
        "ESCALATED: Always end your response with a question mark — "
        "this profile repeatedly received responses that ended without a question."
    ),
}


def _load_eval_log_entries() -> list[dict]:
    """Parse all EVAL_ENTRY metadata blocks from eval_log.md."""
    if not EVAL_LOG_FILE.exists():
        return []

    text = EVAL_LOG_FILE.read_text(encoding="utf-8")
    entries: list[dict] = []
    for m in _EVAL_ENTRY_RE.finditer(text):
        try:
            entry = json.loads(m.group(1))
            entries.append(entry)
        except json.JSONDecodeError:
            continue
    return entries


def _failure_counts_by_profile() -> dict[str, Counter]:
    """
    Return {profile_id: Counter({failure_type: count})} across all logged entries.
    Each entry may have multiple failure types; count each separately.
    """
    entries = _load_eval_log_entries()
    by_profile: dict[str, Counter] = {}

    for entry in entries:
        profile = entry.get("profile", "")
        if not profile:
            continue
        failures = entry.get("failures", [])
        if profile not in by_profile:
            by_profile[profile] = Counter()
        for f in failures:
            by_profile[profile][f] += 1

    return by_profile


def _already_escalated(profile_id: str, failure_type: str) -> bool:
    """Check whether this failure type has already been promoted for this profile."""
    rule_file = Path(__file__).parent.parent / "profiles" / "hard_rules" / f"{profile_id}.txt"
    if not rule_file.exists():
        return False
    existing = rule_file.read_text(encoding="utf-8")
    marker = f"ESCALATED: {failure_type.split('_')[0].upper()}"
    return marker in existing or _RULE_TEMPLATES.get(failure_type, "") in existing


def check_and_escalate(profile_id: str | None = None) -> list[tuple[str, str]]:
    """
    Scan the eval log and promote any failure type that has reached the threshold.

    Args:
        profile_id: if given, only check that profile; otherwise check all profiles.

    Returns:
        List of (profile_id, failure_type) tuples that were escalated this call.
    """
    counts = _failure_counts_by_profile()
    escalated: list[tuple[str, str]] = []

    profiles_to_check = [profile_id] if profile_id else list(counts.keys())

    for pid in profiles_to_check:
        if pid not in counts:
            continue
        for failure_type, count in counts[pid].items():
            if count >= ESCALATION_THRESHOLD:
                if _already_escalated(pid, failure_type):
                    continue
                rule = _RULE_TEMPLATES.get(
                    failure_type,
                    f"ESCALATED: Repeated failure '{failure_type}' for this profile.",
                )
                add_hard_rule(pid, rule)
                escalated.append((pid, failure_type))
                print(
                    f"[ESCALATION] Profile '{pid}': '{failure_type}' hit {count} failures "
                    f"— promoted to hard rule."
                )

    return escalated


def get_failure_summary(profile_id: str) -> str:
    """Return a human-readable failure summary for a profile (for diagnostics)."""
    counts = _failure_counts_by_profile()
    if profile_id not in counts:
        return f"No failures logged for profile '{profile_id}'."

    lines = [f"Failure counts for '{profile_id}':"]
    for failure_type, count in counts[profile_id].most_common():
        marker = " [ESCALATED]" if _already_escalated(profile_id, failure_type) else ""
        lines.append(f"  {failure_type}: {count}{marker}")
    return "\n".join(lines)
