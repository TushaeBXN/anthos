"""
nist_context.py — SELECT layer for LLM session context

At session start, loads:
  • only the NIST SP 800-53 control definitions relevant to the current
    task scope (not the full catalog)
  • the last 10 entries from compliance_learnings.md
  • all escalated hard rules from blocked_claims.md

Returns a context string to prepend to the LLM system prompt so the model
is aware of applicable controls and prior failure patterns before generating.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from anthos.compliance_learnings import (
    BLOCKED_FILE,
    LEARNINGS_FILE,
    _parse_entries,
    _ENTRY_RE,
)

_DATA_DIR   = Path(__file__).parent.parent / "data"
_NIST_FILE  = _DATA_DIR / "nist_controls.json"

# Map output type → relevant NIST control family prefixes
_TYPE_FAMILIES: dict[str, list[str]] = {
    "compliance_assessment": ["AC", "CA", "CM", "IA", "RA", "SA", "SI"],
    "incident_report":       ["AU", "IR", "SC", "SI"],
    "threat_description":    ["RA", "SC", "SI", "IR"],
    "general":               ["SI", "SC"],   # minimal default
}

# Keywords in a topic/prompt that hint at the output type
_TYPE_HINTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:compliance|audit|assessment|control|governance|policy)\b", re.I), "compliance_assessment"),
    (re.compile(r"\b(?:incident|breach|intrusion|attack|compromise|response)\b",  re.I), "incident_report"),
    (re.compile(r"\b(?:threat|vulnerability|exploit|malware|ransomware|CVE)\b",   re.I), "threat_description"),
]


def detect_output_type(text: str) -> str:
    """Infer output type from topic text; defaults to 'general'."""
    for pattern, output_type in _TYPE_HINTS:
        if pattern.search(text):
            return output_type
    return "general"


def load_nist_controls(output_type: str = "general") -> dict[str, dict]:
    """Return the subset of NIST controls relevant to output_type."""
    if not _NIST_FILE.exists():
        return {}
    all_controls: dict[str, dict] = json.loads(_NIST_FILE.read_text(encoding="utf-8"))
    families = _TYPE_FAMILIES.get(output_type, _TYPE_FAMILIES["general"])
    return {
        ctrl_id: ctrl
        for ctrl_id, ctrl in all_controls.items()
        if any(ctrl_id.startswith(fam + "-") for fam in families)
    }


def load_recent_learnings(n: int = 10) -> list[tuple[str, str]]:
    """Return the last n (header, body) entry pairs from compliance_learnings.md."""
    if not LEARNINGS_FILE.exists():
        return []
    raw = LEARNINGS_FILE.read_text(encoding="utf-8")
    all_entries = _parse_entries(raw)
    return all_entries[-n:]


def load_hard_rules() -> list[str]:
    """Return all active lines from blocked_claims.md."""
    if not BLOCKED_FILE.exists():
        return []
    lines = []
    for raw in BLOCKED_FILE.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped and not stripped.startswith("#"):
            lines.append(stripped)
    return lines


def build_session_context(output_type: str = "general") -> str:
    """
    Compose the context string injected at LLM session start.
    Includes relevant NIST controls, last 10 learnings, and hard rules.
    """
    controls = load_nist_controls(output_type)
    learnings = load_recent_learnings(10)
    hard_rules = load_hard_rules()

    parts: list[str] = []

    # NIST controls section
    if controls:
        ctrl_lines = [
            f"  {cid}: {c['name']} ({c['family']})"
            for cid, c in sorted(controls.items())
        ]
        parts.append(
            "APPLICABLE NIST SP 800-53 Rev 5 CONTROLS FOR THIS SESSION:\n"
            + "\n".join(ctrl_lines)
        )

    # Recent learnings section
    if learnings:
        learning_lines = []
        for header, body in learnings:
            m = _ENTRY_RE.match(header)
            if m:
                result = m.group(4)
                check  = m.group(3)
                note   = body.lstrip("- ").strip()
                learning_lines.append(f"  [{result}] {check}: {note}")
        if learning_lines:
            parts.append(
                "RECENT COMPLIANCE LEARNINGS (last 10 runs):\n"
                + "\n".join(learning_lines)
            )

    # Hard rules section
    if hard_rules:
        parts.append(
            "PROHIBITED PATTERNS — DO NOT GENERATE OUTPUT MATCHING THESE:\n"
            + "\n".join(f"  • {r}" for r in hard_rules)
        )

    if not parts:
        return ""

    return (
        "\n\n[VERIFIER CONTEXT — READ BEFORE GENERATING]\n"
        + "\n\n".join(parts)
        + "\n[END VERIFIER CONTEXT]\n"
    )
