"""
context_engineering/verifier.py

VERIFIER GATE — three automated checks run after every tutor response, before
the response is sent to the student.

Checks:
  (a) answer_giveaway    — response contains a direct numerical answer without scaffolding
  (b) scaffolding_quality — response does not ask any guiding question
  (c) ends_with_question — response does not end with a question mark

If any check fails the caller should regenerate, injecting the failure context.
All failures are logged to eval_log.md.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

EVAL_LOG_FILE = Path(__file__).parent.parent / "eval_log.md"

# ─────────────────────────────────────────────────────────────────────────────
# Regex patterns for answer giveaway detection
# ─────────────────────────────────────────────────────────────────────────────

_ANSWER_PATTERNS = [
    re.compile(r"\bthe answer is\s+[-\d,\.]+", re.IGNORECASE),
    re.compile(r"\bthe solution is\s+[-\d,\.]+", re.IGNORECASE),
    re.compile(r"\bthe result is\s+[-\d,\.]+", re.IGNORECASE),
    re.compile(r"\bequals\s+[-\d,\.]+\b", re.IGNORECASE),
    re.compile(r"\bso\s+[a-zA-Z]\s*=\s*[-\d,\.]+\b", re.IGNORECASE),
    re.compile(r"\btherefore\s+[a-zA-Z]\s*=\s*[-\d,\.]+\b", re.IGNORECASE),
    re.compile(r"\bwhich gives\s+[a-zA-Z]?\s*=?\s*[-\d,\.]+\b", re.IGNORECASE),
    # bare "x = 5" or "y = -3.2" at end of a clause (followed by punctuation or EOL)
    re.compile(r"\b[a-zA-Z]\s*=\s*[-\d,\.]+\s*(?:[,\.!]|$)", re.MULTILINE),
]


class CheckResult(NamedTuple):
    passed: bool
    check_name: str
    reason: str


# ─────────────────────────────────────────────────────────────────────────────
# Individual checks
# ─────────────────────────────────────────────────────────────────────────────

def check_answer_giveaway(response: str) -> CheckResult:
    """
    Fails if the response contains a direct numerical answer pattern.
    A question mark later in the response does NOT excuse a giveaway —
    once the number is stated, the answer is given.
    """
    for pat in _ANSWER_PATTERNS:
        m = pat.search(response)
        if m:
            snippet = m.group(0)[:60]
            return CheckResult(
                passed=False,
                check_name="answer_giveaway",
                reason=f"Direct answer pattern found: «{snippet}»",
            )
    return CheckResult(passed=True, check_name="answer_giveaway", reason="")


def check_scaffolding_quality(response: str) -> CheckResult:
    """
    Fails if the response contains no guiding question at all.
    Even a single '?' satisfies this check — quality of the question
    is evaluated by the human reviewer, not here.
    """
    if "?" not in response:
        return CheckResult(
            passed=False,
            check_name="scaffolding_quality",
            reason="Response contains no guiding question (no '?' found)",
        )
    return CheckResult(passed=True, check_name="scaffolding_quality", reason="")


def check_ends_with_question(response: str) -> CheckResult:
    """
    Fails if the response does not end with a question mark (ignoring trailing whitespace).
    """
    if not response.rstrip().endswith("?"):
        last_40 = response.rstrip()[-40:]
        return CheckResult(
            passed=False,
            check_name="ends_with_question",
            reason=f"Response does not end with '?'. Tail: «...{last_40}»",
        )
    return CheckResult(passed=True, check_name="ends_with_question", reason="")


# ─────────────────────────────────────────────────────────────────────────────
# Gate: run all three checks
# ─────────────────────────────────────────────────────────────────────────────

def run_verifier(response: str) -> list[CheckResult]:
    """Return a list of all failing CheckResults. Empty list = all passed."""
    results = [
        check_answer_giveaway(response),
        check_scaffolding_quality(response),
        check_ends_with_question(response),
    ]
    return [r for r in results if not r.passed]


def build_regeneration_prompt(failures: list[CheckResult]) -> str:
    """
    Build the injected context for regeneration — explains exactly what went wrong
    so the model can correct it.
    """
    lines = [
        "[TUTOR QUALITY CHECK FAILED — REGENERATE]",
        "Your previous response violated the following tutoring rules:",
    ]
    for f in failures:
        lines.append(f"  FAIL {f.check_name}: {f.reason}")
    lines += [
        "",
        "Rules to follow in your regenerated response:",
        "  1. NEVER state the numerical answer directly.",
        "  2. ALWAYS include at least one guiding question (use '?').",
        "  3. Your response MUST end with a question mark.",
        "",
        "Regenerate your response now, correcting all violations above.",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def log_failures(
    profile_id: str,
    domain: str,
    failures: list[CheckResult],
    response_excerpt: str,
) -> None:
    """Append a failure record to eval_log.md."""
    if not failures:
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    fail_names = ", ".join(f.check_name for f in failures)
    excerpt = response_excerpt[:200].replace("\n", " ")

    EVAL_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not EVAL_LOG_FILE.exists():
        EVAL_LOG_FILE.write_text("# Eval Failure Log\n\n", encoding="utf-8")

    meta = json.dumps({
        "profile": profile_id,
        "date": timestamp,
        "domain": domain,
        "failures": [f.check_name for f in failures],
    }, separators=(",", ":"))

    with EVAL_LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(f"<!-- EVAL_ENTRY {meta} -->\n")
        fh.write(
            f"**{timestamp}** | `{profile_id}` | domain: {domain} | FAIL: {fail_names}\n"
        )
        for f in failures:
            fh.write(f"- {f.check_name}: {f.reason}\n")
        fh.write(f"- Response: «{excerpt}»\n\n")
