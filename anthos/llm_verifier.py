"""
llm_verifier.py — VERIFIER GATE for LLM-generated output

Two-check verifier applied to every LLM output before it surfaces:
  (a) ClaimScopeChecker  — blocks autonomy/capability claims beyond what
      the codebase demonstrably implements.
  (b) NistAccuracyChecker — blocks references to NIST control IDs that
      are not present in data/nist_controls.json.

blocked_claims.md is consulted first on every run (fast-reject).
Failed outputs are logged and the failure reason is returned so callers
can inject it into a regeneration prompt (max 3 attempts).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from anthos.compliance_learnings import LearningsLog
from anthos.nist_context import load_nist_controls

logger = logging.getLogger(__name__)

# ── Autonomy / capability claim patterns ──────────────────────────────────────
# These detect assertions that this system does something it does not actually
# do (real-time monitoring, autonomous blocking, live system access, etc.)
_AUTONOMY_PATTERNS: list[str] = [
    # "I actively monitor", "I can autonomously detect", "I will automatically respond", etc.
    r"\bI (?:(?:am|will|can|do) )?(?:actively|currently|automatically|autonomously) "
    r"(?:monitor|protect|detect|respond|enforce|block|prevent|secure)\b",

    # "I am monitoring / protecting / blocking ..."
    r"\bI (?:am|will|can|do) (?:monitor|protect|detect|respond|enforce|block|prevent|secure)(?:ing)?\b",

    # "this system / this AI continuously monitors …"
    r"\bthis (?:system|model|AI|assistant) (?:actively|currently|automatically|autonomously) "
    r"(?:monitors|protects|detects|responds|enforces|blocks|prevents|secures)\b",

    r"\breal-?time (?:threat|anomaly|intrusion|attack) (?:detection|response|prevention|monitoring)\b",

    r"\bautomatically (?:remediates?|quarantines?|isolates?|patches?|updates?)\b",

    r"\bI have (?:access to|connected? to|integrated? with) (?:your|live|production|real)\b",

    r"\bcontinuously (?:scans?|monitors?|analyzes?|checks?|inspects?)\b",

    r"\bwill (?:automatically|autonomously) (?:flag|alert|notify|respond to|stop|block)\b",
]

# NIST SP 800-53 control ID: two uppercase letters, hyphen, 1-2 digits,
# optional parenthesized enhancement (e.g. AC-2, SI-3(1), SA-11(2))
_NIST_ID_RE = re.compile(r"\b([A-Z]{2})-(\d{1,2})(?:\((\d{1,2})\))?\b")


class VerificationResult:
    __slots__ = ("passed", "check_name", "reason")

    def __init__(self, passed: bool, check_name: str, reason: str = ""):
        self.passed     = passed
        self.check_name = check_name
        self.reason     = reason

    def __bool__(self) -> bool:
        return self.passed


class ClaimScopeChecker:
    """Rejects outputs that claim autonomy or live-system capabilities."""

    def __init__(self, blocked_patterns: list[str]):
        self._builtin  = [re.compile(p, re.IGNORECASE) for p in _AUTONOMY_PATTERNS]
        # blocked_claims.md entries — treated as literal substrings (case-insensitive)
        self._blocked  = [
            re.compile(re.escape(b), re.IGNORECASE)
            for b in blocked_patterns
            if b.strip()
        ]

    def check(self, text: str) -> VerificationResult:
        # Fast-reject: blocked_claims.md first
        for pat in self._blocked:
            m = pat.search(text)
            if m:
                return VerificationResult(
                    False, "claim_scope",
                    f"Matches prohibited pattern: {pat.pattern[:100]}",
                )

        # Built-in autonomy detectors
        for pat in self._builtin:
            m = pat.search(text)
            if m:
                return VerificationResult(
                    False, "claim_scope",
                    f"Autonomy/capability claim detected: «{m.group(0)}»",
                )

        return VerificationResult(True, "claim_scope")


class NistAccuracyChecker:
    """Rejects outputs that cite NIST control IDs absent from the mapping."""

    def __init__(self, controls: dict[str, dict]):
        self._known = {k.upper() for k in controls}

    def check(self, text: str) -> VerificationResult:
        matches = _NIST_ID_RE.findall(text)
        if not matches:
            return VerificationResult(True, "nist_accuracy")

        for family, num, enhancement in matches:
            base_id = f"{family}-{num}".upper()
            enh_id  = f"{family}-{num}({enhancement})".upper() if enhancement else None

            if enh_id is not None:
                if enh_id not in self._known and base_id not in self._known:
                    return VerificationResult(
                        False, "nist_accuracy",
                        f"Unknown NIST control referenced: {enh_id}",
                    )
            elif base_id not in self._known:
                return VerificationResult(
                    False, "nist_accuracy",
                    f"Unknown NIST control referenced: {base_id}",
                )

        return VerificationResult(True, "nist_accuracy")


class VerifierGate:
    """
    Orchestrates both checks.  Callers:
      1. Construct once per session with a shared LearningsLog.
      2. Call verify(text, output_type) on every LLM output.
      3. If it returns (False, reason), inject reason into a retry prompt
         and call verify again on the new output (up to MAX_RETRIES times).
    """

    MAX_RETRIES = 3

    def __init__(
        self,
        learnings: LearningsLog,
        output_type: str = "general",
    ):
        controls = load_nist_controls(output_type)
        blocked  = learnings.load_blocked_claims()
        self._claim_checker = ClaimScopeChecker(blocked)
        self._nist_checker  = NistAccuracyChecker(controls)
        self._learnings     = learnings
        self._output_type   = output_type

    def verify(self, text: str) -> tuple[bool, str]:
        """
        Run both checks in order.  Returns (passed, failure_reason).
        Appends a learning entry regardless of outcome.
        """
        claim_result = self._claim_checker.check(text)
        if not claim_result:
            self._learnings.append(
                self._output_type,
                "claim_scope",
                "FAIL",
                claim_result.reason,
            )
            logger.warning("[verifier] claim_scope FAIL — %s", claim_result.reason)
            return False, f"[claim_scope] {claim_result.reason}"

        nist_result = self._nist_checker.check(text)
        if not nist_result:
            self._learnings.append(
                self._output_type,
                "nist_accuracy",
                "FAIL",
                nist_result.reason,
            )
            logger.warning("[verifier] nist_accuracy FAIL — %s", nist_result.reason)
            return False, f"[nist_accuracy] {nist_result.reason}"

        self._learnings.append(
            self._output_type,
            "both",
            "PASS",
            "Output cleared both claim_scope and nist_accuracy checks",
        )
        return True, ""


def make_retry_note(failure_reason: str) -> str:
    """Format a failure reason for injection into the next generation prompt."""
    return (
        f"\n\n[VERIFIER REJECTION — REGENERATE]\n"
        f"Your previous response was blocked for the following reason:\n"
        f"  {failure_reason}\n"
        f"Do not repeat the flagged content. Generate a corrected response "
        f"that avoids the prohibited pattern entirely.\n"
        f"[END REJECTION NOTE]\n"
    )
