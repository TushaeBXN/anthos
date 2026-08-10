"""
eval/eval_context_loop.py — Context engineering loop for Anthos evaluation.

Seven layers, each handling a distinct responsibility:

  WRITE   — append structured entry to eval_learnings.md after each eval run
  SELECT  — load phase/modality-scoped context at the start of each new run
  VERIFY  — gate checks before marking a result as a valid checkpoint
  FEEDBACK — parse failures → dataset targeting flags for next data batch
  ESCALATE — promote repeated failures to training_rules.md hard rules
  COMPRESS — archive oldest entries when eval_learnings.md exceeds 60 entries
  COLIBRI  — track MoE expert activation rates for 34B+ disk-streamed variants
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

# ─── File paths ──────────────────────────────────────────────────────────────

EVAL_ROOT       = Path(__file__).parent
LEARNINGS_PATH  = EVAL_ROOT / "eval_learnings.md"
RULES_PATH      = EVAL_ROOT / "training_rules.md"
FLAGS_PATH      = EVAL_ROOT / "dataset_targeting_flags.json"

# ─── Constants ───────────────────────────────────────────────────────────────

ALL_MODALITIES            = ["vision", "audio", "text"]
ENTRY_SEP                 = "---"
ARCHIVE_BLOCK_LIMIT       = 60   # archive trigger
ARCHIVE_BATCH             = 30   # entries to archive per compression
CONTEXT_HISTORY_LIMIT     = 10   # last N entries loaded per SELECT
REGRESSION_THRESHOLD      = 0.05  # 5% metric drop → regression failure
ESCALATION_THRESHOLD      = 3    # failure appearances → hard rule
ROUTING_COLLAPSE_MAX      = 0.95  # any single routing bucket above this = collapsed
COLIBRI_ACTIVATION_WARN   = 0.40  # >40% experts loaded/pass = routing efficiency warning
COLIBRI_PARAM_THRESHOLD_B = 34   # variants ≥34B use disk-streaming (Colibri)

# Mapping from training tier name → canonical eval phase
TIER_TO_PHASE: dict[str, str] = {
    "smoke":              "Pretraining",
    "proof":              "Pretraining",
    "research":           "Pretraining",
    "ethnic":             "Pretraining",
    "distill":            "Pretraining",
    "history":            "Alignment",
    "identity_hardening": "Alignment",
    "instruct":           "Instruction",
    "sft":                "Instruction",
    "convo_smoke":        "Instruction",
    # Four-phase train_anthos.py phases
    "foundation":         "Pretraining",
    "identity":           "Alignment",
    "instruction":        "Instruction",
}


# ─────────────────────────────────────────────────────────────────────────────
# EvalEntry — one structured log record
# ─────────────────────────────────────────────────────────────────────────────

class EvalEntry:
    """Represents one structured evaluation result entry (max 4 lines on disk)."""

    __slots__ = (
        "date", "phase", "variant", "modality", "metrics",
        "failures", "missing", "is_checkpoint", "block_reason",
        "expert_activation_rate", "routing_efficiency_warning",
    )

    def __init__(
        self,
        date: str,
        phase: str,
        variant: str,
        modality: str,
        metrics: dict[str, float],
        failures: list[str],
        missing: str,
        is_checkpoint: bool = False,
        block_reason: Optional[str] = None,
        expert_activation_rate: Optional[float] = None,
        routing_efficiency_warning: bool = False,
    ):
        self.date                    = date
        self.phase                   = phase
        self.variant                 = variant
        self.modality                = modality
        self.metrics                 = metrics
        self.failures                = failures
        self.missing                 = missing
        self.is_checkpoint           = is_checkpoint
        self.block_reason            = block_reason
        self.expert_activation_rate  = expert_activation_rate
        self.routing_efficiency_warning = routing_efficiency_warning

    # ── Serialisation ────────────────────────────────────────────────────────

    def to_lines(self) -> list[str]:
        """Produce the on-disk lines (max 4, +1 optional Colibri line)."""
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in sorted(self.metrics.items()))
        fail_str    = ", ".join(self.failures) if self.failures else "none"
        ckpt_tag    = " [CHECKPOINT]"                      if self.is_checkpoint else ""
        blk_tag     = f" [BLOCKED: {self.block_reason}]"  if self.block_reason  else ""

        lines = [
            (
                f"date: {self.date}"
                f" | phase: {self.phase}"
                f" | variant: {self.variant}"
                f" | modality: {self.modality}"
                f"{ckpt_tag}{blk_tag}"
            ),
            f"metrics: {metrics_str}",
            f"failures: {fail_str}",
            f"missing: {self.missing}",
        ]

        if self.expert_activation_rate is not None:
            warn = " [ROUTING_EFFICIENCY_WARNING]" if self.routing_efficiency_warning else ""
            lines.append(f"expert_activation_rate: {self.expert_activation_rate:.3f}{warn}")

        return lines

    # ── Deserialisation ──────────────────────────────────────────────────────

    @classmethod
    def from_block(cls, block: str) -> Optional["EvalEntry"]:
        """Parse a raw entry block. Returns None on any parse error."""
        lines = [l.strip() for l in block.strip().splitlines() if l.strip()]
        if len(lines) < 4:
            return None
        try:
            header = lines[0]
            is_checkpoint = "[CHECKPOINT]" in header
            blk_match     = re.search(r"\[BLOCKED: ([^\]]+)\]", header)
            block_reason  = blk_match.group(1) if blk_match else None
            clean_header  = re.sub(r"\[.*?\]", "", header)

            def _field(name: str) -> str:
                m = re.search(rf"{name}:\s*([^|]+)", clean_header)
                return m.group(1).strip() if m else ""

            date     = _field("date")
            phase    = _field("phase")
            variant  = _field("variant")
            modality = _field("modality")

            metrics: dict[str, float] = {}
            if lines[1].startswith("metrics:"):
                for pair in lines[1][len("metrics:"):].strip().split(","):
                    pair = pair.strip()
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        try:
                            metrics[k.strip()] = float(v.strip())
                        except ValueError:
                            pass

            fail_raw = lines[2][len("failures:"):].strip() if lines[2].startswith("failures:") else ""
            failures = [f.strip() for f in fail_raw.split(",") if f.strip() and f.strip() != "none"]

            missing = lines[3][len("missing:"):].strip() if lines[3].startswith("missing:") else ""

            expert_activation_rate   = None
            routing_efficiency_warning = False
            if len(lines) >= 5 and lines[4].startswith("expert_activation_rate:"):
                rate_raw = lines[4][len("expert_activation_rate:"):].strip()
                routing_efficiency_warning = "[ROUTING_EFFICIENCY_WARNING]" in rate_raw
                rate_clean = re.sub(r"\[.*?\]", "", rate_raw).strip()
                try:
                    expert_activation_rate = float(rate_clean)
                except ValueError:
                    pass

            return cls(
                date=date, phase=phase, variant=variant, modality=modality,
                metrics=metrics, failures=failures, missing=missing,
                is_checkpoint=is_checkpoint, block_reason=block_reason,
                expert_activation_rate=expert_activation_rate,
                routing_efficiency_warning=routing_efficiency_warning,
            )
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# EvalContextLoop — orchestrates all seven layers
# ─────────────────────────────────────────────────────────────────────────────

class EvalContextLoop:
    """
    Manages the full eval context engineering loop.

    Quick start::

        loop = EvalContextLoop()

        # Before eval: load scoped context
        ctx = loop.select_context(phase="Alignment", modality="text")

        # After eval: verify + write in one call
        result = loop.verify_and_write(
            phase="Alignment", variant="anthos_1b", modality="text",
            metrics={"gsm8k": 0.72, "mmlu": 0.58},
            failures=["low_thought_diversity"],
            missing="more multi-dialect instruction pairs",
            routing_distribution={"hard": 0.55, "easy": 0.45},
        )

        # Get dataset targeting flags for next data batch
        flags = loop.get_dataset_targeting_flags(phase="Alignment", modality="text")
    """

    def __init__(
        self,
        learnings_path: Path = LEARNINGS_PATH,
        rules_path:     Path = RULES_PATH,
        flags_path:     Path = FLAGS_PATH,
    ):
        self.learnings_path = Path(learnings_path)
        self.rules_path     = Path(rules_path)
        self.flags_path     = Path(flags_path)
        self._bootstrap()

    # ── Bootstrap ────────────────────────────────────────────────────────────

    def _bootstrap(self) -> None:
        """Create missing files with appropriate headers."""
        self.learnings_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.learnings_path.exists():
            self.learnings_path.write_text(_LEARNINGS_HEADER, encoding="utf-8")

        if not self.rules_path.exists():
            self.rules_path.write_text(_RULES_HEADER, encoding="utf-8")

    # ── Low-level I/O ────────────────────────────────────────────────────────

    def _raw_entry_blocks(self) -> list[str]:
        """Return text blocks for each non-archived raw entry."""
        text = self.learnings_path.read_text(encoding="utf-8")
        blocks = []
        for segment in text.split(f"\n{ENTRY_SEP}\n"):
            seg = segment.strip()
            # Skip file header, archive sections, and empty segments
            if not seg or seg.startswith("#") or "<!-- ARCHIVE" in seg:
                continue
            if "date:" in seg:
                blocks.append(seg)
        return blocks

    def _all_entries(self) -> list[EvalEntry]:
        return [e for b in self._raw_entry_blocks() if (e := EvalEntry.from_block(b))]

    def _append_raw_block(self, lines: list[str]) -> None:
        block = "\n".join(lines)
        with open(self.learnings_path, "a", encoding="utf-8") as f:
            f.write(f"\n{ENTRY_SEP}\n{block}\n")

    def _hard_rules(self) -> list[str]:
        if not self.rules_path.exists():
            return []
        rules = []
        for line in self.rules_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("- ") or line.startswith("* "):
                rules.append(line[2:].strip())
        return rules

    # ═════════════════════════════════════════════════════════════════════════
    # LAYER 2 — SELECT
    # ═════════════════════════════════════════════════════════════════════════

    def select_context(self, phase: str, modality: str) -> dict:
        """
        Load eval context for the given phase/modality pair.

        Returns the last CONTEXT_HISTORY_LIMIT entries scoped to exactly
        (phase, modality) plus all hard rules from training_rules.md.
        Cross-phase history is excluded entirely.
        """
        all_entries = self._all_entries()
        scoped = [e for e in all_entries if e.phase == phase and e.modality == modality]
        recent = scoped[-CONTEXT_HISTORY_LIMIT:]

        return {
            "phase":          phase,
            "modality":       modality,
            "recent_entries": recent,
            "hard_rules":     self._hard_rules(),
            "total_for_combo": len(scoped),
        }

    # ═════════════════════════════════════════════════════════════════════════
    # LAYER 3 — VERIFIER GATE
    # ═════════════════════════════════════════════════════════════════════════

    def _check_modality_coverage(
        self, modality: str, modalities_tested: Optional[list[str]]
    ) -> tuple[bool, str]:
        """
        (a) Modality coverage check.

        A multi-modal eval must exercise all three modalities.
        A declared single-modality eval passes automatically.
        """
        if modality != "multi":
            return True, ""

        if modalities_tested is None:
            return (
                False,
                "modality_coverage: multi-modal eval declared but no modalities_tested"
                " list was provided — cannot verify coverage",
            )

        missing = [m for m in ALL_MODALITIES if m not in modalities_tested]
        if missing:
            return (
                False,
                f"modality_coverage: multi-modal eval missing {missing}"
                f" — only {modalities_tested} were exercised",
            )
        return True, ""

    def _check_community_routing(
        self, routing_distribution: Optional[dict[str, float]]
    ) -> tuple[bool, str]:
        """
        (b) Community routing check.

        Verifies the routing distribution across token difficulty buckets is not
        collapsed (one bucket receiving ≥95% of all traffic).
        """
        if not routing_distribution:
            return True, ""

        values = list(routing_distribution.values())
        if not values:
            return True, ""

        max_share = max(values)
        if max_share >= ROUTING_COLLAPSE_MAX:
            dominant = max(routing_distribution, key=routing_distribution.get)
            return (
                False,
                f"community_routing: routing collapsed — '{dominant}' bucket"
                f" received {max_share:.1%} of traffic;"
                " Community Gate may be degenerate",
            )
        return True, ""

    def _check_regression(
        self, phase: str, modality: str, metrics: dict[str, float]
    ) -> tuple[bool, str]:
        """
        (c) Regression check.

        Fails if any metric dropped >5% relative to the most recent valid
        entry for the same (phase, modality) combination.
        """
        all_entries = self._all_entries()
        # Only compare against checkpoint entries that had real metrics
        prior = [
            e for e in all_entries
            if e.phase == phase and e.modality == modality and e.metrics
        ]
        if not prior:
            return True, ""

        prev = prior[-1]
        drops = []
        for key, val in metrics.items():
            if key in prev.metrics and prev.metrics[key] > 0:
                rel_drop = (prev.metrics[key] - val) / prev.metrics[key]
                if rel_drop > REGRESSION_THRESHOLD:
                    drops.append(
                        f"{key}: {prev.metrics[key]:.4f} → {val:.4f}"
                        f" ({rel_drop * 100:.1f}% drop)"
                    )

        if drops:
            return False, f"regression: {'; '.join(drops)}"
        return True, ""

    def run_verifier_gate(
        self,
        phase: str,
        modality: str,
        metrics: dict[str, float],
        routing_distribution: Optional[dict[str, float]] = None,
        modalities_tested: Optional[list[str]] = None,
    ) -> tuple[bool, list[str]]:
        """
        Run all three verifier checks and return (all_passed, failure_messages).
        A non-empty failure list means the result must not be marked as a checkpoint.
        """
        failures: list[str] = []

        ok, msg = self._check_modality_coverage(modality, modalities_tested)
        if not ok:
            failures.append(msg)

        ok, msg = self._check_community_routing(routing_distribution)
        if not ok:
            failures.append(msg)

        ok, msg = self._check_regression(phase, modality, metrics)
        if not ok:
            failures.append(msg)

        return len(failures) == 0, failures

    # ═════════════════════════════════════════════════════════════════════════
    # LAYER 1 — WRITE
    # ═════════════════════════════════════════════════════════════════════════

    def _write_entry(
        self,
        phase: str,
        variant: str,
        modality: str,
        metrics: dict[str, float],
        failures: list[str],
        missing: str,
        is_checkpoint: bool,
        block_reason: Optional[str],
        expert_activation_rate: Optional[float],
    ) -> EvalEntry:
        routing_warn = (
            expert_activation_rate is not None
            and expert_activation_rate > COLIBRI_ACTIVATION_WARN
        )
        entry = EvalEntry(
            date=datetime.utcnow().strftime("%Y-%m-%d"),
            phase=phase,
            variant=variant,
            modality=modality,
            metrics=metrics,
            failures=failures,
            missing=missing,
            is_checkpoint=is_checkpoint,
            block_reason=block_reason,
            expert_activation_rate=expert_activation_rate,
            routing_efficiency_warning=routing_warn,
        )
        self._append_raw_block(entry.to_lines())
        return entry

    # ═════════════════════════════════════════════════════════════════════════
    # COMBINED verify_and_write — main public entry point
    # ═════════════════════════════════════════════════════════════════════════

    def verify_and_write(
        self,
        phase: str,
        variant: str,
        modality: str,
        metrics: dict[str, float],
        failures: list[str],
        missing: str,
        routing_distribution: Optional[dict[str, float]] = None,
        modalities_tested: Optional[list[str]] = None,
        expert_activation_rate: Optional[float] = None,
    ) -> dict:
        """
        Full eval loop step: verify → write → check hard rules → escalate → compress.

        Parameters
        ----------
        phase                 : "Alignment" | "Pretraining" | "Instruction"
        variant               : model variant name, e.g. "anthos_1b"
        modality              : "text" | "vision" | "audio" | "multi"
        metrics               : dict of metric_name → float score
        failures              : list of failure pattern strings observed
        missing               : single note on what the training data is missing
        routing_distribution  : optional dict of bucket → fraction for routing check
        modalities_tested     : required when modality="multi" for coverage check
        expert_activation_rate: optional float (0–1) for Colibri 34B+ variants

        Returns
        -------
        dict with keys:
          gate_passed       — bool
          gate_failures     — list[str]
          is_checkpoint     — bool (same as gate_passed)
          triggered_rules   — list[str] hard rules fired
          escalated_rules   — list[str] newly promoted to training_rules.md
          compressed        — bool whether compression ran
        """
        # ── VERIFIER GATE ─────────────────────────────────────────────────
        gate_passed, gate_failures = self.run_verifier_gate(
            phase=phase,
            modality=modality,
            metrics=metrics,
            routing_distribution=routing_distribution,
            modalities_tested=modalities_tested,
        )

        combined_failures = list(failures) + gate_failures
        block_reason = "; ".join(gate_failures) if gate_failures else None

        # ── CHECK AGAINST EXISTING HARD RULES ────────────────────────────
        triggered_rules = self._check_hard_rules(combined_failures)

        # ── WRITE ─────────────────────────────────────────────────────────
        entry = self._write_entry(
            phase=phase,
            variant=variant,
            modality=modality,
            metrics=metrics,
            failures=combined_failures,
            missing=missing,
            is_checkpoint=gate_passed,
            block_reason=block_reason,
            expert_activation_rate=expert_activation_rate,
        )

        # ── ESCALATION ────────────────────────────────────────────────────
        escalated = self._run_escalation(phase, modality)

        # ── COMPRESS ──────────────────────────────────────────────────────
        compressed = self._run_compression()

        # ── FEEDBACK FLAGS ────────────────────────────────────────────────
        self._write_targeting_flags(phase, modality)

        return {
            "gate_passed":     gate_passed,
            "gate_failures":   gate_failures,
            "is_checkpoint":   gate_passed,
            "triggered_rules": triggered_rules,
            "escalated_rules": escalated,
            "compressed":      compressed,
            "entry":           entry,
        }

    # ═════════════════════════════════════════════════════════════════════════
    # LAYER 5 — ESCALATION
    # ═════════════════════════════════════════════════════════════════════════

    def _run_escalation(self, phase: str, modality: str) -> list[str]:
        """
        Count failure pattern occurrences for this (phase, modality) pair.
        Promote any pattern that has appeared ESCALATION_THRESHOLD or more
        times as a new hard rule in training_rules.md.
        """
        scoped = [
            e for e in self._all_entries()
            if e.phase == phase and e.modality == modality
        ]

        pattern_counts: dict[str, int] = {}
        for e in scoped:
            for f in e.failures:
                # Normalise: strip gate-prefix labels, cap length
                clean = re.sub(
                    r"^(modality_coverage|community_routing|regression):\s*", "", f
                )[:100]
                pattern_counts[clean] = pattern_counts.get(clean, 0) + 1

        existing = set(self._hard_rules())
        new_rules: list[str] = []

        for pattern, count in pattern_counts.items():
            if count >= ESCALATION_THRESHOLD:
                rule = f"[{phase}/{modality}] {pattern} (seen {count}×)"
                if rule not in existing:
                    new_rules.append(rule)

        if new_rules:
            with open(self.rules_path, "a", encoding="utf-8") as f:
                f.write(f"\n## Escalated {datetime.utcnow().strftime('%Y-%m-%d')}\n")
                for rule in new_rules:
                    f.write(f"- {rule}\n")

        return new_rules

    def _check_hard_rules(self, failures: list[str]) -> list[str]:
        """Return hard rules whose keywords overlap with the current failure list."""
        rules = self._hard_rules()
        triggered = []
        for rule in rules:
            rule_words = set(w for w in rule.split() if len(w) > 4)
            for f in failures:
                f_words = set(f.split())
                if rule_words & f_words:
                    triggered.append(rule)
                    break
        return list(dict.fromkeys(triggered))  # deduplicate, preserve order

    # ═════════════════════════════════════════════════════════════════════════
    # LAYER 6 — COMPRESS
    # ═════════════════════════════════════════════════════════════════════════

    def _run_compression(self) -> bool:
        """
        Archive the oldest ARCHIVE_BATCH raw entries when the total exceeds
        ARCHIVE_BLOCK_LIMIT. Metric scores are preserved verbatim in the archive.
        """
        raw_blocks = self._raw_entry_blocks()
        if len(raw_blocks) <= ARCHIVE_BLOCK_LIMIT:
            return False

        to_archive   = raw_blocks[:ARCHIVE_BATCH]
        to_keep      = raw_blocks[ARCHIVE_BATCH:]
        archived     = [e for b in to_archive if (e := EvalEntry.from_block(b))]
        archive_date = datetime.utcnow().strftime("%Y-%m-%d")
        archive_block = _build_archive_block(archive_date, archived)

        # Rebuild file: header + archive block(s) already in the file +
        # the new archive block + remaining live entries
        existing_text = self.learnings_path.read_text(encoding="utf-8")

        # Locate where existing archive blocks end (after <!-- END ARCHIVE -->)
        header_and_archives = _LEARNINGS_HEADER
        archive_re = re.compile(
            r"<!-- ARCHIVE:.*?<!-- END ARCHIVE -->", re.DOTALL
        )
        for match in archive_re.finditer(existing_text):
            header_and_archives += match.group(0) + "\n"

        kept_text = "".join(
            f"\n{ENTRY_SEP}\n{block}\n" for block in to_keep
        )
        self.learnings_path.write_text(
            header_and_archives + archive_block + kept_text,
            encoding="utf-8",
        )
        return True

    # ═════════════════════════════════════════════════════════════════════════
    # LAYER 4 — FEEDBACK LOOP
    # ═════════════════════════════════════════════════════════════════════════

    def get_dataset_targeting_flags(self, phase: str, modality: str) -> dict:
        """
        Parse recent eval_learnings entries for (phase, modality) and return
        targeting flags that dataset generation scripts use to oversample
        weak modalities and failure-prone domains.
        """
        ctx    = self.select_context(phase=phase, modality=modality)
        recent = ctx["recent_entries"]

        pattern_counts: dict[str, int] = {}
        weak_modalities: set[str] = set()
        low_metric_domains: set[str] = set()

        for e in recent:
            # Failure pattern frequency
            for f in e.failures:
                clean = re.sub(
                    r"^(modality_coverage|community_routing|regression):\s*", "", f
                ).strip()
                if clean and clean != "none":
                    pattern_counts[clean] = pattern_counts.get(clean, 0) + 1

            # Weak modalities: single-modality entries with poor avg score
            if e.modality in ALL_MODALITIES and e.metrics:
                avg = sum(e.metrics.values()) / len(e.metrics)
                if avg < 0.50:
                    weak_modalities.add(e.modality)

            # Low-metric domains heuristic: look for domain keywords in failures
            for f in e.failures:
                for domain in ("cybersecurity", "coding", "math", "identity", "vision", "audio"):
                    if domain in f.lower():
                        low_metric_domains.add(domain)

        top_failures = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        flags = {
            "phase":                phase,
            "modality":             modality,
            "top_failure_patterns": [p for p, _ in top_failures],
            "failure_weights":      {p: c for p, c in top_failures},
            "weak_modalities":      sorted(weak_modalities),
            "oversample_modalities": sorted(weak_modalities),
            "low_metric_domains":   sorted(low_metric_domains),
            "generated_at":         datetime.utcnow().isoformat(),
        }
        return flags

    def _write_targeting_flags(self, phase: str, modality: str) -> None:
        """Persist current targeting flags to FLAGS_PATH for dataset scripts."""
        flags = self.get_dataset_targeting_flags(phase=phase, modality=modality)
        self.flags_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.flags_path, "w", encoding="utf-8") as f:
            json.dump(flags, f, indent=2)

    # ═════════════════════════════════════════════════════════════════════════
    # LAYER 7 — COLIBRI helpers (34B+ disk-streaming)
    # ═════════════════════════════════════════════════════════════════════════

    @staticmethod
    def is_colibri_variant(variant: str) -> bool:
        """
        Return True when the variant is a 34B+ model that uses disk-streaming
        (Colibri) inference.  Matches anthos_50b, anthos_100b, or any name
        that encodes a parameter count ≥ COLIBRI_PARAM_THRESHOLD_B.
        """
        name = variant.lower()
        # Explicit large variants
        if any(tag in name for tag in ("50b", "100b", "34b", "70b")):
            return True
        # Extract numeric suffix like "anthos_45b"
        m = re.search(r"(\d+)b", name)
        if m:
            try:
                return int(m.group(1)) >= COLIBRI_PARAM_THRESHOLD_B
            except ValueError:
                pass
        return False

    @staticmethod
    def colibri_note(
        expert_activation_rate: float,
    ) -> tuple[float, bool]:
        """
        Compute the Colibri routing efficiency note fields.

        Returns
        -------
        (expert_activation_rate, routing_efficiency_warning)
        """
        warning = expert_activation_rate > COLIBRI_ACTIVATION_WARN
        return expert_activation_rate, warning


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_archive_block(archive_date: str, entries: list[EvalEntry]) -> str:
    """
    Build a compressed archive section grouped by training phase.
    Metric scores are preserved verbatim — they must not be summarised to prose.
    """
    by_phase: dict[str, list[EvalEntry]] = {}
    for e in entries:
        by_phase.setdefault(e.phase, []).append(e)

    lines = [f"<!-- ARCHIVE: {archive_date} — {len(entries)} entries -->"]
    for phase in sorted(by_phase):
        lines.append(f"### Archive — {phase} ({archive_date})")
        for e in by_phase[phase]:
            m_str  = ", ".join(f"{k}={v:.4f}" for k, v in sorted(e.metrics.items()))
            f_str  = ", ".join(e.failures) if e.failures else "none"
            c_tag  = " [CHECKPOINT]" if e.is_checkpoint else ""
            expert = (
                f" | expert_rate={e.expert_activation_rate:.3f}"
                + (" [ROUTING_EFFICIENCY_WARNING]" if e.routing_efficiency_warning else "")
                if e.expert_activation_rate is not None else ""
            )
            lines.append(
                f"  {e.date} | {e.variant} | {e.modality}{c_tag}"
                f" | metrics: {m_str}"
                f" | failures: {f_str}"
                f"{expert}"
            )
    lines.append("<!-- END ARCHIVE -->\n")
    return "\n".join(lines) + "\n"


_LEARNINGS_HEADER = """\
# Eval Learnings
Structured evaluation results for the Anthos context engineering loop.
Each entry is at most 4 lines (+ optional Colibri line), separated by `---`.
Do not edit entries manually — use EvalContextLoop.

"""

_RULES_HEADER = """\
# Training Rules
Hard rules escalated from eval_learnings.md by EvalContextLoop.
A rule is added automatically when the same failure pattern appears in
three or more eval runs for the same training phase and modality.

"""
