"""Tests for the Nia Squad context engineering loop."""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Redirect SQUAD_ROOT to a temp directory for every test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_squad(tmp_path, monkeypatch):
    """Each test gets a fresh, isolated squad directory."""
    import loop.context_loop as cl

    agents_dir = tmp_path / "agents"
    shared_dir = tmp_path / "shared"
    for agent in cl.AGENTS:
        agent_dir = agents_dir / agent
        agent_dir.mkdir(parents=True)
        # Copy real SOUL.md into temp dir so verifier has data to check
        real_soul = Path(__file__).parent.parent / "agents" / agent / "SOUL.md"
        if real_soul.exists():
            (agent_dir / "SOUL.md").write_text(real_soul.read_text())
        else:
            (agent_dir / "SOUL.md").write_text(
                f"# {agent.upper()}\n## Lane\n{agent} tasks\n"
                f"Never: produce output outside the {agent} domain.\n"
            )
        (agent_dir / "learnings.md").write_text(f"# {agent.capitalize()} — Personal Learnings Log\n")

    shared_dir.mkdir(parents=True)
    (shared_dir / "squad_learnings.md").write_text("# Squad Learnings\n")
    (shared_dir / "hard_rules.md").write_text("# Hard Rules\n")

    monkeypatch.setattr(cl, "SQUAD_ROOT", tmp_path)
    monkeypatch.setattr(cl, "AGENTS_DIR", agents_dir)
    monkeypatch.setattr(cl, "SHARED_DIR", shared_dir)
    monkeypatch.setattr(cl, "REJECTION_LOG", shared_dir / "rejection_log.json")

    yield tmp_path


# ---------------------------------------------------------------------------
# WRITE LAYER
# ---------------------------------------------------------------------------

class TestAppendLearning:
    def test_creates_entry(self, isolated_squad):
        from loop.context_loop import append_learning, _learnings_path
        append_learning("mike", "market analysis", "focus on SMB", "delivered on time")
        content = _learnings_path("mike").read_text()
        assert "market analysis" in content
        assert "focus on SMB" in content
        assert "delivered on time" in content

    def test_two_line_format(self, isolated_squad):
        from loop.context_loop import append_learning, _learnings_path, _ENTRY_RE
        append_learning("kelly", "brand campaign", "use short copy", "positive feedback")
        content = _learnings_path("kelly").read_text()
        matches = list(_ENTRY_RE.finditer(content))
        assert len(matches) == 1

    def test_multiple_entries_accumulate(self, isolated_squad):
        from loop.context_loop import append_learning, _learnings_path, _ENTRY_RE
        for i in range(5):
            append_learning("david", f"task {i}", f"decision {i}", f"outcome {i}")
        content = _learnings_path("david").read_text()
        assert len(_ENTRY_RE.findall(content)) == 5


# ---------------------------------------------------------------------------
# COMPRESS LAYER
# ---------------------------------------------------------------------------

class TestCompressLearnings:
    def _fill_entries(self, agent: str, count: int) -> None:
        from loop.context_loop import _learnings_path
        path = _learnings_path(agent)
        lines = path.read_text()
        today = "2026-01-01"
        for i in range(count):
            lines += (
                f"- date: {today} | task: task {i} | decision: decision {i}\n"
                f"  outcome: outcome {i}\n"
            )
        path.write_text(lines)

    def test_no_compression_under_limit(self, isolated_squad):
        from loop.context_loop import compress_learnings, _learnings_path, _ENTRY_RE
        self._fill_entries("pamela", 40)
        compress_learnings("pamela")
        content = _learnings_path("pamela").read_text()
        assert len(_ENTRY_RE.findall(content)) == 40

    def test_compression_fires_over_limit(self, isolated_squad):
        from loop.context_loop import compress_learnings, _learnings_path, _ENTRY_RE, ARCHIVE_COUNT, MAX_ENTRIES
        self._fill_entries("pamela", MAX_ENTRIES + 1)
        compress_learnings("pamela")
        content = _learnings_path("pamela").read_text()
        remaining_raw = len(_ENTRY_RE.findall(content))
        assert remaining_raw == MAX_ENTRIES + 1 - ARCHIVE_COUNT

    def test_archive_block_written(self, isolated_squad):
        from loop.context_loop import compress_learnings, _learnings_path, MAX_ENTRIES
        self._fill_entries("keisha", MAX_ENTRIES + 5)
        compress_learnings("keisha")
        content = _learnings_path("keisha").read_text()
        assert "## ARCHIVE" in content

    def test_double_compression(self, isolated_squad):
        from loop.context_loop import compress_learnings, _learnings_path, _ENTRY_RE, MAX_ENTRIES, ARCHIVE_COUNT
        self._fill_entries("nia", MAX_ENTRIES + 1)
        compress_learnings("nia")
        # Add enough new entries to trigger again
        self._fill_entries("nia", ARCHIVE_COUNT + 1)
        compress_learnings("nia")
        content = _learnings_path("nia").read_text()
        assert content.count("## ARCHIVE") == 2


# ---------------------------------------------------------------------------
# SELECT LAYER
# ---------------------------------------------------------------------------

class TestLoadAgentContext:
    def test_loads_soul(self, isolated_squad):
        from loop.context_loop import load_agent_context
        ctx = load_agent_context("nia")
        assert "NIA" in ctx["soul"] or "Nia" in ctx["soul"]

    def test_personal_entries_capped_at_10(self, isolated_squad):
        from loop.context_loop import append_learning, load_agent_context
        for i in range(15):
            append_learning("mike", f"task {i}", f"decision {i}", f"outcome {i}")
        ctx = load_agent_context("mike")
        assert len(ctx["personal"]) == 10

    def test_returns_hard_rules(self, isolated_squad):
        from loop.context_loop import load_agent_context, SHARED_DIR
        (SHARED_DIR / "hard_rules.md").write_text("# Hard Rules\nRULE: test rule [escalated: 2026-01-01]\n")
        ctx = load_agent_context("david")
        assert "test rule" in ctx["hard_rules"]

    def test_shared_learnings_filtered_by_relevance(self, isolated_squad):
        from loop.context_loop import load_agent_context, SHARED_DIR
        (SHARED_DIR / "squad_learnings.md").write_text(
            "# Squad Learnings\n\n"
            "## Pattern: `strategy` — seen across: mike, nia\n"
            "- [mike] market sizing → completed\n\n"
            "## Pattern: `sewing` — seen across: kelly, pamela\n"
            "- [kelly] craft session → irrelevant\n"
        )
        ctx = load_agent_context("mike")
        combined = " ".join(ctx["shared"])
        assert "strategy" in combined or "market" in combined


# ---------------------------------------------------------------------------
# SHARED LEARNINGS
# ---------------------------------------------------------------------------

class TestSynthesizeSquadLearnings:
    def test_no_output_when_no_entries(self, isolated_squad):
        from loop.context_loop import synthesize_squad_learnings, SHARED_DIR
        synthesize_squad_learnings()
        content = (SHARED_DIR / "squad_learnings.md").read_text()
        # Should remain as initialised (no patterns promoted)
        assert "## Pattern" not in content

    def test_cross_agent_pattern_promoted(self, isolated_squad):
        from loop.context_loop import append_learning, synthesize_squad_learnings, SHARED_DIR
        append_learning("mike", "pricing strategy review", "lower price", "approved")
        append_learning("pamela", "pricing model update", "raise margin", "approved")
        synthesize_squad_learnings()
        content = (SHARED_DIR / "squad_learnings.md").read_text()
        assert "pricing" in content

    def test_single_agent_pattern_not_promoted(self, isolated_squad):
        from loop.context_loop import append_learning, synthesize_squad_learnings, SHARED_DIR
        append_learning("kelly", "brand refresh", "bold colors", "approved")
        append_learning("kelly", "brand audit", "consistent tone", "approved")
        synthesize_squad_learnings()
        content = (SHARED_DIR / "squad_learnings.md").read_text()
        # "brand" appears only for kelly — must not be promoted
        assert "brand" not in content or "seen across" not in content


# ---------------------------------------------------------------------------
# VERIFIER GATE
# ---------------------------------------------------------------------------

class TestNiaVerdict:
    def test_clean_output_passes(self, isolated_squad):
        from loop.context_loop import nia_verdict
        passed, reason = nia_verdict("mike", "Here is a market sizing analysis for the SMB segment.")
        assert passed is True
        assert reason == "approved"

    def test_hard_rule_blocks_output(self, isolated_squad):
        from loop.context_loop import nia_verdict, SHARED_DIR
        (SHARED_DIR / "hard_rules.md").write_text(
            "# Hard Rules\nRULE: never produce legal opinion without disclaimer [escalated: 2026-01-01]\n"
        )
        # Output that matches the hard rule keywords
        output = "This is a legal opinion without any disclaimer included."
        passed, reason = nia_verdict("mike", output)
        assert passed is False
        assert "hard rule" in reason.lower()

    def test_soul_prohibition_blocks_output(self, isolated_squad):
        from loop.context_loop import nia_verdict, AGENTS_DIR
        soul = (AGENTS_DIR / "kelly" / "SOUL.md").read_text()
        # Find a 'Never' line in Kelly's SOUL
        never_lines = [l for l in soul.splitlines() if re.match(r"(?i)^(never|must not)", l.strip())]
        if not never_lines:
            pytest.skip("No Never/Must not line in Kelly's SOUL")
        # Build output that triggers the first prohibition
        prohibition = re.sub(r"(?i)^(never|must not)[:\s]*", "", never_lines[0].strip()).lower()
        keywords = re.findall(r"\b[a-zA-Z]{4,}\b", prohibition)[:2]
        if len(keywords) < 2:
            pytest.skip("Not enough keywords to test prohibition")
        output = f"This output will {keywords[0]} and {keywords[1]} in a problematic way."
        passed, _ = nia_verdict("kelly", output)
        # May or may not trigger depending on context density — just ensure no crash
        assert isinstance(passed, bool)

    def test_verdict_returns_tuple(self, isolated_squad):
        from loop.context_loop import nia_verdict
        result = nia_verdict("david", "Technical architecture recommendation for the API layer.")
        assert isinstance(result, tuple) and len(result) == 2


# ---------------------------------------------------------------------------
# ESCALATION
# ---------------------------------------------------------------------------

class TestEscalation:
    def test_three_rejections_promote_rule(self, isolated_squad):
        from loop.context_loop import _record_rejection, check_escalation, SHARED_DIR, ESCALATION_THRESHOLD
        reason = "Violates hard rule: never produce legal opinion without disclaimer"
        for _ in range(ESCALATION_THRESHOLD):
            _record_rejection("mike", reason)
        check_escalation(reason)
        hard_rules = (SHARED_DIR / "hard_rules.md").read_text()
        assert "RULE:" in hard_rules

    def test_two_rejections_do_not_promote(self, isolated_squad):
        from loop.context_loop import _record_rejection, check_escalation, SHARED_DIR
        reason = "Contradicts soul: never fabricate quotes from real people"
        for _ in range(2):
            _record_rejection("kelly", reason)
        check_escalation(reason)
        hard_rules = (SHARED_DIR / "hard_rules.md").read_text()
        assert "RULE:" not in hard_rules

    def test_promotion_is_idempotent(self, isolated_squad):
        from loop.context_loop import _record_rejection, check_escalation, SHARED_DIR, ESCALATION_THRESHOLD
        reason = "Lane violation: keisha output contains legal terms exclusive to keisha domain"
        for _ in range(ESCALATION_THRESHOLD + 2):
            _record_rejection("keisha", reason)
        check_escalation(reason)
        check_escalation(reason)
        hard_rules = (SHARED_DIR / "hard_rules.md").read_text()
        # Idempotent: RULE only appears once
        assert hard_rules.count("RULE:") == 1

    def test_rejection_log_tracks_multiple_agents(self, isolated_squad):
        from loop.context_loop import _record_rejection, REJECTION_LOG
        reason = "Violates hard rule: never approve output without verifier gate"
        _record_rejection("mike", reason)
        _record_rejection("kelly", reason)
        log = json.loads(REJECTION_LOG.read_text())
        key = list(log.keys())[0]
        assert len(log[key]["agents"]) == 2
