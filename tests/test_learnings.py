"""
Tests for the context engineering loop infrastructure.

Covers all six layers:
  WRITE     test_write_entry_format, test_write_appends_multiple
  SELECT    test_select_returns_hard_rules, test_select_limits_recent
  COMPRESS  test_compress_triggers_at_50, test_compress_preserves_hard
  VERIFY    test_run_tests_passes, test_run_tests_fails_on_bad_suite
  ESCALATE  test_escalation_promotes_to_claude_md
  ISOLATE   test_subagent_context_excludes_history
"""

import re
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

# Insert project root so learnings_manager is importable without triggering
# anthos/__init__.py (which has a torch dependency).
sys.path.insert(0, str(Path(__file__).parent.parent))
import learnings_manager as lm_mod
from learnings_manager import LearningsManager, _HARD_TAG, _ENTRY_SEP


# ── Fixture ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_lm(tmp_path):
    """LearningsManager backed by temp files so tests don't touch the real repo."""
    lm = LearningsManager(
        learnings_path=tmp_path / "learnings.md",
        claude_md_path=tmp_path / "CLAUDE.md",
    )
    return lm


# ── WRITE LAYER ──────────────────────────────────────────────────────────────

class TestWriteLayer:
    def test_write_creates_file(self, tmp_lm):
        assert tmp_lm.learnings_path.exists()

    def test_write_entry_format(self, tmp_lm):
        tmp_lm.write_entry(
            task="increase learning rate",
            succeeded="loss dropped",
            failed="none",
            rule="LR 1e-3 works best for dim=64",
        )
        text = tmp_lm.learnings_path.read_text()
        # Must have today's date
        assert re.search(r"\d{4}-\d{2}-\d{2}", text)
        # Must have OK: and FAIL: on the same line
        assert "OK:" in text and "FAIL:" in text
        # Must have RULE:
        assert "RULE:" in text

    def test_write_entry_exactly_three_content_lines(self, tmp_lm):
        tmp_lm.write_entry(
            task="some task",
            succeeded="it worked",
            failed="nothing",
            rule="do this always",
        )
        entries = tmp_lm._load_entries()
        assert len(entries) == 1
        lines = [l for l in entries[0].splitlines() if l.strip()]
        assert len(lines) == 3, f"Expected 3 lines, got {len(lines)}: {lines}"

    def test_write_appends_multiple(self, tmp_lm):
        for i in range(3):
            tmp_lm.write_entry(
                task=f"task {i}",
                succeeded="ok",
                failed="none",
                rule=f"rule {i}",
            )
        entries = tmp_lm._load_entries()
        assert len(entries) == 3

    def test_write_hard_flag(self, tmp_lm):
        tmp_lm.write_entry(
            task="critical task",
            succeeded="ok",
            failed="none",
            rule="never do X",
            hard=True,
        )
        text = tmp_lm.learnings_path.read_text()
        assert _HARD_TAG in text


# ── SELECT LAYER ─────────────────────────────────────────────────────────────

class TestSelectLayer:
    def test_select_empty_returns_empty_string(self, tmp_lm):
        assert tmp_lm.select_context() == ""

    def test_select_returns_recent_entries(self, tmp_lm):
        for i in range(5):
            tmp_lm.write_entry(task=f"t{i}", succeeded="ok", failed="none", rule=f"r{i}")
        ctx = tmp_lm.select_context()
        assert "t4" in ctx  # most recent should appear

    def test_select_limits_to_10_recent(self, tmp_lm):
        # Write 15 non-hard entries; only last 10 (task5–task14) appear in context
        for i in range(15):
            tmp_lm.write_entry(task=f"task{i}", succeeded="ok", failed="none", rule=f"rule{i}")
        ctx = tmp_lm.select_context()
        # task0–task4 are outside the 10-entry window
        for excluded in range(5):
            # Use word-boundary-aware check: "task0 " or "task0\n" etc.
            assert not re.search(rf"\btask{excluded}\b", ctx), f"task{excluded} unexpectedly in context"
        # task14 (most recent) must be present
        assert "task14" in ctx

    def test_select_always_includes_hard_rules(self, tmp_lm):
        # Write 15 non-hard entries and 1 hard entry at the start
        tmp_lm.write_entry(task="hard task", succeeded="ok", failed="none", rule="permanent rule", hard=True)
        for i in range(15):
            tmp_lm.write_entry(task=f"regular{i}", succeeded="ok", failed="none", rule=f"rule{i}")
        ctx = tmp_lm.select_context()
        # The hard entry is always present
        assert "permanent rule" in ctx
        assert _HARD_TAG in ctx

    def test_select_includes_header(self, tmp_lm):
        tmp_lm.write_entry(task="t", succeeded="ok", failed="none", rule="r")
        ctx = tmp_lm.select_context()
        assert "Injected Learnings" in ctx


# ── COMPRESS LAYER ───────────────────────────────────────────────────────────

class TestCompressLayer:
    def test_compress_triggers_above_50(self, tmp_lm):
        # Write 51 entries; compression should fire
        for i in range(51):
            tmp_lm.write_entry(task=f"task{i}", succeeded="ok", failed="none", rule=f"rule{i}")
        text = tmp_lm.learnings_path.read_text()
        assert "ARCHIVE" in text

    def test_compress_archive_contains_summary(self, tmp_lm):
        for i in range(51):
            tmp_lm.write_entry(task=f"do thing {i}", succeeded="ok", failed="none", rule=f"rule{i}")
        text = tmp_lm.learnings_path.read_text()
        assert "Tasks" in text or "Rules Captured" in text

    def test_compress_preserves_hard_rules(self, tmp_lm):
        # Hard entry first, then fill to trigger compression
        tmp_lm.write_entry(task="hard task", succeeded="ok", failed="none", rule="never skip tests", hard=True)
        for i in range(51):
            tmp_lm.write_entry(task=f"regular{i}", succeeded="ok", failed="none", rule=f"r{i}")
        entries = tmp_lm._load_entries()
        hard = [e for e in entries if _HARD_TAG in e]
        assert len(hard) == 1
        assert "never skip tests" in hard[0]

    def test_compress_keeps_recent_entries(self, tmp_lm):
        for i in range(51):
            tmp_lm.write_entry(task=f"task{i}", succeeded="ok", failed="none", rule=f"rule{i}")
        entries = tmp_lm._load_entries()
        # After compressing 25 oldest from 51, we should have 26 remaining raw entries
        non_hard = [e for e in entries if _HARD_TAG not in e]
        assert len(non_hard) == 26


# ── VERIFIER GATE ────────────────────────────────────────────────────────────

class TestVerifierGate:
    def test_run_tests_passes_on_real_suite(self, tmp_lm):
        """The real test suite must pass — this is the definitive green-gate check."""
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch not installed — test_anthos.py cannot be collected")
        passed, output = tmp_lm.run_tests()
        assert passed, f"Test suite failed:\n{output}"

    def test_run_tests_returns_tuple(self, tmp_lm):
        passed, output = tmp_lm.run_tests()
        assert isinstance(passed, bool)
        assert isinstance(output, str)

    def test_run_tests_captures_output(self, tmp_lm):
        _, output = tmp_lm.run_tests()
        # pytest always emits something
        assert len(output) > 0

    def test_run_tests_fails_on_bad_test_dir(self, tmp_path):
        """A manager pointing at a non-existent tests/ dir should return failed."""
        lm = LearningsManager(
            learnings_path=tmp_path / "learnings.md",
            claude_md_path=tmp_path / "CLAUDE.md",
        )
        # Monkeypatch _ROOT so pytest looks in tmp_path (no tests/ there)
        original_root = lm_mod._ROOT
        lm_mod._ROOT = tmp_path
        try:
            passed, output = lm.run_tests()
            assert not passed
        finally:
            lm_mod._ROOT = original_root


# ── ESCALATION ───────────────────────────────────────────────────────────────

class TestEscalation:
    def test_escalation_promotes_after_3_occurrences(self, tmp_lm):
        for _ in range(3):
            tmp_lm.write_entry(
                task="experiment",
                succeeded="ok",
                failed="gradient explodes on lr=1e-1",
                rule="cap lr at 1e-3",
            )
        claude_text = tmp_lm.claude_md_path.read_text()
        assert "cap lr at 1e-3" in claude_text

    def test_escalation_not_triggered_below_threshold(self, tmp_lm):
        for _ in range(2):
            tmp_lm.write_entry(
                task="experiment",
                succeeded="ok",
                failed="unique failure xyz",
                rule="handle xyz carefully",
            )
        claude_text = tmp_lm.claude_md_path.read_text()
        assert "handle xyz carefully" not in claude_text

    def test_escalation_is_idempotent(self, tmp_lm):
        for _ in range(6):  # 6 occurrences should not duplicate the rule
            tmp_lm.write_entry(
                task="exp",
                succeeded="ok",
                failed="oom on batch=512",
                rule="keep batch ≤ 128",
            )
        claude_text = tmp_lm.claude_md_path.read_text()
        assert claude_text.count("keep batch ≤ 128") == 1

    def test_escalation_places_rule_under_hard_rules_heading(self, tmp_lm):
        for _ in range(3):
            tmp_lm.write_entry(
                task="exp",
                succeeded="ok",
                failed="nan loss when n_experts=1",
                rule="always use n_experts≥2",
            )
        claude_text = tmp_lm.claude_md_path.read_text()
        hard_rules_pos = claude_text.find("## Hard Rules")
        rule_pos = claude_text.find("always use n_experts≥2")
        assert hard_rules_pos != -1
        assert rule_pos > hard_rules_pos

    def test_escalation_skips_empty_failures(self, tmp_lm):
        for _ in range(5):
            tmp_lm.write_entry(
                task="exp",
                succeeded="everything worked",
                failed="none",
                rule="keep doing this",
            )
        # "none" should never trigger escalation
        claude_text = tmp_lm.claude_md_path.read_text()
        assert "keep doing this" not in claude_text


# ── ISOLATION ────────────────────────────────────────────────────────────────

class TestIsolation:
    def test_subagent_context_contains_task(self, tmp_lm):
        ctx = tmp_lm.build_subagent_context(task_spec="train for 100 steps")
        assert "train for 100 steps" in ctx

    def test_subagent_context_contains_files(self, tmp_lm):
        ctx = tmp_lm.build_subagent_context(
            task_spec="run experiment",
            relevant_files=["anthos/main.py", "anthos/autonomous_agent.py"],
        )
        assert "anthos/main.py" in ctx
        assert "anthos/autonomous_agent.py" in ctx

    def test_subagent_context_contains_hard_rules(self, tmp_lm):
        # Inject a rule into the existing ## Hard Rules section of CLAUDE.md
        text = tmp_lm.claude_md_path.read_text()
        if "## Hard Rules" in text:
            text = text.replace("## Hard Rules\n", "## Hard Rules\n- never use lr>0.1\n", 1)
        else:
            text += "\n## Hard Rules\n- never use lr>0.1\n"
        tmp_lm.claude_md_path.write_text(text)
        ctx = tmp_lm.build_subagent_context(task_spec="test")
        assert "never use lr>0.1" in ctx

    def test_subagent_context_does_not_include_full_learnings(self, tmp_lm):
        # Write several entries; subagent context should NOT contain them
        for i in range(5):
            tmp_lm.write_entry(task=f"task{i}", succeeded="ok", failed="none", rule=f"rule{i}")
        ctx = tmp_lm.build_subagent_context(task_spec="isolated subtask")
        # Regular (non-hard) entries should not bleed into subagent context
        assert "task0" not in ctx
        assert "task4" not in ctx

    def test_subagent_context_has_no_conversation_history(self, tmp_lm):
        ctx = tmp_lm.build_subagent_context(task_spec="subtask")
        # The isolation contract: only Task, Relevant Files, Hard Rules sections
        assert "## Task" in ctx
        assert "## Hard Rules" in ctx
        # No "Injected Learnings" (that's the main-loop select context)
        assert "Injected Learnings" not in ctx
