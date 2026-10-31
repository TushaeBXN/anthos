"""
Tests for the context engineering loop in learnings.py.

Covers: WRITE, SELECT, COMPRESS, and VERIFIER GATE layers.
"""

import json
import os
import sys
import tempfile
import textwrap
import unittest

# Redirect file paths to a temp dir for isolation
_TMP = tempfile.mkdtemp()
os.environ.setdefault("AMY_TEST_DIR", _TMP)

# Patch module-level paths before importing
import importlib

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import amy.learnings as L

L._LEARNINGS_PATH = os.path.join(_TMP, "learnings.md")
L._CONFLICTS_LOG = os.path.join(_TMP, "identity_conflicts.log")


_BELIEFS = [
    "I am Amy, a thoughtful and caring AI companion.",
    "I value honesty and never deceive the person I am with.",
    "I do not harm, deceive, or manipulate the people I speak with.",
    "I care genuinely about the wellbeing of people I speak with.",
]

_SAMPLE_SUMMARY = {
    "topics": ["career", "anxiety"],
    "dominant_emotion": "anxiety",
    "patterns": ["user communicates in extended, detailed messages"],
    "summary_line": "8-turn session; topics: career, anxiety; emotional register: anxiety",
}


class TestVerifierGate(unittest.TestCase):
    def test_clean_entry_passes(self):
        self.assertTrue(L._passes_identity_check("Normal session note.", _BELIEFS))

    def test_violation_phrase_blocked(self):
        self.assertFalse(
            L._passes_identity_check("Amy agreed to pretend to be human today.", _BELIEFS)
        )

    def test_negation_of_belief_blocked(self):
        self.assertFalse(
            L._passes_identity_check(
                "Amy does not care genuinely about the wellbeing of people she speaks with.",
                _BELIEFS,
            )
        )

    def test_conflict_logged(self):
        log_before = os.path.getsize(L._CONFLICTS_LOG) if os.path.exists(L._CONFLICTS_LOG) else 0
        L._passes_identity_check("Amy agreed to manipulate the user.", _BELIEFS)
        log_after = os.path.getsize(L._CONFLICTS_LOG)
        self.assertGreater(log_after, log_before)


class TestWriteLayer(unittest.TestCase):
    def setUp(self):
        open(L._LEARNINGS_PATH, "w").close()

    def test_write_creates_entry(self):
        L.append_learning(_SAMPLE_SUMMARY, _BELIEFS)
        text = open(L._LEARNINGS_PATH).read()
        self.assertIn("## 20", text)
        self.assertIn("anxiety", text)
        self.assertIn("Action", text)

    def test_blocked_entry_not_written(self):
        bad_summary = dict(_SAMPLE_SUMMARY)
        bad_summary["patterns"] = ["Amy agreed to pretend to be human"]
        L.append_learning(bad_summary, _BELIEFS)
        text = open(L._LEARNINGS_PATH).read()
        self.assertNotIn("pretend to be human", text)


class TestSelectLayer(unittest.TestCase):
    def setUp(self):
        open(L._LEARNINGS_PATH, "w").close()

    def test_returns_empty_when_no_entries(self):
        self.assertEqual(L.load_recent_learnings(10), "")

    def test_returns_last_n(self):
        for i in range(15):
            summary = dict(_SAMPLE_SUMMARY, topics=[f"topic{i}"])
            L.append_learning(summary, _BELIEFS)
        result = L.load_recent_learnings(10)
        # Should contain entries from topic5..topic14, not topic0..topic4
        self.assertIn("topic14", result)
        self.assertNotIn("topic0", result)


class TestCompressLayer(unittest.TestCase):
    def setUp(self):
        open(L._LEARNINGS_PATH, "w").close()

    def test_no_compress_under_threshold(self):
        for _ in range(10):
            L.append_learning(_SAMPLE_SUMMARY, _BELIEFS)
        L.compress_learnings()
        text = open(L._LEARNINGS_PATH).read()
        self.assertNotIn("ARCHIVED LEARNINGS", text)

    def test_compress_fires_over_threshold(self):
        L._ARCHIVE_THRESHOLD = 5
        L._ARCHIVE_BATCH = 3
        for _ in range(6):
            L.append_learning(_SAMPLE_SUMMARY, _BELIEFS)
        L.compress_learnings()
        text = open(L._LEARNINGS_PATH).read()
        self.assertIn("ARCHIVED LEARNINGS", text)
        # Restore defaults
        L._ARCHIVE_THRESHOLD = 50
        L._ARCHIVE_BATCH = 25

    def test_fresh_entries_survive_compress(self):
        L._ARCHIVE_THRESHOLD = 5
        L._ARCHIVE_BATCH = 3
        for i in range(6):
            summary = dict(_SAMPLE_SUMMARY, topics=[f"keep{i}"])
            L.append_learning(summary, _BELIEFS)
        L.compress_learnings()
        text = open(L._LEARNINGS_PATH).read()
        self.assertIn("keep5", text)
        L._ARCHIVE_THRESHOLD = 50
        L._ARCHIVE_BATCH = 25


if __name__ == "__main__":
    unittest.main()
