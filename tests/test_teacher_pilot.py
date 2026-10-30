"""Teacher-pilot helpers: passage sampling from the token shards and the quality checks."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import teacher_pilot as T


def test_ungrounded_counts_invented_names_and_numbers():
    src = "The Nile flows north through Egypt. It is about 6,650 km long."
    good = "The Nile flows north through Egypt and is about 6,650 km long."
    bad = "The river, studied by Herodotus in 450 BC, flows north."
    assert T.ungrounded(src, good) == 0.0
    assert T.ungrounded(src, bad) > 0.5


def test_judge_flags():
    src = "plants make food from light " * 30
    assert "refusal" in T.judge(src, "I'm sorry, I cannot help with that.")["flags"]
    assert "too_short" in T.judge(src, "Plants make food.")["flags"]
    assert "repetitive" in T.judge(src, ("the same words again and again " * 60))["flags"]


def test_sampling_returns_whole_documents():
    out = T.sample_passages("data/pretrain_cl100k", 3, seed=1)
    assert len(out) == 3 and all(len(p.split()) > 100 for p in out)
