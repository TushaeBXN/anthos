"""ExternalMemoryReader against the current Engram interface: filtering, wing mapping, and failures that are not silent."""
import os
import sys
import warnings

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytest.importorskip("engram")
from anthos.memory import ExternalMemoryReader


class FakeSearcher:
    def __init__(self, results=None, boom=None):
        self.results, self.boom, self.calls = results or [], boom, []

    def search(self, query, n=10, wing=None, room=None, **kw):
        self.calls.append(dict(query=query, n=n, wing=wing, room=room))
        if self.boom:
            raise self.boom
        return self.results


class FakeStack:
    def __init__(self, text="L0 identity + L1 story", boom=None):
        self.text, self.boom, self.wings = text, boom, []

    def wake_up(self, wing=None):
        self.wings.append(wing)
        if self.boom:
            raise self.boom
        return self.text


def reader(searcher, stack=None, **kw):
    r = ExternalMemoryReader(tokenizer=None, **kw)
    r._engram_available = True
    r._objs = (searcher, stack or FakeStack())
    return r


def test_superseded_memories_are_dropped_and_disputed_ones_are_tagged():
    s = FakeSearcher([
        {"text": "Brian built Anthos", "status": "current"},
        {"text": "Anthos has 7M parameters", "status": "superseded"},
        {"text": "Anthos uses 8 loops", "status": "disputed"},
        {"text": "plain note"},
    ])
    out = reader(s, n_results=4).retrieve("who built anthos")
    assert out == ["Brian built Anthos", "[DISPUTED] Anthos uses 8 loops", "plain note"]


def test_default_wing_means_search_everything_and_a_named_wing_is_passed_through():
    s = FakeSearcher([{"text": "x"}])
    reader(s, n_results=3).retrieve("q")
    assert s.calls[-1] == dict(query="q", n=3, wing=None, room=None)
    reader(s, engram_wing="anthos", engram_room="training", n_results=2).retrieve("q")
    assert s.calls[-1] == dict(query="q", n=2, wing="anthos", room="training")


def test_a_broken_engram_falls_back_but_says_so_once():
    r = reader(FakeSearcher(boom=RuntimeError("api moved")), fallback_memories=["fallback one", "fallback two"], n_results=1)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert r.retrieve("q") == ["fallback one"]
        assert r.retrieve("q") == ["fallback one"]
    assert len(w) == 1 and "api moved" in str(w[0].message)


def test_wake_up_context_uses_the_stack_and_survives_failure():
    stack = FakeStack()
    assert reader(FakeSearcher(), stack).wake_up_context() == "L0 identity + L1 story"
    assert stack.wings == [None]
    assert reader(FakeSearcher(), stack, engram_wing="anthos").wake_up_context() == "L0 identity + L1 story"
    assert stack.wings[-1] == "anthos"
    bad = reader(FakeSearcher(), FakeStack(boom=ValueError("nope")))
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        assert bad.wake_up_context() == ""
