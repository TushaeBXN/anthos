"""
anthos/synapse.py — Persistent conversational context engine for Anthos.

Solves two problems:
  1. History resets to zero on every quit — sessions are lost.
  2. History grows unboundedly within a session — context window fills up
     faster than the native arch can handle (small model, ~200 new tokens).

Features:
  - Saves/loads sessions to disk between runs
  - Tracks facts the user has stated about themselves (persists forever)
  - Compresses old turns using ESCompressor (no LLM call needed) or
    an optional summarize_fn for richer summaries
  - Exports history in both formats:
      .as_dict_history()   → list[dict]  for chat_anthos.py (Qwen LoRA)
      .as_tuple_history()  → list[tuple] for chat_native.py (native arch)

Usage (chat_anthos.py):
    from anthos.synapse import Synapse
    brain = Synapse.load("~/.anthos/sessions/anthos_session.json")
    history = brain.as_dict_history()
    # ... chat loop ...
    brain.add_turn("user", user_input)
    brain.add_turn("assistant", response)
    brain.save()

Usage (chat_native.py):
    from anthos.synapse import Synapse
    brain = Synapse.load("~/.anthos/sessions/anthos_session.json")
    history = brain.as_tuple_history()
    # ... chat loop ...
    brain.add_turn("user", user_input)
    brain.add_turn("assistant", response)
    brain.save()
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from anthos.memory_compress import ESCompressor


_DEFAULT_PATH = Path.home() / ".anthos" / "sessions" / "anthos_session.json"
_COMPRESS_AFTER = 10   # compress oldest turns when history exceeds this
_KEEP_RECENT    = 6    # always keep this many recent turns uncompressed


class Synapse:
    def __init__(
        self,
        path: Path,
        summarize_fn: Optional[Callable[[list[dict]], str]] = None,
    ):
        self.path = Path(path).expanduser()
        self.summarize_fn = summarize_fn
        self._compressor = ESCompressor(confidence=3)

        # Persistent state (saved to disk)
        self.turns: list[dict] = []          # {role, content, ts}
        self.compressed: list[str] = []      # summaries of older turn batches
        self.user_facts: list[str] = []      # things the user has told Anthos

    # ── Turn management ───────────────────────────────────────────────────────

    def add_turn(self, role: str, content: str):
        self.turns.append({
            "role": role,
            "content": content,
            "ts": datetime.now().isoformat(timespec="seconds"),
        })
        self._extract_user_facts(role, content)

        if len(self.turns) > _COMPRESS_AFTER:
            self._compress_oldest()

    # ── History export (two formats for the two chat scripts) ─────────────────

    def as_dict_history(self) -> list[dict]:
        """For chat_anthos.py — returns list[{"role": ..., "content": ...}]."""
        result = []
        if self.compressed:
            summary = " | ".join(self.compressed[-3:])
            result.append({"role": "system", "content": f"[Earlier context] {summary}"})
        for t in self.turns[-_KEEP_RECENT:]:
            result.append({"role": t["role"], "content": t["content"]})
        return result

    def as_tuple_history(self) -> list[tuple[str, str]]:
        """For chat_native.py — returns list[(user_text, assistant_text)]."""
        pairs = []
        recent = self.turns[-_KEEP_RECENT:]
        i = 0
        while i < len(recent) - 1:
            if recent[i]["role"] == "user" and recent[i + 1]["role"] == "assistant":
                pairs.append((recent[i]["content"], recent[i + 1]["content"]))
                i += 2
            else:
                i += 1
        return pairs

    # ── System prompt injection ───────────────────────────────────────────────

    def context_injection(self) -> str:
        """
        Returns a string to APPEND to the SYSTEM prompt at session start.
        Contains: known user facts + compressed prior session summaries.
        Empty string if nothing has been learned yet.
        """
        lines = []

        if self.user_facts:
            lines.append("\n[What Anthos knows about this user]")
            for f in self.user_facts[-10:]:
                lines.append(f"  • {f}")

        if self.compressed:
            lines.append("\n[Prior session summary]")
            for s in self.compressed[-2:]:
                lines.append(f"  {s}")

        return "\n".join(lines)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "turns": self.turns,
            "compressed": self.compressed,
            "user_facts": self.user_facts,
        }
        self.path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(
        cls,
        path: str | Path = _DEFAULT_PATH,
        summarize_fn=None,
    ) -> "Synapse":
        instance = cls(path=path, summarize_fn=summarize_fn)
        p = Path(path).expanduser()
        if p.exists():
            data = json.loads(p.read_text())
            instance.turns      = data.get("turns", [])
            instance.compressed = data.get("compressed", [])
            instance.user_facts = data.get("user_facts", [])
        return instance

    def new_session(self):
        """
        Call this to start a fresh session while keeping long-term facts.
        Compresses any remaining turns into history before clearing.
        """
        if self.turns:
            self._compress_oldest(force_all=True)
        self.turns = []

    # ── Internal ──────────────────────────────────────────────────────────────

    def _compress_oldest(self, force_all: bool = False):
        cutoff = 0 if force_all else _KEEP_RECENT
        to_compress = self.turns[:-cutoff] if cutoff else self.turns[:]
        self.turns  = self.turns[-cutoff:] if cutoff else []

        if not to_compress:
            return

        if self.summarize_fn:
            msgs = [{"role": t["role"], "content": t["content"]} for t in to_compress]
            summary = self.summarize_fn(msgs)
        else:
            # Fallback: ES-compress each turn and join
            parts = []
            for t in to_compress:
                compressed = self._compressor.compress(t["content"])
                parts.append(f"{t['role']}: {compressed}")
            summary = " | ".join(parts)

        self.compressed.append(summary)

    _FACT_PATTERNS = [
        (r"\bmy name is\s+(\w+)",                      lambda g: f"User name: {g[0]}"),
        (r"\bi(?:'m| am)\s+(?:a |an )?(\w[\w\s]{2,30})", lambda g: f"User is: {g[0].strip()}"),
        (r"\bi work (?:at|for|with)\s+([^.,!?]{3,40})",  lambda g: f"User works at: {g[0].strip()}"),
        (r"\bmy project (?:is |called |named )?(.{3,60})", lambda g: f"User project: {g[0].strip()}"),
        (r"\bi(?:'m| am) working on\s+(.{3,60})",      lambda g: f"User working on: {g[0].strip()}"),
        (r"\bremember that\s+(.{3,120})",               lambda g: f"Remember: {g[0].strip()}"),
    ]

    def _extract_user_facts(self, role: str, content: str):
        if role != "user":
            return
        for pattern, fmt in self._FACT_PATTERNS:
            m = re.search(pattern, content, re.IGNORECASE)
            if m:
                fact = fmt(m.groups())
                if fact not in self.user_facts:
                    self.user_facts.append(fact)
