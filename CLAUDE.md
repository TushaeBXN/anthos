# CLAUDE.md

Project: **Anthos** — Thought-Token Bifurcated Recurrent Transformer

A PyTorch research model with a continuous self-improvement loop.  This file
is the authoritative source of hard rules for agents working in this repo.
It is maintained both by humans and automatically by `LearningsManager`
(see `anthos/learnings_manager.py`).

---

## Project Layout

| Path | Purpose |
|---|---|
| `anthos/main.py` | Core architecture — `AnthosConfig`, `Anthos`, `ThoughtTokenPool` |
| `anthos/autonomous_agent.py` | Original improvement loop (do not modify logic here) |
| `anthos/agent_loop.py` | Learnings-aware wrapper — extend this, not the original |
| `anthos/learnings_manager.py` | All six context-engineering layers |
| `learnings.md` | Auto-maintained run log — do not hand-edit `---` separators |
| `tests/test_anthos.py` | Pytest suite — must stay green before any loop iteration advances |

---

## Development Rules

- Run `python -m pytest tests/ -v` before committing any change to `anthos/`.
- Never modify `autonomous_agent.py` directly — subclass via `agent_loop.py`.
- Keep every `learnings.md` entry to exactly 3 lines (enforced by `write_entry`).
- Hard rules below are permanent and are never compressed by the archive step.

---

## Hard Rules

_Auto-populated by `LearningsManager` when the same failure appears ≥3 times._
_Human-authored hard rules may also be added here manually._
