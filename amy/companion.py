"""
Amy — AI companion session orchestrator.

Session lifecycle:
    1. Load amy_self.json (identity anchor)
    2. SELECT: inject last 10 learnings into context
    3. Run conversation loop
    4. On exit: run_session_end() — isolated sequential calls to
         a) memory.update_memory()
         b) state.update_state()
         c) learnings.append_learning()  +  learnings.compress_learnings()

Nothing in this file writes to learnings, memory, or state during the
conversation itself — only at session end via run_session_end().
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

from amy import memory, state, learnings, thoughts, voice, tools


# ---------------------------------------------------------------------------
# Config & identity
# ---------------------------------------------------------------------------

_SELF_PATH = os.path.join(os.path.dirname(__file__), "amy_self.json")

_LLM_MODEL = os.environ.get("AMY_MODEL", "claude-sonnet-5")
_VOICE_ON = os.environ.get("AMY_VOICE", "0") == "1"


def _load_self() -> dict:
    with open(_SELF_PATH) as f:
        return json.load(f)


def _build_system_prompt(amy_self: dict, recent_learnings: str) -> str:
    beliefs = "\n".join(f"- {b}" for b in amy_self.get("founding_beliefs", []))
    style = amy_self.get("voice_style", "warm and direct")

    parts = [
        f"You are {amy_self['name']}, an AI companion.",
        "",
        "## Core Beliefs",
        beliefs,
        "",
        f"## Voice Style",
        style,
    ]

    if recent_learnings.strip():
        parts += ["", recent_learnings.strip()]

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LLM call (Anthropic SDK)
# ---------------------------------------------------------------------------

def _call_llm(system: str, history: list[dict]) -> str:
    try:
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=_LLM_MODEL,
            max_tokens=1024,
            system=system,
            messages=history,
        )
        return response.content[0].text
    except ImportError:
        return "[anthropic SDK not installed — install with: pip install anthropic]"
    except Exception as exc:
        return f"[LLM error: {exc}]"


# ---------------------------------------------------------------------------
# Session end — ISOLATION: three sequential, non-interleaved writes
# ---------------------------------------------------------------------------

def run_session_end(transcript: list[dict], amy_self: dict) -> None:
    """
    Called once when the session exits.

    Order is fixed and must not be changed:
        1. memory  — ChromaDB episodic store
        2. state   — JSON state file
        3. learnings — learnings.md + compress check

    Each function is self-contained; no shared mutable state between them.
    """
    session_summary = thoughts.summarize_session(transcript)
    founding_beliefs = amy_self.get("founding_beliefs", [])

    # 1. MEMORY
    memory.update_memory(transcript)

    # 2. STATE
    state.update_state(session_summary)

    # 3. LEARNINGS (write then compress)
    learnings.append_learning(session_summary, founding_beliefs)
    learnings.compress_learnings()


# ---------------------------------------------------------------------------
# Main conversation loop
# ---------------------------------------------------------------------------

def run_session() -> None:
    amy_self = _load_self()

    # SELECT: load last 10 learnings into context
    recent_learnings = learnings.load_recent_learnings(n=10)
    system_prompt = _build_system_prompt(amy_self, recent_learnings)

    if _VOICE_ON:
        voice.set_voice_enabled(True)

    transcript: list[dict] = []
    name = amy_self.get("name", "Amy")

    print(f"{name} is ready. Type 'quit' or 'exit' to end the session.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.lower() in {"quit", "exit", "goodbye", "bye"}:
            break

        if not user_input:
            continue

        transcript.append({"role": "user", "content": user_input})

        reply = _call_llm(system_prompt, transcript)
        transcript.append({"role": "assistant", "content": reply})

        print(f"{name}: {reply}\n")
        if _VOICE_ON:
            voice.speak(reply)

    if transcript:
        print("\n[Session ended — saving memory, state, and learnings...]")
        run_session_end(transcript, amy_self)
        print("[Done.]\n")


if __name__ == "__main__":
    run_session()
