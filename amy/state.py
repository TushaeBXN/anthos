"""
Session-level state tracking for Amy.

Tracks emotional register, topic drift, and behavioral patterns
across a single session and persists a summary to state.json.

Public API (called sequentially at session end by companion.py):
    update_state(session_data: dict) -> None
    load_state() -> dict
"""

import json
import os
from datetime import datetime

_STATE_PATH = os.path.join(os.path.dirname(__file__), "state.json")

_DEFAULT_STATE = {
    "last_updated": None,
    "dominant_emotion": "neutral",
    "recent_topics": [],
    "session_count": 0,
    "cumulative_patterns": [],
}


def load_state() -> dict:
    if not os.path.exists(_STATE_PATH):
        return dict(_DEFAULT_STATE)
    try:
        with open(_STATE_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return dict(_DEFAULT_STATE)


def _save_state(state: dict) -> None:
    with open(_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def update_state(session_data: dict) -> None:
    """
    Merge session-level observations into the persistent state file.

    session_data keys (all optional):
        dominant_emotion: str
        topics: list[str]
        patterns: list[str]
    """
    state = load_state()

    state["last_updated"] = datetime.utcnow().isoformat()
    state["session_count"] = state.get("session_count", 0) + 1

    if emotion := session_data.get("dominant_emotion"):
        state["dominant_emotion"] = emotion

    new_topics = session_data.get("topics", [])
    existing = state.get("recent_topics", [])
    combined = (existing + new_topics)[-20:]
    state["recent_topics"] = list(dict.fromkeys(combined))

    new_patterns = session_data.get("patterns", [])
    existing_patterns = state.get("cumulative_patterns", [])
    combined_patterns = (existing_patterns + new_patterns)[-50:]
    state["cumulative_patterns"] = list(dict.fromkeys(combined_patterns))

    _save_state(state)
