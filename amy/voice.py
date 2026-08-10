"""
Voice output for Amy.

Speaks text aloud using pyttsx3 (offline, cross-platform) if available,
otherwise falls back to printing.  Callers should not depend on audio
actually playing — treat it as best-effort.
"""

from __future__ import annotations

import os

_ENGINE = None
_VOICE_ENABLED = os.environ.get("AMY_VOICE", "1") == "1"


def _get_engine():
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        engine.setProperty("volume", 0.9)
        _ENGINE = engine
        return _ENGINE
    except Exception:
        return None


def speak(text: str) -> None:
    """Convert text to speech. Falls back silently if TTS is unavailable."""
    if not _VOICE_ENABLED:
        return
    engine = _get_engine()
    if engine is None:
        return
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception:
        pass


def set_voice_enabled(enabled: bool) -> None:
    global _VOICE_ENABLED
    _VOICE_ENABLED = enabled
