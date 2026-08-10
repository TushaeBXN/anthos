"""
Audio input (speech-to-text) for Amy.

Uses SpeechRecognition with the Google Web Speech API by default.
Falls back to returning None if microphone or library is unavailable.
"""

from __future__ import annotations

import os

_SR_AVAILABLE = False
try:
    import speech_recognition as sr
    _SR_AVAILABLE = True
except ImportError:
    pass


def listen(timeout: int = 5, phrase_limit: int = 15) -> str | None:
    """
    Listen from the default microphone and return transcribed text, or None.

    timeout:      seconds to wait for speech to begin
    phrase_limit: max seconds of audio to capture
    """
    if not _SR_AVAILABLE:
        return None

    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.pause_threshold = 0.8

    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(
                source, timeout=timeout, phrase_time_limit=phrase_limit
            )
        return recognizer.recognize_google(audio)
    except (sr.WaitTimeoutError, sr.UnknownValueError, sr.RequestError, OSError):
        return None


def listen_loop(on_text, stop_phrase: str = "goodbye amy") -> None:
    """
    Continuously listen and call on_text(text) for each utterance.

    Stops when stop_phrase is detected (case-insensitive).
    """
    while True:
        text = listen()
        if text is None:
            continue
        if stop_phrase and stop_phrase in text.lower():
            break
        on_text(text)
