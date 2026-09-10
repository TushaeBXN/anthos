"""
anthos/multilingual.py — Language detection and routing for Anthos

Strategy:
  1. Detect the language of the incoming message
  2. If non-English, note the language and ask Anthos to respond in that language
  3. Inject a language directive into the system prompt

No translation API needed — Anthos responds in the detected language
by being told to via the system prompt. Works for any language the
training data touched (English-primary, but system prompt instruction
is language-agnostic for capable models).
"""

from __future__ import annotations

LANGUAGE_NAMES = {
    "af": "Afrikaans", "ar": "Arabic", "bg": "Bulgarian", "bn": "Bengali",
    "ca": "Catalan", "cs": "Czech", "cy": "Welsh", "da": "Danish",
    "de": "German", "el": "Greek", "en": "English", "es": "Spanish",
    "et": "Estonian", "fa": "Persian", "fi": "Finnish", "fr": "French",
    "gu": "Gujarati", "he": "Hebrew", "hi": "Hindi", "hr": "Croatian",
    "hu": "Hungarian", "id": "Indonesian", "it": "Italian", "ja": "Japanese",
    "kn": "Kannada", "ko": "Korean", "lt": "Lithuanian", "lv": "Latvian",
    "mk": "Macedonian", "ml": "Malayalam", "mr": "Marathi", "ne": "Nepali",
    "nl": "Dutch", "no": "Norwegian", "pa": "Punjabi", "pl": "Polish",
    "pt": "Portuguese", "ro": "Romanian", "ru": "Russian", "sk": "Slovak",
    "sl": "Slovenian", "so": "Somali", "sq": "Albanian", "sr": "Serbian",
    "sv": "Swedish", "sw": "Swahili", "ta": "Tamil", "te": "Telugu",
    "th": "Thai", "tl": "Filipino", "tr": "Turkish", "uk": "Ukrainian",
    "ur": "Urdu", "vi": "Vietnamese", "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
}


def detect_language(text: str) -> str:
    """Returns ISO language code, defaults to 'en' on failure."""
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        return "en"


def language_directive(lang_code: str) -> str:
    """
    Returns a system prompt addition telling Anthos to respond in
    the detected language. Empty string for English.
    """
    if lang_code == "en" or lang_code.startswith("en"):
        return ""
    lang_name = LANGUAGE_NAMES.get(lang_code, lang_code.upper())
    return (
        f"\n\nIMPORTANT: The user is communicating in {lang_name}. "
        f"You MUST respond entirely in {lang_name}. "
        f"Do not switch to English."
    )


def is_english(text: str) -> bool:
    return detect_language(text) == "en"
