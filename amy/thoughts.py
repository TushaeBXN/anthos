"""
Amy's internal reasoning layer.

Produces structured observations from a session transcript that
the learnings layer and state layer can consume.  Nothing here
touches disk or calls external services.
"""

from __future__ import annotations

import re
from collections import Counter


_EMOTION_WORDS = {
    "joy": ["happy", "excited", "great", "wonderful", "love", "fantastic", "glad", "thrilled"],
    "sadness": ["sad", "upset", "crying", "depressed", "lonely", "miss", "lost", "grief"],
    "anxiety": ["worried", "anxious", "nervous", "scared", "afraid", "stress", "overwhelmed"],
    "anger": ["angry", "frustrated", "annoyed", "furious", "mad", "irritated"],
    "curiosity": ["wonder", "curious", "interesting", "fascinating", "tell me", "how does"],
    "neutral": [],
}


def infer_dominant_emotion(transcript: list[dict]) -> str:
    """Return the most prominent emotional register detected in user turns."""
    user_text = " ".join(
        t["content"].lower() for t in transcript if t.get("role") == "user"
    )
    scores: dict[str, int] = {e: 0 for e in _EMOTION_WORDS}
    for emotion, words in _EMOTION_WORDS.items():
        for word in words:
            scores[emotion] += len(re.findall(r"\b" + re.escape(word) + r"\b", user_text))
    scores.pop("neutral", None)
    if not any(scores.values()):
        return "neutral"
    return max(scores, key=lambda e: scores[e])


def extract_topics(transcript: list[dict], max_topics: int = 5) -> list[str]:
    """
    Heuristic topic extraction: noun phrases that appear more than once
    or are explicitly named by the user.

    Returns up to max_topics topic strings.
    """
    user_text = " ".join(
        t["content"] for t in transcript if t.get("role") == "user"
    )
    words = re.findall(r"\b[A-Za-z]{4,}\b", user_text.lower())
    stopwords = {
        "that", "this", "with", "have", "from", "they", "been", "when",
        "what", "just", "know", "will", "your", "about", "like", "more",
        "then", "into", "some", "also", "over", "only", "much", "them",
    }
    counts = Counter(w for w in words if w not in stopwords)
    return [word for word, _ in counts.most_common(max_topics)]


def extract_patterns(transcript: list[dict]) -> list[str]:
    """
    Identify surface-level behavioral or conversational patterns.

    Returns short pattern strings suitable for learnings.md entries.
    """
    patterns = []
    user_turns = [t for t in transcript if t.get("role") == "user"]
    avg_len = (
        sum(len(t["content"].split()) for t in user_turns) / len(user_turns)
        if user_turns else 0
    )

    if avg_len > 60:
        patterns.append("user communicates in extended, detailed messages")
    elif avg_len < 15:
        patterns.append("user prefers short, concise exchanges")

    questions = sum(1 for t in user_turns if "?" in t["content"])
    if questions > len(user_turns) * 0.5:
        patterns.append("user is in inquiry mode — primarily asking questions")

    if any("thank" in t["content"].lower() for t in user_turns):
        patterns.append("user expressed gratitude during session")

    return patterns


def summarize_session(transcript: list[dict]) -> dict:
    """
    Produce a structured summary dict consumed by state.py and learnings.py.

    Keys: dominant_emotion, topics, patterns, turn_count, summary_line
    """
    dominant_emotion = infer_dominant_emotion(transcript)
    topics = extract_topics(transcript)
    patterns = extract_patterns(transcript)
    turn_count = len(transcript)
    topic_str = ", ".join(topics) if topics else "general conversation"

    summary_line = (
        f"{turn_count}-turn session; topics: {topic_str}; "
        f"emotional register: {dominant_emotion}"
    )

    return {
        "dominant_emotion": dominant_emotion,
        "topics": topics,
        "patterns": patterns,
        "turn_count": turn_count,
        "summary_line": summary_line,
    }
