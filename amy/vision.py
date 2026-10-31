"""
Vision input for Amy.

Captures or loads an image and returns a text description.
Uses OpenCV for camera capture and PIL for image loading.
Actual vision understanding is delegated to the LLM via base64 encoding.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path


def capture_frame() -> bytes | None:
    """Capture a single frame from the default camera. Returns raw JPEG bytes or None."""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return None
        _, buf = cv2.imencode(".jpg", frame)
        return buf.tobytes()
    except Exception:
        return None


def load_image(path: str) -> bytes | None:
    """Load an image from disk and return raw bytes, or None on failure."""
    try:
        return Path(path).read_bytes()
    except OSError:
        return None


def to_base64(image_bytes: bytes) -> str:
    """Encode image bytes as a base64 string for embedding in API payloads."""
    return base64.b64encode(image_bytes).decode("utf-8")


def describe_image_prompt(base64_image: str, mime: str = "image/jpeg") -> dict:
    """
    Build the vision content block used in a multimodal LLM message.

    Returns a dict that can be appended to a messages list.
    """
    return {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime,
                    "data": base64_image,
                },
            },
            {"type": "text", "text": "Describe what you see in this image briefly."},
        ],
    }
