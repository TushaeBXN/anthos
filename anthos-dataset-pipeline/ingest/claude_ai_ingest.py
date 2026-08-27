"""
ingest/claude_ai_ingest.py
Parses the conversations.json from a Claude.ai data export
(Settings → Account → Export Data) into ShareGPT JSONL.

Each multi-turn conversation becomes one ShareGPT example.
Only conversations that contain at least one code block are kept.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CLAUDE_EXPORT_PATH, CLAUDE_AI_RAW_OUT, ensure_output_dirs

CODE_FENCE = "```"


def has_code(text):
    return CODE_FENCE in (text or "")


def extract_text(content):
    """
    Claude export content can be a string or a list of content blocks.
    Returns plain text.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "tool_result":
                    # skip tool results — they're context, not training signal
                    pass
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return ""


def conversation_to_sharegpt(conv):
    """
    Returns a ShareGPT dict or None if the conversation has no code.
    """
    messages = conv.get("chat_messages", [])
    turns = []
    has_any_code = False

    for msg in messages:
        role = msg.get("sender", "")
        raw_content = msg.get("content", "")
        text = extract_text(raw_content).strip()

        if not text:
            continue

        if role == "human":
            turns.append({"from": "human", "value": text})
        elif role == "assistant":
            if has_code(text):
                has_any_code = True
            turns.append({"from": "gpt", "value": text})

    if not has_any_code or len(turns) < 2:
        return None

    # Must start with human
    if turns[0]["from"] != "human":
        return None

    return {
        "conversations": turns,
        "source": "claude_ai",
        "conversation_id": conv.get("uuid", ""),
        "title": conv.get("name", ""),
    }


def run():
    ensure_output_dirs()
    export_path = Path(CLAUDE_EXPORT_PATH)

    if not export_path.exists():
        print(f"✗ Export file not found: {export_path}")
        print("  Go to Claude.ai → Settings → Account → Export Data")
        print("  Then set CLAUDE_EXPORT_PATH env var to the conversations.json path")
        return

    with export_path.open() as f:
        data = json.load(f)

    # Export format is either a list of conversations or {"conversations": [...]}
    if isinstance(data, list):
        conversations = data
    elif isinstance(data, dict):
        conversations = data.get("conversations", [])
    else:
        print("✗ Unexpected export format")
        return

    count = 0
    skipped = 0

    with CLAUDE_AI_RAW_OUT.open("w") as out:
        for conv in conversations:
            example = conversation_to_sharegpt(conv)
            if example is None:
                skipped += 1
                continue
            out.write(json.dumps(example) + "\n")
            count += 1

    print(f"✓ Claude.ai ingest complete — {count} kept, {skipped} skipped → {CLAUDE_AI_RAW_OUT}")


if __name__ == "__main__":
    run()
