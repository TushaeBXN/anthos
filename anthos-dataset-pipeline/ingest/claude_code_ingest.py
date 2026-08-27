"""
ingest/claude_code_ingest.py
Reads Claude Code session logs from ~/.claude/projects/
and converts them to ShareGPT JSONL.

Claude Code stores sessions as JSONL files under:
  ~/.claude/projects/<project-hash>/<session-id>.jsonl

Each line in a session file is a message event.
We reconstruct human/assistant turns and keep sessions with code.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CLAUDE_CODE_PROJECTS_DIR, CLAUDE_CODE_RAW_OUT, ensure_output_dirs

CODE_FENCE = "```"


def has_code(text):
    return CODE_FENCE in (text or "")


def extract_text_from_content(content):
    """Extract plain text from Claude Code message content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type", "")
                if btype == "text":
                    parts.append(block.get("text", ""))
                elif btype == "tool_result":
                    # Include tool results as context in the human turn
                    for sub in block.get("content", []):
                        if isinstance(sub, dict) and sub.get("type") == "text":
                            parts.append(f"[Tool result]: {sub.get('text', '')}")
        return "\n".join(p for p in parts if p.strip())
    return ""


def parse_session_file(session_path):
    """
    Parse a single .jsonl session file into a list of (role, text) tuples.
    Returns None if the session has no code or fewer than 2 turns.
    """
    turns = []
    has_any_code = False

    try:
        with session_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                role = event.get("role", "")
                content = event.get("content", "")

                # Some formats wrap in a "message" key
                if not role and "message" in event:
                    msg = event["message"]
                    role = msg.get("role", "")
                    content = msg.get("content", "")

                if role not in ("user", "assistant"):
                    continue

                text = extract_text_from_content(content).strip()
                if not text:
                    continue

                mapped_role = "human" if role == "user" else "gpt"

                if mapped_role == "gpt" and has_code(text):
                    has_any_code = True

                turns.append({"from": mapped_role, "value": text})

    except Exception as e:
        print(f"    Error reading {session_path.name}: {e}")
        return None

    if not has_any_code or len(turns) < 2:
        return None

    if turns[0]["from"] != "human":
        return None

    return turns


def run():
    ensure_output_dirs()

    if not CLAUDE_CODE_PROJECTS_DIR.exists():
        print(f"✗ Claude Code projects dir not found: {CLAUDE_CODE_PROJECTS_DIR}")
        print("  Make sure Claude Code has been used on this machine.")
        return

    project_dirs = [p for p in CLAUDE_CODE_PROJECTS_DIR.iterdir() if p.is_dir()]
    print(f"  Found {len(project_dirs)} Claude Code project folders")

    count = 0
    skipped = 0

    with CLAUDE_CODE_RAW_OUT.open("w") as out:
        for project_dir in sorted(project_dirs):
            session_files = list(project_dir.glob("*.jsonl"))
            if not session_files:
                continue

            print(f"  {project_dir.name} — {len(session_files)} session(s)")

            for session_file in sorted(session_files):
                turns = parse_session_file(session_file)
                if turns is None:
                    skipped += 1
                    continue

                example = {
                    "conversations": turns,
                    "source": "claude_code",
                    "project": project_dir.name,
                    "session": session_file.stem,
                }
                out.write(json.dumps(example) + "\n")
                count += 1

    print(f"✓ Claude Code ingest complete — {count} kept, {skipped} skipped → {CLAUDE_CODE_RAW_OUT}")


if __name__ == "__main__":
    run()
