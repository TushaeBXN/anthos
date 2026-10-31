"""
download_stories.py — Global African Storybook downloader + processor
──────────────────────────────────────────────────────────────────────
Downloads all English stories from global-asp/asp-source, cleans the
Markdown, strips metadata, and saves plain text to data/ethnic_stories.txt.

Run once:
    python3 download_stories.py
"""

import re
import ssl
import time
import json
import urllib.request
from pathlib import Path

# macOS Python 3.12 ships without bundled certs — bypass verification for GitHub
_ctx = ssl.create_default_context()
_ctx.check_hostname = False
_ctx.verify_mode = ssl.CERT_NONE

RAW_DIR  = Path("raw_stories")
OUT_FILE = Path("data/ethnic_stories.txt")
API_URL  = "https://api.github.com/repos/global-asp/asp-source/contents/en"
RAW_BASE = "https://raw.githubusercontent.com/global-asp/asp-source/master/en"

RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)


# ── Step 1: fetch file listing ────────────────────────────────────────────────

print("Fetching story list from GitHub API...")
req = urllib.request.Request(API_URL, headers={"User-Agent": "anthos-downloader"})
with urllib.request.urlopen(req, context=_ctx) as r:
    files = json.load(r)

story_files = [
    f["name"] for f in files
    if f["name"].endswith(".md") and f["name"] != "README.md"
]
print(f"Found {len(story_files)} English stories.\n")


# ── Step 2: download raw .md files ───────────────────────────────────────────

def download(filename: str) -> str:
    path = RAW_DIR / filename
    if path.exists():
        return path.read_text(encoding="utf-8")
    url = f"{RAW_BASE}/{filename}"
    req = urllib.request.Request(url, headers={"User-Agent": "anthos-downloader"})
    with urllib.request.urlopen(req, context=_ctx) as r:
        text = r.read().decode("utf-8")
    path.write_text(text, encoding="utf-8")
    return text


# ── Step 3: clean Markdown → plain text ──────────────────────────────────────

# Metadata block at end of each story looks like:
#   * License: ...
#   * Author: ...
#   * Illustrator: ...
#   * Language: ...
#   * Level: ...
METADATA_RE = re.compile(
    r"\n[\*\-]\s*(License|Author|Illustrator|Language|Level|Translator"
    r"|Acknowledgement|Source|Funded|Original)[^\n]*",
    re.IGNORECASE,
)

def clean(raw: str) -> str | None:
    lines = raw.splitlines()
    cleaned = []
    for line in lines:
        # Skip H1 title lines (# Title) — keep as plain text without #
        if line.startswith("# "):
            cleaned.append(line[2:].strip())
        # Skip page dividers (## or ---)
        elif line.startswith("##") or line.strip() == "---":
            if cleaned and cleaned[-1] != "":
                cleaned.append("")   # blank line between pages
        # Skip metadata lines (* License: ...)
        elif re.match(r"^\*\s*(License|Author|Illustrator|Language|Level|"
                      r"Translator|Acknowledgement|Source|Funded|Original)",
                      line, re.IGNORECASE):
            continue
        else:
            stripped = line.strip()
            cleaned.append(stripped)

    # Join and collapse excessive blank lines
    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # Skip if too short (< 50 chars) — likely empty/stub
    if len(text) < 50:
        return None
    return text


# ── Step 4: download, clean, and write ───────────────────────────────────────

stories = []
failed  = []

for i, fname in enumerate(story_files, 1):
    try:
        raw  = download(fname)
        text = clean(raw)
        if text:
            stories.append(text)
        if i % 25 == 0 or i == len(story_files):
            print(f"  [{i}/{len(story_files)}] {len(stories)} stories processed...")
        time.sleep(0.05)   # be polite to GitHub
    except Exception as e:
        failed.append(fname)
        print(f"  [!] Failed: {fname} — {e}")

# Write final dataset: stories separated by double newline
OUT_FILE.write_text("\n\n".join(stories), encoding="utf-8")

total_chars  = OUT_FILE.stat().st_size
total_words  = sum(len(s.split()) for s in stories)

print(f"""
✓ ethnic_stories.txt ready
  Stories  : {len(stories)}
  Words    : {total_words:,}
  Size     : {total_chars / 1024:.1f} KB
  Path     : {OUT_FILE}
""")

if failed:
    print(f"  [!] {len(failed)} files failed: {failed}")
