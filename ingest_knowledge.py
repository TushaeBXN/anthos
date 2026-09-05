#!/usr/bin/env python3
"""
ingest_knowledge.py — Convert GitHub repos and web articles to Anthos SFT training data

Clones repos, extracts markdown/text, converts to instruction-response pairs,
appends to data/knowledge_sft.jsonl.

Usage:
    python ingest_knowledge.py
    python ingest_knowledge.py --dry-run     # count files without writing
    python ingest_knowledge.py --append      # append to existing file
"""

import json
import re
import subprocess
import tempfile
import argparse
import urllib.request
from pathlib import Path

OUTPUT = Path("data/knowledge_sft.jsonl")

ANTHOS_SYSTEM = (
    "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
    "You are knowledgeable about AI, machine learning, software engineering, "
    "security research, and system design. Answer clearly and directly."
)

# ─────────────────────────────────────────────────────────────────────────────
# Knowledge sources
# ─────────────────────────────────────────────────────────────────────────────

GITHUB_REPOS = [
    {
        "url":   "https://github.com/huggingface/smol-course",
        "topic": "training small language models, fine-tuning, SFT, RLHF, alignment",
    },
    {
        "url":   "https://github.com/Sumanth077/Hands-On-AI-Engineering",
        "topic": "AI engineering, LLM applications, agent design, production AI systems",
    },
    {
        "url":   "https://github.com/humanlayer/humanlayer",
        "topic": "human-in-the-loop AI, agent approval flows, AI safety controls",
    },
    {
        "url":   "https://github.com/humanlayer/advanced-context-engineering-for-coding-agents",
        "topic": "context engineering, coding agents, prompt architecture",
    },
    {
        "url":   "https://github.com/systemdesign42/system-design-academy",
        "topic": "system design, distributed systems, scalability, architecture",
    },
    {
        "url":   "https://github.com/liquidslr/system-design-notes",
        "topic": "system design, databases, caching, load balancing, microservices",
    },
    {
        "url":   "https://github.com/rapid7/metasploit-framework",
        "topic": "penetration testing, security research, vulnerability assessment",
        "docs_only": True,   # only extract docs/ and README — skip exploit code
    },
]

RAW_URLS = [
    {
        "url":   "https://raw.githubusercontent.com/humanlayer/advanced-context-engineering-for-coding-agents/main/ace-fca.md",
        "title": "Advanced Context Engineering for Coding Agents",
        "topic": "context engineering, coding agents, FCA pattern",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Markdown → instruction pairs
# ─────────────────────────────────────────────────────────────────────────────

def clean_markdown(text: str) -> str:
    text = re.sub(r'```[\s\S]*?```', lambda m: m.group(0), text)  # keep code blocks
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)                    # remove images
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)         # flatten links
    text = re.sub(r'\n{3,}', '\n\n', text)                        # collapse blanks
    return text.strip()


def split_sections(text: str, min_chars: int = 150, max_chars: int = 2500) -> list[str]:
    sections = re.split(r'\n#{1,3} ', text)
    out = []
    for s in sections:
        s = s.strip()
        if len(s) < min_chars:
            continue
        if len(s) > max_chars:
            # chunk by paragraph
            paras = s.split('\n\n')
            chunk = ""
            for p in paras:
                if len(chunk) + len(p) > max_chars:
                    if chunk:
                        out.append(chunk.strip())
                    chunk = p
                else:
                    chunk += "\n\n" + p
            if chunk.strip():
                out.append(chunk.strip())
        else:
            out.append(s)
    return out


def make_pairs(sections: list[str], topic: str, source: str) -> list[dict]:
    pairs = []
    for section in sections:
        lines = section.strip().splitlines()
        if not lines:
            continue

        heading = lines[0].strip().lstrip('#').strip()
        body    = '\n'.join(lines[1:]).strip() if len(lines) > 1 else section

        if not body or len(body) < 80:
            continue

        question = f"Explain {heading} in the context of {topic}." if heading else f"What does this cover about {topic}?"

        pairs.append({
            "conversations": [
                {"from": "system", "value": ANTHOS_SYSTEM},
                {"from": "human",  "value": question},
                {"from": "gpt",    "value": body[:2000]},
            ]
        })

    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# GitHub repo ingestion
# ─────────────────────────────────────────────────────────────────────────────

def ingest_repo(repo: dict, tmpdir: str, dry_run: bool) -> int:
    url      = repo["url"]
    topic    = repo["topic"]
    docs_only = repo.get("docs_only", False)
    name     = url.rstrip("/").split("/")[-1]
    dest     = Path(tmpdir) / name

    print(f"\n  Cloning {url} ...")
    result = subprocess.run(
        ["git", "clone", "--depth=1", "--quiet", url, str(dest)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"    FAILED: {result.stderr.strip()}")
        return 0

    # Find markdown files
    if docs_only:
        md_files = list(dest.glob("docs/**/*.md")) + list(dest.glob("README*"))
    else:
        md_files = list(dest.rglob("*.md")) + list(dest.rglob("*.txt"))

    pairs = []
    for md in md_files:
        try:
            text = md.read_text(errors="ignore")
            text = clean_markdown(text)
            sections = split_sections(text)
            pairs.extend(make_pairs(sections, topic, url))
        except Exception:
            continue

    print(f"    {len(md_files)} files → {len(pairs)} pairs")

    if not dry_run:
        with open(OUTPUT, "a") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")

    return len(pairs)


def ingest_url(source: dict, dry_run: bool) -> int:
    url   = source["url"]
    topic = source["topic"]
    title = source.get("title", url)

    print(f"\n  Fetching {url} ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"    FAILED: {e}")
        return 0

    text     = clean_markdown(text)
    sections = split_sections(text)
    pairs    = make_pairs(sections, topic, url)

    print(f"    {len(sections)} sections → {len(pairs)} pairs")

    if not dry_run:
        with open(OUTPUT, "a") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")

    return len(pairs)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Count only, don't write")
    parser.add_argument("--append",  action="store_true", help="Append to existing file")
    args = parser.parse_args()

    if not args.dry_run and not args.append and OUTPUT.exists():
        print(f"Output {OUTPUT} already exists. Use --append to add to it.")
        return

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    total = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        print("=== GitHub Repos ===")
        for repo in GITHUB_REPOS:
            total += ingest_repo(repo, tmpdir, args.dry_run)

    print("\n=== Raw URLs ===")
    for source in RAW_URLS:
        total += ingest_url(source, args.dry_run)

    print(f"\n{'─'*50}")
    print(f"Total pairs: {total:,}")
    if not args.dry_run:
        lines = sum(1 for _ in open(OUTPUT)) if OUTPUT.exists() else 0
        print(f"Output file: {OUTPUT} ({lines:,} lines)")
        print(f"\nTo include in SFT training:")
        print(f"  cat data/sft_combined.jsonl data/knowledge_sft.jsonl > data/sft_all.jsonl")


if __name__ == "__main__":
    main()
