#!/usr/bin/env python3
"""
ingest_ai_info_folder.py — Ingest new content from The Ai info folder into Anthos SFT data

Targets (not overlapping with already-ingested training_data/ folder):
  1. Amy learning/Topics for Amy.md  → AI concept Q&As (prompting, RAG, tool use, MLOps)
  2. Anthos Stuff/anthos-intelligence-knowledge-base.md → resource/identity Q&As
  3. Learning/Code books/*.pdf       → Python, ML, networking knowledge (text-layer PDFs only)

Output: data/ai_info_sft.jsonl

Usage:
    python ingest_ai_info_folder.py
    python ingest_ai_info_folder.py --dry-run
"""

import json
import re
import argparse
from pathlib import Path

BASE    = Path("/Users/dadsmacpro/Desktop/The Ai info")
OUTPUT  = Path("data/ai_info_sft.jsonl")

MIN_PDF_CHARS = 2000   # skip PDFs with less than this — probably scanned
MAX_CHUNK     = 1800
MIN_CHUNK     = 100

ANTHOS_SYSTEM = (
    "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
    "You are knowledgeable about AI, machine learning, software engineering, "
    "Python, and building intelligent systems. Answer clearly and practically."
)

ANTHOS_AI_SYSTEM = (
    "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
    "You understand advanced AI concepts: RAG, function calling, fine-tuning, "
    "prompt engineering, agent design, and production ML systems. "
    "Explain clearly with practical examples."
)

# ─────────────────────────────────────────────────────────────────────────────
# Text utils
# ─────────────────────────────────────────────────────────────────────────────

def clean(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def clean_markdown(text: str) -> str:
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def chunk_text(text: str, max_chars: int = MAX_CHUNK, min_chars: int = MIN_CHUNK) -> list[str]:
    paras = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    current = ""
    for p in paras:
        if len(current) + len(p) + 2 > max_chars:
            if current and len(current) >= min_chars:
                chunks.append(current.strip())
            current = p
        else:
            current = current + "\n\n" + p if current else p
    if current and len(current) >= min_chars:
        chunks.append(current.strip())
    return chunks


def make_pair(question: str, answer: str, system: str = ANTHOS_SYSTEM) -> dict:
    return {
        "conversations": [
            {"from": "system", "value": system},
            {"from": "human",  "value": question.strip()},
            {"from": "gpt",    "value": answer.strip()[:2500]},
        ]
    }


# ─────────────────────────────────────────────────────────────────────────────
# Source 1: Topics for Amy.md → AI concept Q&As
# ─────────────────────────────────────────────────────────────────────────────

def ingest_amy_topics() -> list[dict]:
    path = BASE / "Amy learning" / "Topics for Amy.md"
    if not path.exists():
        print(f"  Not found: {path}")
        return []

    text  = clean_markdown(path.read_text(errors="ignore"))
    lines = text.splitlines()

    # Split by "Priority" sections and "Week" sections
    sections = []
    current  = []
    for line in lines:
        if re.match(r'(Priority \d|Week \d|#### Week)', line):
            if current:
                sections.append('\n'.join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append('\n'.join(current))

    pairs = []
    for section in sections:
        section = section.strip()
        if len(section) < MIN_CHUNK:
            continue
        lines_s = section.splitlines()
        heading = lines_s[0].strip()
        body    = '\n'.join(lines_s[1:]).strip()
        if not body or len(body) < 80:
            continue
        for chunk in chunk_text(body):
            if len(chunk) < 80:
                continue
            q = f"Explain {heading.lower().replace('*', '').strip()} in the context of AI development." if heading else "What are key AI concepts an assistant should master?"
            pairs.append(make_pair(q, chunk, ANTHOS_AI_SYSTEM))

    print(f"  Topics for Amy.md → {len(pairs)} pairs")
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Source 2: anthos-intelligence-knowledge-base.md → Q&As about AI resources
# ─────────────────────────────────────────────────────────────────────────────

KB_ANTHOS_SYSTEM = (
    "You are Anthos, an AI assistant created by Brian Tushae Thomas at Anthos Intelligence. "
    "You know the full landscape of free AI, ML, and programming learning resources, "
    "and can recommend the right resource for any learning goal."
)

def ingest_knowledge_base() -> list[dict]:
    path = BASE / "Anthos Stuff" / "Anthos Intelligence — Think in Streams_files" / "anthos-intelligence-knowledge-base.md"
    if not path.exists():
        print(f"  Not found: {path}")
        return []

    text = path.read_text(errors="ignore")

    # Split by ## sections
    raw_sections = re.split(r'\n## ', text)

    pairs = []
    for section in raw_sections[1:]:   # skip preamble
        lines  = section.strip().splitlines()
        heading = lines[0].strip().lstrip('#').strip()
        body    = '\n'.join(lines[1:]).strip()
        if not body or len(body) < 100:
            continue
        # Extract resources as bullet lines
        resource_lines = [l.strip() for l in body.splitlines()
                          if l.strip().startswith('-') and len(l.strip()) > 40]
        if not resource_lines:
            continue
        resource_text = '\n'.join(resource_lines[:20])
        q = f"What are the best free resources to learn {heading}?"
        pairs.append(make_pair(q, resource_text, KB_ANTHOS_SYSTEM))

        # Also create a "recommend for goal" pair
        if len(resource_lines) >= 3:
            goal_q = f"I want to understand {heading} deeply. What should I read first?"
            pairs.append(make_pair(goal_q, resource_lines[0].lstrip('- '), KB_ANTHOS_SYSTEM))

    print(f"  anthos-intelligence-knowledge-base.md → {len(pairs)} pairs")
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Source 3: Learning/Code books/*.pdf → Python/ML/networking knowledge
# ─────────────────────────────────────────────────────────────────────────────

def try_extract_pdf(path: Path) -> str:
    try:
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf:
            pages = pdf.pages[:80]  # cap at 80 pages to keep runtime reasonable
            text  = "\n\n".join(page.extract_text() or "" for page in pages)
        return text
    except Exception:
        return ""


def ingest_code_books(dry_run: bool = False) -> list[dict]:
    books_dir = BASE / "Learning" / "Code books"
    if not books_dir.exists():
        print(f"  Not found: {books_dir}")
        return []

    all_pairs = []
    pdf_files = list(books_dir.glob("*.pdf"))

    for pdf in sorted(pdf_files):
        raw = try_extract_pdf(pdf)
        if len(raw) < MIN_PDF_CHARS:
            print(f"  SKIP (scanned/empty) {pdf.name}")
            continue

        text  = clean(raw)
        topic = pdf.stem.replace("Copy of ", "").replace("_", " ").strip()
        pairs = []

        for chunk in chunk_text(text, max_chars=2000):
            if len(chunk) < 150:
                continue
            q = f"Explain this Python/programming concept: {chunk[:80].strip()}..."
            pairs.append(make_pair(q, chunk, ANTHOS_SYSTEM))

        print(f"  {pdf.name[:50]:50s} → {len(pairs)} pairs")
        all_pairs.extend(pairs)

    print(f"  Code books total → {len(all_pairs)} pairs from {len(pdf_files)} PDFs")
    return all_pairs


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output",  default=str(OUTPUT))
    parser.add_argument("--skip-books", action="store_true", help="Skip slow PDF extraction")
    args = parser.parse_args()

    print("=== Anthos Intelligence — AI Info Ingestion ===\n")

    all_pairs = []

    print("[1/3] Topics for Amy.md")
    all_pairs.extend(ingest_amy_topics())

    print("\n[2/3] Anthos Intelligence Knowledge Base")
    all_pairs.extend(ingest_knowledge_base())

    if not args.skip_books:
        print("\n[3/3] Code Books PDFs")
        all_pairs.extend(ingest_code_books(dry_run=args.dry_run))
    else:
        print("\n[3/3] Code Books — skipped (--skip-books)")

    print(f"\n{'─'*50}")
    print(f"Total new pairs: {len(all_pairs):,}")

    if args.dry_run:
        print("DRY RUN — nothing written")
        return

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for p in all_pairs:
            f.write(json.dumps(p) + "\n")

    print(f"Output: {output} ({len(all_pairs):,} lines)")
    print(f"\nTo merge into master training set:")
    print(f"  cat data/sft_master.jsonl {output} > /tmp/sft_new.jsonl && mv /tmp/sft_new.jsonl data/sft_master.jsonl")


if __name__ == "__main__":
    main()
