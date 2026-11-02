#!/usr/bin/env python3
"""
ingest_local_docs.py — Convert local docs/PDFs/JSONLs to Anthos SFT format

Processes:
  - .docx files (agent guides, blueprints)
  - .pdf files (books, guides)
  - .jsonl files in instruction format {instruction, response}
  - Converts everything to Anthos conversations format

Usage:
    python ingest_local_docs.py
    python ingest_local_docs.py --folder "/path/to/folder"
"""

import json
import re
import argparse
from pathlib import Path

OUTPUT  = Path("data/local_docs_sft.jsonl")
FOLDER  = Path("/Users/dadsmacpro/Desktop/The Ai info/Learning/training_data")

ANTHOS_SYSTEM = (
    "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
    "You have deep knowledge of AI agents, automation, financial strategy, "
    "and technology. Answer clearly and practically."
)

# ─────────────────────────────────────────────────────────────────────────────
# Text processing
# ─────────────────────────────────────────────────────────────────────────────

def clean(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def chunk_paragraphs(paras: list[str], min_chars: int = 120, max_chars: int = 1800) -> list[str]:
    chunks = []
    current = ""
    for p in paras:
        if len(current) + len(p) > max_chars:
            if current.strip() and len(current) >= min_chars:
                chunks.append(current.strip())
            current = p
        else:
            current += "\n\n" + p if current else p
    if current.strip() and len(current) >= min_chars:
        chunks.append(current.strip())
    return chunks


def make_pair(question: str, answer: str) -> dict:
    return {
        "conversations": [
            {"from": "system", "value": ANTHOS_SYSTEM},
            {"from": "human",  "value": question.strip()},
            {"from": "gpt",    "value": answer.strip()[:2000]},
        ]
    }


# ─────────────────────────────────────────────────────────────────────────────
# Docx ingestion
# ─────────────────────────────────────────────────────────────────────────────

AGENT_SYSTEM = (
    "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
    "You understand AI agents, how to deploy them, how to monetize them, "
    "and how to build agent-powered businesses. You can guide users through "
    "setting up agents using available tools, APIs, and infrastructure."
)

def ingest_docx(path: Path) -> list[dict]:
    try:
        import docx as _docx
    except ImportError:
        print("  pip install python-docx")
        return []

    doc   = _docx.Document(str(path))
    paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()
             and not p.text.strip().startswith("To make notes")]

    # Split into sections by heading (all-caps or short lines)
    sections = []
    current_heading = ""
    current_body    = []

    for p in paras:
        is_heading = (len(p) < 80 and (p.isupper() or p.endswith("?") or
                      any(p.startswith(f"{i}.") for i in range(1, 20))))
        if is_heading:
            if current_body:
                sections.append((current_heading, "\n\n".join(current_body)))
            current_heading = p
            current_body    = []
        else:
            current_body.append(p)
    if current_body:
        sections.append((current_heading, "\n\n".join(current_body)))

    pairs = []
    for heading, body in sections:
        if len(body) < 100:
            continue
        chunks = chunk_paragraphs(body.split("\n\n"))
        for chunk in chunks:
            if not chunk or len(chunk) < 80:
                continue
            q = f"How does {heading.lower()} work for AI agents?" if heading else "Explain how AI agents can generate income and be deployed practically."
            p = make_pair(q, chunk)
            p["conversations"][0]["value"] = AGENT_SYSTEM
            pairs.append(p)

    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# PDF ingestion
# ─────────────────────────────────────────────────────────────────────────────

def ingest_pdf(path: Path) -> list[dict]:
    try:
        import pdfplumber
    except ImportError:
        try:
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "pip", "install", "pdfplumber", "-q"])
            import pdfplumber
        except Exception:
            print(f"  Skipping {path.name} — install pdfplumber")
            return []

    pairs = []
    topic = path.stem.replace("_", " ").replace("-", " ")

    try:
        with pdfplumber.open(str(path)) as pdf:
            full_text = "\n\n".join(
                page.extract_text() or "" for page in pdf.pages
            )
    except Exception as e:
        print(f"  Failed reading {path.name}: {e}")
        return []

    full_text = clean(full_text)
    paras     = [p.strip() for p in full_text.split("\n\n") if len(p.strip()) > 60]
    chunks    = chunk_paragraphs(paras)

    for chunk in chunks:
        q = f"Explain this from {topic}: {chunk[:80]}..."
        pairs.append(make_pair(q, chunk))

    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# JSONL ingestion (instruction format → conversations)
# ─────────────────────────────────────────────────────────────────────────────

def ingest_jsonl(path: Path) -> list[dict]:
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Already in conversations format
            if "conversations" in item:
                pairs.append(item)
                continue

            # instruction / response format
            instruction = item.get("instruction") or item.get("prompt") or item.get("input", "")
            response    = item.get("response") or item.get("output") or item.get("completion", "")

            if instruction and response and len(response) > 40:
                pairs.append(make_pair(instruction, response))

    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=str(FOLDER))
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    folder = Path(args.folder)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not folder.exists():
        print(f"Folder not found: {folder}")
        return

    all_pairs = []

    for path in sorted(folder.iterdir()):
        if path.name.startswith("."):
            continue

        suffix = path.suffix.lower()
        print(f"  {path.name} ...", end=" ", flush=True)

        if suffix == ".docx":
            pairs = ingest_docx(path)
        elif suffix == ".pdf":
            pairs = ingest_pdf(path)
        elif suffix == ".jsonl":
            pairs = ingest_jsonl(path)
        else:
            print("skipped")
            continue

        print(f"{len(pairs)} pairs")
        all_pairs.extend(pairs)

    print(f"\n  Total: {len(all_pairs):,} pairs")

    with open(output, "w") as f:
        for p in all_pairs:
            f.write(json.dumps(p) + "\n")

    print(f"  Output: {output}")
    print(f"\nTo add to full training set:")
    print(f"  cat data/sft_all.jsonl {output} > data/sft_master.jsonl")


if __name__ == "__main__":
    main()
