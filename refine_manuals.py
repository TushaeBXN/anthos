"""
refine_manuals.py — Internet Archive PDF → Anthos ShareGPT JSONL

Downloads technical manuals from archive.org and converts them into
instruction-following training examples in the ShareGPT format that
Anthos expects.

Usage:
    pip install internetarchive pymupdf4llm
    python3 refine_manuals.py --query "cybersecurity manual" --limit 30
    python3 refine_manuals.py --query "networking firewall" --limit 20
    python3 refine_manuals.py --query "electronics repair manual" --limit 20

Output appends to data/teacher_conversations.jsonl (same file Anthos trains on).
"""

import os
import re
import json
import argparse
import subprocess
from pathlib import Path


# ── Instruction templates (varied so model doesn't memorize the wrapper) ──────
INSTRUCTION_TEMPLATES = [
    "Summarize the key technical points from this manual section.",
    "Explain what this section of the manual is describing.",
    "What does this documentation say? Give a clear explanation.",
    "Break down the technical content from this manual excerpt.",
    "What are the important details in this technical passage?",
    "Explain this documentation in plain terms.",
    "What is this manual section about?",
    "Describe what this technical document is covering.",
]


def clean_text(text: str) -> str:
    """Remove noise common in PDF extractions."""
    # Collapse excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    # Remove page headers/footers (short lines with only numbers/dashes)
    lines = [l for l in text.split('\n') if not re.match(r'^\s*[\d\-–—]+\s*$', l)]
    text = '\n'.join(lines).strip()
    return text


def chunk_text(text: str, max_chars: int = 600) -> list[str]:
    """
    Split text into chunks at paragraph or sentence boundaries.
    600 chars ≈ 150 tokens — fits comfortably in seq_len=256.
    """
    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 40]
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) < max_chars:
            current += (" " if current else "") + para
        else:
            if current:
                chunks.append(current)
            # If single paragraph is too long, split by sentence
            if len(para) > max_chars:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                sub = ""
                for s in sentences:
                    if len(sub) + len(s) < max_chars:
                        sub += (" " if sub else "") + s
                    else:
                        if sub:
                            chunks.append(sub)
                        sub = s
                if sub:
                    chunks.append(sub)
                current = ""
            else:
                current = para

    if current:
        chunks.append(current)

    return [c for c in chunks if len(c) > 80]  # skip tiny fragments


def pdf_to_chunks(pdf_path: Path) -> list[str]:
    """Extract and chunk text from a PDF using pymupdf4llm."""
    try:
        import pymupdf4llm
        md_text = pymupdf4llm.to_markdown(str(pdf_path))
        cleaned = clean_text(md_text)
        return chunk_text(cleaned)
    except ImportError:
        print("  [!] pymupdf4llm not installed. Run: pip install pymupdf4llm")
        return []
    except Exception as e:
        print(f"  [!] Failed to parse {pdf_path.name}: {e}")
        return []


def chunks_to_sharegpt(chunks: list[str], source_name: str) -> list[dict]:
    """Convert text chunks to ShareGPT format for Anthos."""
    import random
    records = []
    templates = INSTRUCTION_TEMPLATES.copy()
    random.shuffle(templates)

    for i, chunk in enumerate(chunks):
        template = templates[i % len(templates)]
        records.append({
            "conversations": [
                {
                    "from": "system",
                    "value": "You are Anthos, a helpful and knowledgeable assistant."
                },
                {
                    "from": "human",
                    "value": template
                },
                {
                    "from": "gpt",
                    "value": chunk
                }
            ],
            "source": source_name
        })

    return records


def download_ia(query: str, destdir: Path, limit: int = 30) -> list[Path]:
    """Download PDFs from Internet Archive using the ia CLI."""
    destdir.mkdir(parents=True, exist_ok=True)

    print(f"\n[ia] Searching: '{query}' (limit {limit})")
    cmd = [
        "ia", "download",
        "--search", f"mediatype:texts AND ({query})",
        "--glob", "*.pdf",
        "--destdir", str(destdir),
        "--no-directories",
    ]

    # ia doesn't have a native --limit for download, so we search first
    search_cmd = [
        "ia", "search",
        f"mediatype:texts AND ({query})",
        "--itemlist",
        "-n", str(limit),
    ]

    try:
        result = subprocess.run(search_cmd, capture_output=True, text=True, timeout=30)
        identifiers = [l.strip() for l in result.stdout.splitlines() if l.strip()][:limit]
        print(f"[ia] Found {len(identifiers)} items")

        downloaded = []
        for item_id in identifiers:
            print(f"[ia] Downloading: {item_id}")
            dl_cmd = ["ia", "download", item_id, "--glob", "*.pdf",
                      "--destdir", str(destdir), "--no-directories", "--quiet"]
            subprocess.run(dl_cmd, timeout=120)
            pdfs = list(destdir.glob(f"*.pdf"))
            downloaded = pdfs

        return list(destdir.glob("*.pdf"))

    except FileNotFoundError:
        print("[!] 'ia' command not found. Install with: pip install internetarchive")
        return []
    except subprocess.TimeoutExpired:
        print("[!] Download timed out — returning whatever was grabbed so far")
        return list(destdir.glob("*.pdf"))


def main():
    parser = argparse.ArgumentParser(description="Internet Archive → Anthos training data")
    parser.add_argument("--query",   type=str, default="cybersecurity manual",
                        help="Search query for archive.org")
    parser.add_argument("--limit",   type=int, default=30,
                        help="Max number of items to download")
    parser.add_argument("--destdir", type=str, default="data/raw_manuals",
                        help="Where to store downloaded PDFs")
    parser.add_argument("--out",     type=str, default="data/teacher_conversations.jsonl",
                        help="Output JSONL file (appended to)")
    parser.add_argument("--local",   type=str, default=None,
                        help="Skip download, process PDFs already in this folder")
    args = parser.parse_args()

    dest   = Path(args.destdir)
    out    = Path(args.out)

    # ── Download or use local ─────────────────────────────────────────────────
    if args.local:
        pdf_files = list(Path(args.local).glob("*.pdf"))
        print(f"[refinery] Using {len(pdf_files)} local PDFs from {args.local}")
    else:
        pdf_files = download_ia(args.query, dest, args.limit)

    if not pdf_files:
        print("[!] No PDFs found. Check your query or install the ia CLI.")
        return

    # ── Process PDFs ──────────────────────────────────────────────────────────
    all_records = []
    for pdf in pdf_files:
        print(f"[refinery] Processing {pdf.name}...")
        chunks  = pdf_to_chunks(pdf)
        records = chunks_to_sharegpt(chunks, source_name=pdf.stem)
        all_records.extend(records)
        print(f"  → {len(chunks)} chunks")

    # ── Write output ──────────────────────────────────────────────────────────
    import random
    random.shuffle(all_records)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    print(f"\n── Refinery complete ───────────────────────────────")
    print(f"  PDFs processed:  {len(pdf_files)}")
    print(f"  Chunks created:  {len(all_records)}")
    print(f"  Appended to:     {out}")
    print(f"────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
