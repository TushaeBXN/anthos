"""
PDF → Anthos Training Data
Converts any PDF into training material for Anthos (native arch or Qwen LoRA).

Supports chunked/resumable runs for Google Colab (time-limited) and RunPod.
Automatically removes duplicate and near-duplicate passages so every training
sample contains unique information.

Usage:
    python pdf_to_training.py input.pdf                        # full run
    python pdf_to_training.py input.pdf --chunks 0-49         # pages 0–49 only
    python pdf_to_training.py input.pdf --chunks 50-99        # pages 50–99
    python pdf_to_training.py input.pdf --merge-outputs        # combine all chunk outputs

Output formats (use --format to pick):
    instruct  — {"instruction": ..., "response": ...} JSONL  (default)
    chat      — {"pos": ..., "neg": ...} JSONL  (like persona_pairs.json)
    raw       — plain text for PackedTextDataset / history tier
    all       — writes all three

Deduplication (always on):
    Exact duplicates are removed via SHA-256 hash.
    Near-duplicates (paraphrases, repeated boilerplate) are removed via
    Jaccard similarity on word trigrams — threshold tunable with --sim-threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import textwrap
import time
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PDF EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_pages(pdf_path: str) -> list[dict]:
    """Return list of {page_num, text} dicts."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        sys.exit("PyMuPDF not installed. Run: pip install pymupdf")

    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            pages.append({"page_num": i, "text": text})
    doc.close()
    return pages


# ─────────────────────────────────────────────────────────────────────────────
# DEDUPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace — used for fingerprinting only."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _trigrams(text: str) -> set[tuple[str, ...]]:
    words = _normalize(text).split()
    if len(words) < 3:
        return {tuple(words)}
    return {tuple(words[i:i+3]) for i in range(len(words) - 2)}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class Deduplicator:
    """
    Two-pass deduplication:
      1. Exact — SHA-256 hash of normalized text (O(1) lookup).
      2. Near  — Jaccard similarity on word trigrams against a sliding window
                 of recent passages (avoids O(n²) over the whole corpus).

    The sliding window keeps memory bounded regardless of PDF size.
    Cross-chunk deduplication works because seen hashes are persisted in the
    progress file and reloaded on resume.
    """

    def __init__(
        self,
        sim_threshold: float = 0.7,
        window_size: int = 500,
        seen_hashes: set[str] | None = None,
    ):
        self.sim_threshold = sim_threshold
        self.window_size   = window_size
        # exact-dup fingerprints — persisted across chunks
        self._hashes: set[str] = seen_hashes or set()
        # recent trigram sets for near-dup window
        self._window: list[set] = []

    def is_duplicate(self, text: str) -> tuple[bool, str]:
        """Return (is_dup, reason). Registers non-dup so future calls can compare."""
        norm = _normalize(text)

        # 1. Exact duplicate
        h = _sha256(norm)
        if h in self._hashes:
            return True, "exact"

        # 2. Near-duplicate via Jaccard on trigrams
        tgrams = _trigrams(norm)
        for prev in self._window[-self.window_size:]:
            if _jaccard(tgrams, prev) >= self.sim_threshold:
                return True, "near"

        # Not a duplicate — register it
        self._hashes.add(h)
        self._window.append(tgrams)
        if len(self._window) > self.window_size * 2:
            self._window = self._window[-self.window_size:]
        return False, ""

    @property
    def seen_hashes(self) -> list[str]:
        return list(self._hashes)


# ─────────────────────────────────────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    # Collapse hyphenated line-breaks: "knowl-\nedge" → "knowledge"
    text = re.sub(r"-\n(\w)", r"\1", text)
    # Normalize whitespace but preserve paragraph breaks
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove page numbers / headers that are lone numbers or very short lines
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not re.fullmatch(r"\d{1,4}", stripped):
            lines.append(stripped)
    return "\n".join(lines).strip()


# ─────────────────────────────────────────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────────────────────────────────────────

def split_into_passages(text: str, max_words: int = 200) -> list[str]:
    """Split text at paragraph boundaries, keeping passages under max_words."""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    passages: list[str] = []
    current: list[str] = []
    current_words = 0

    for para in paragraphs:
        wc = len(para.split())
        if current_words + wc > max_words and current:
            passages.append(" ".join(current))
            current = []
            current_words = 0
        # If a single paragraph exceeds max_words, hard-split it
        if wc > max_words:
            words = para.split()
            for i in range(0, len(words), max_words):
                passages.append(" ".join(words[i:i + max_words]))
        else:
            current.append(para)
            current_words += wc

    if current:
        passages.append(" ".join(current))

    return [p for p in passages if len(p.split()) >= 10]


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING SAMPLE GENERATION
# ─────────────────────────────────────────────────────────────────────────────

_QUESTION_TEMPLATES = [
    "What does the following passage explain?",
    "Summarize the key points of this passage.",
    "What is the main idea discussed below?",
    "What can be learned from this text?",
    "Explain what the following excerpt is about.",
    "What information is conveyed in this passage?",
    "Describe what the following text is saying.",
    "What is the author communicating in this excerpt?",
]

def passage_to_instruct(passage: str, idx: int) -> dict:
    question = _QUESTION_TEMPLATES[idx % len(_QUESTION_TEMPLATES)]
    return {
        "instruction": f"{question}\n\n{passage}",
        "response": passage,
    }


def passage_to_chat_pair(passage: str, idx: int) -> dict:
    question = _QUESTION_TEMPLATES[idx % len(_QUESTION_TEMPLATES)]
    prompt = f"<|user|> {question}\n\n{passage} <|assistant|>"
    return {
        "pos": f"{prompt} {passage}",
        "neg": f"{prompt} I don't have information about that.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# PROGRESS / CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────────

class Progress:
    """
    Persists which pages have been processed and seen content hashes so
    deduplication works correctly across separate Colab/RunPod sessions.
    """

    def __init__(self, progress_file: Path):
        self.path = progress_file
        self._data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {
            "done_pages": [],
            "total_samples": 0,
            "skipped_dupes": 0,
            "seen_hashes": [],
        }

    def save(self):
        self.path.write_text(json.dumps(self._data, indent=2))

    def mark_done(self, page_num: int):
        if page_num not in self._data["done_pages"]:
            self._data["done_pages"].append(page_num)

    def is_done(self, page_num: int) -> bool:
        return page_num in self._data["done_pages"]

    def add_samples(self, n: int):
        self._data["total_samples"] += n

    def add_skipped(self, n: int):
        self._data["skipped_dupes"] = self._data.get("skipped_dupes", 0) + n

    def sync_hashes(self, hashes: list[str]):
        self._data["seen_hashes"] = hashes

    @property
    def seen_hashes(self) -> set[str]:
        return set(self._data.get("seen_hashes", []))

    @property
    def total_samples(self) -> int:
        return self._data["total_samples"]

    @property
    def skipped_dupes(self) -> int:
        return self._data.get("skipped_dupes", 0)

    @property
    def done_count(self) -> int:
        return len(self._data["done_pages"])


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT WRITERS
# ─────────────────────────────────────────────────────────────────────────────

def append_jsonl(path: Path, records: list[dict]):
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_text(path: Path, passages: list[str]):
    with path.open("a", encoding="utf-8") as f:
        for p in passages:
            f.write(p + "\n\n")


# ─────────────────────────────────────────────────────────────────────────────
# MERGE UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def merge_outputs(out_dir: Path, stem: str):
    """Concatenate all chunk outputs into single final files."""
    for suffix in ["_instruct.jsonl", "_chat.jsonl", "_raw.txt"]:
        parts = sorted(out_dir.glob(f"{stem}_chunk*{suffix}"))
        if not parts:
            continue
        merged = out_dir / f"{stem}_MERGED{suffix}"
        print(f"Merging {len(parts)} files → {merged.name}")
        with merged.open("w", encoding="utf-8") as out:
            for p in parts:
                out.write(p.read_text(encoding="utf-8"))
    print("Merge complete.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def parse_chunk_range(spec: str, total_pages: int) -> tuple[int, int]:
    """Parse '0-49' or '50-' or '-99' into (start, end) page indices."""
    spec = spec.strip()
    if "-" not in spec:
        raise ValueError(f"Invalid --chunks spec '{spec}'. Use 'start-end', e.g. '0-49'.")
    parts = spec.split("-", 1)
    start = int(parts[0]) if parts[0] else 0
    end   = int(parts[1]) if parts[1] else total_pages - 1
    return start, end


def run(args):
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        sys.exit(f"File not found: {pdf_path}")

    out_dir = Path(args.output_dir) if args.output_dir else pdf_path.parent / "training_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = pdf_path.stem.replace(" ", "_")
    chunk_tag = f"_chunk{args.chunks.replace('-', 'to')}" if args.chunks else ""

    # Output file paths
    instruct_out = out_dir / f"{stem}{chunk_tag}_instruct.jsonl"
    chat_out     = out_dir / f"{stem}{chunk_tag}_chat.jsonl"
    raw_out      = out_dir / f"{stem}{chunk_tag}_raw.txt"
    progress_f   = out_dir / f"{stem}{chunk_tag}_progress.json"

    # Merge mode
    if args.merge_outputs:
        merge_outputs(out_dir, stem)
        return

    print(f"\nPDF → Anthos Training Data")
    print(f"  Source : {pdf_path.name}")
    print(f"  Output : {out_dir}")
    print(f"  Format : {args.format}")

    # Extract all pages
    print("\nExtracting text from PDF...")
    pages = extract_pages(str(pdf_path))
    print(f"  {len(pages)} pages with text found")

    # Apply page range
    if args.chunks:
        start_pg, end_pg = parse_chunk_range(args.chunks, len(pages))
        pages = [p for p in pages if start_pg <= p["page_num"] <= end_pg]
        print(f"  Processing pages {start_pg}–{end_pg} ({len(pages)} pages in range)")

    progress = Progress(progress_f)
    if progress.done_count:
        print(f"  Resuming — {progress.done_count} pages already done, skipping them")

    # Restore seen hashes so near-dup detection works across sessions
    dedup = Deduplicator(
        sim_threshold=args.sim_threshold,
        seen_hashes=progress.seen_hashes,
    )

    sample_idx   = progress.total_samples
    new_samples  = 0
    new_skipped  = 0
    start_time   = time.time()

    for page in pages:
        if progress.is_done(page["page_num"]):
            continue

        text = clean_text(page["text"])
        passages = split_into_passages(text, max_words=args.max_words)

        instruct_records = []
        chat_records     = []
        kept_passages    = []
        page_skipped     = 0

        for p in passages:
            is_dup, reason = dedup.is_duplicate(p)
            if is_dup:
                page_skipped += 1
                continue
            instruct_records.append(passage_to_instruct(p, sample_idx))
            chat_records.append(passage_to_chat_pair(p, sample_idx))
            kept_passages.append(p)
            sample_idx += 1

        new_skipped += page_skipped
        progress.add_skipped(page_skipped)

        # Persist updated hash set so next chunk benefits from this session's dedup
        progress.sync_hashes(dedup.seen_hashes)

        if not instruct_records:
            progress.mark_done(page["page_num"])
            progress.save()
            continue

        fmt = args.format
        if fmt in ("instruct", "all"):
            append_jsonl(instruct_out, instruct_records)
        if fmt in ("chat", "all"):
            append_jsonl(chat_out, chat_records)
        if fmt in ("raw", "all"):
            append_text(raw_out, kept_passages)

        n = len(instruct_records)
        new_samples += n
        progress.add_samples(n)
        progress.mark_done(page["page_num"])
        progress.save()

        elapsed = time.time() - start_time
        skip_note = f"  [{page_skipped} dupes removed]" if page_skipped else ""
        print(f"  Page {page['page_num']:4d} → {n:3d} unique samples{skip_note}  "
              f"(total: {progress.total_samples}, elapsed: {elapsed:.0f}s)")

    elapsed = time.time() - start_time
    print(f"\nDone. {new_samples} unique samples written, "
          f"{new_skipped} duplicates removed, in {elapsed:.1f}s")
    print(f"Lifetime totals — samples: {progress.total_samples}, "
          f"dupes removed: {progress.skipped_dupes}")

    if args.format in ("instruct", "all") and instruct_out.exists():
        print(f"\nInstruct JSONL : {instruct_out}")
    if args.format in ("chat", "all") and chat_out.exists():
        print(f"Chat pairs     : {chat_out}")
    if args.format in ("raw", "all") and raw_out.exists():
        print(f"Raw text       : {raw_out}")

    print("\nNext steps:")
    print("  Colab/RunPod instruct training:")
    print(f"    python train.py --tier instruct --data {instruct_out}")
    print("  Or for raw text (history tier):")
    if raw_out.exists():
        print(f"    python train.py --tier history --data {raw_out}")
    print("\nTo merge multiple chunk runs:")
    print(f"    python pdf_to_training.py {pdf_path} --merge-outputs")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Convert a PDF into Anthos training data (instruct, chat, or raw).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
          # Full PDF → instruct JSONL
          python pdf_to_training.py textbook.pdf

          # Only pages 0–49 (Colab session 1)
          python pdf_to_training.py textbook.pdf --chunks 0-49

          # Pages 50–99 (Colab session 2)
          python pdf_to_training.py textbook.pdf --chunks 50-99

          # Last session picks up automatically (progress tracked)
          python pdf_to_training.py textbook.pdf --chunks 100-

          # Merge all chunk outputs into one file
          python pdf_to_training.py textbook.pdf --merge-outputs

          # All formats at once
          python pdf_to_training.py textbook.pdf --format all
        """),
    )
    p.add_argument("pdf", help="Path to input PDF file")
    p.add_argument(
        "--chunks",
        metavar="START-END",
        default=None,
        help="Page range to process, e.g. '0-49', '50-99', '100-' (0-indexed)",
    )
    p.add_argument(
        "--format",
        choices=["instruct", "chat", "raw", "all"],
        default="instruct",
        help="Output format (default: instruct)",
    )
    p.add_argument(
        "--max-words",
        type=int,
        default=200,
        help="Max words per training passage (default: 200)",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <pdf_dir>/training_data/)",
    )
    p.add_argument(
        "--merge-outputs",
        action="store_true",
        help="Merge all chunk output files into final MERGED files",
    )
    p.add_argument(
        "--sim-threshold",
        type=float,
        default=0.7,
        help="Jaccard similarity threshold for near-duplicate removal 0–1 "
             "(default: 0.7 — lower catches more dupes, higher is stricter)",
    )

    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
