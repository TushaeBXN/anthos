"""
Nia Training Dataset Builder
Processes all PDFs in a folder into a single unified training dataset.
Deduplication runs across all files — content seen in file 1 won't appear in file 2.

Usage:
    # Full run (all PDFs, all formats)
    python3 nia_build_dataset.py

    # Resume after a Colab/RunPod timeout — picks up from where it stopped
    python3 nia_build_dataset.py --resume

    # Only process specific files (by index, printed at start)
    python3 nia_build_dataset.py --files 0-9

    # Preview what would be processed without writing anything
    python3 nia_build_dataset.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

# ─── reuse the core logic from pdf_to_training.py ────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from pdf_to_training import (
    extract_pages,
    clean_text,
    split_into_passages,
    passage_to_instruct,
    passage_to_chat_pair,
    append_jsonl,
    append_text,
    Deduplicator,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

PDF_DIR    = Path("/Users/dadsmacpro/Documents/NIA LESSONS")
OUT_DIR    = Path("/Users/dadsmacpro/nia/training_data")
STATE_FILE = OUT_DIR / "nia_build_state.json"

MAX_WORDS     = 200   # words per passage
SIM_THRESHOLD = 0.7   # near-dup Jaccard threshold

# ─────────────────────────────────────────────────────────────────────────────
# STATE — tracks progress across all files and sessions
# ─────────────────────────────────────────────────────────────────────────────

class BuildState:
    def __init__(self):
        self._s = self._load()

    def _load(self) -> dict:
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text())
        return {
            "done_files": [],       # filenames fully processed
            "total_samples": 0,
            "total_skipped": 0,
            "seen_hashes": [],      # cross-file exact-dup fingerprints
            "file_stats": {},       # filename → {samples, skipped}
        }

    def save(self):
        STATE_FILE.write_text(json.dumps(self._s, indent=2))

    def is_done(self, filename: str) -> bool:
        return filename in self._s["done_files"]

    def mark_done(self, filename: str, samples: int, skipped: int):
        if filename not in self._s["done_files"]:
            self._s["done_files"].append(filename)
        self._s["file_stats"][filename] = {"samples": samples, "skipped": skipped}
        self._s["total_samples"] += samples
        self._s["total_skipped"] += skipped

    def sync_hashes(self, hashes: list[str]):
        self._s["seen_hashes"] = hashes

    @property
    def seen_hashes(self) -> set[str]:
        return set(self._s.get("seen_hashes", []))

    @property
    def total_samples(self) -> int:
        return self._s["total_samples"]

    @property
    def total_skipped(self) -> int:
        return self._s["total_skipped"]

    @property
    def done_count(self) -> int:
        return len(self._s["done_files"])

    def reset(self):
        STATE_FILE.unlink(missing_ok=True)
        self._s = self._load()


# ─────────────────────────────────────────────────────────────────────────────
# PROCESS ONE PDF
# ─────────────────────────────────────────────────────────────────────────────

def process_pdf(
    pdf_path: Path,
    dedup: Deduplicator,
    out_instruct: Path,
    out_chat: Path,
    out_raw: Path,
    sample_idx: int,
    max_words: int,
) -> tuple[int, int]:
    """Returns (samples_written, dupes_skipped)."""
    try:
        pages = extract_pages(str(pdf_path))
    except Exception as e:
        print(f"    ERROR reading PDF: {e}")
        return 0, 0

    instruct_records = []
    chat_records     = []
    kept_passages    = []
    skipped          = 0

    for page in pages:
        text     = clean_text(page["text"])
        passages = split_into_passages(text, max_words=max_words)
        for p in passages:
            is_dup, _ = dedup.is_duplicate(p)
            if is_dup:
                skipped += 1
                continue
            instruct_records.append(passage_to_instruct(p, sample_idx))
            chat_records.append(passage_to_chat_pair(p, sample_idx))
            kept_passages.append(p)
            sample_idx += 1

    if instruct_records:
        append_jsonl(out_instruct, instruct_records)
        append_jsonl(out_chat,     chat_records)
        append_text(out_raw,       kept_passages)

    return len(instruct_records), skipped


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_file_range(spec: str, total: int) -> tuple[int, int]:
    if "-" not in spec:
        raise ValueError(f"Use 'start-end' format, e.g. '0-9'")
    parts = spec.split("-", 1)
    start = int(parts[0]) if parts[0] else 0
    end   = int(parts[1]) if parts[1] else total - 1
    return start, end


def run(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out_instruct = OUT_DIR / "nia_instruct.jsonl"
    out_chat     = OUT_DIR / "nia_chat.jsonl"
    out_raw      = OUT_DIR / "nia_raw.txt"

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {PDF_DIR}")

    if args.files:
        start, end = parse_file_range(args.files, len(pdfs))
        pdfs = pdfs[start:end+1]

    state = BuildState()
    if args.reset:
        print("Resetting build state...")
        state.reset()
        state = BuildState()

    print(f"\nNia Training Dataset Builder")
    print(f"  Source  : {PDF_DIR}")
    print(f"  Output  : {OUT_DIR}")
    print(f"  PDFs    : {len(pdfs)} files")
    print(f"  Done    : {state.done_count} already processed")
    print(f"  Samples : {state.total_samples} so far")
    print()

    if args.dry_run:
        print("DRY RUN — files that would be processed:")
        for i, p in enumerate(pdfs):
            status = "DONE" if state.is_done(p.name) else "pending"
            print(f"  [{i:2d}] {status:7s}  {p.name}")
        return

    dedup = Deduplicator(
        sim_threshold=SIM_THRESHOLD,
        seen_hashes=state.seen_hashes,
    )

    session_samples = 0
    session_skipped = 0
    start_time      = time.time()

    for i, pdf in enumerate(pdfs):
        if state.is_done(pdf.name):
            print(f"  [{i:2d}] SKIP  {pdf.name}")
            continue

        print(f"  [{i:2d}] Processing  {pdf.name} ...", end="", flush=True)

        samples, skipped = process_pdf(
            pdf_path    = pdf,
            dedup       = dedup,
            out_instruct= out_instruct,
            out_chat    = out_chat,
            out_raw     = out_raw,
            sample_idx  = state.total_samples + session_samples,
            max_words   = MAX_WORDS,
        )

        session_samples += samples
        session_skipped += skipped

        state.mark_done(pdf.name, samples, skipped)
        state.sync_hashes(dedup.seen_hashes)
        state.save()

        elapsed = time.time() - start_time
        skip_note = f"  ({skipped} dupes removed)" if skipped else ""
        print(f"  {samples} samples{skip_note}  [{elapsed:.0f}s]")

    elapsed = time.time() - start_time
    print(f"\n{'─'*60}")
    print(f"Session complete in {elapsed:.1f}s")
    print(f"  New samples   : {session_samples}")
    print(f"  Dupes removed : {session_skipped}")
    print(f"  Total samples : {state.total_samples}")
    print(f"  Total dupes   : {state.total_skipped}")
    print(f"\nOutput files:")
    print(f"  {out_instruct}")
    print(f"  {out_chat}")
    print(f"  {out_raw}")
    print(f"\nTo train Nia (instruct tier):")
    print(f"  python3 train.py --tier instruct --data {out_instruct}")
    print(f"\nTo train on RunPod — upload {OUT_DIR} and run the same command.")
    print(f"\nTo resume in a new Colab/RunPod session:")
    print(f"  python3 nia_build_dataset.py --resume")
    print(f"  (Upload nia_build_state.json alongside the script so it knows what's done)")


def main():
    p = argparse.ArgumentParser(
        description="Build Nia's training dataset from all PDFs in NIA LESSONS/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 nia_build_dataset.py                  # process all PDFs
  python3 nia_build_dataset.py --dry-run        # preview without writing
  python3 nia_build_dataset.py --files 0-9      # only first 10 PDFs
  python3 nia_build_dataset.py --resume         # continue after timeout
  python3 nia_build_dataset.py --reset          # start over from scratch
"""
    )
    p.add_argument("--files",    metavar="START-END", help="Only process PDFs at these indices")
    p.add_argument("--dry-run",  action="store_true",  help="Preview without writing files")
    p.add_argument("--resume",   action="store_true",  help="Continue from last saved state (default behavior)")
    p.add_argument("--reset",    action="store_true",  help="Wipe state and start over")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
