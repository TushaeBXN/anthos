#!/usr/bin/env python3
"""
run_pipeline.py — Anthos Training Dataset Pipeline
Single entrypoint. Runs all stages in order.

Usage:
    python run_pipeline.py                  # full run
    python run_pipeline.py --skip-github    # skip GitHub (use cached)
    python run_pipeline.py --only-normalize # just re-merge existing raw files

Env vars required (set in shell or .env):
    GITHUB_TOKEN         — GitHub personal access token (read:repo scope)
    CLAUDE_EXPORT_PATH   — path to conversations.json from Claude.ai export
    HF_TOKEN             — HuggingFace token (for push step)
    ANTHROPIC_API_KEY    — for thought-token augmentation step (optional)
"""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Anthos Dataset Pipeline")
    parser.add_argument("--skip-github", action="store_true", help="Skip GitHub ingest")
    parser.add_argument("--skip-claude-ai", action="store_true", help="Skip Claude.ai export ingest")
    parser.add_argument("--skip-claude-code", action="store_true", help="Skip Claude Code ingest")
    parser.add_argument("--only-normalize", action="store_true", help="Only run normalize step")
    parser.add_argument("--push", action="store_true", help="Push to HuggingFace after normalize")
    parser.add_argument("--augment", action="store_true", help="Run thought-token augmentation")
    args = parser.parse_args()

    print("=" * 60)
    print("  Anthos Intelligence — Training Dataset Pipeline")
    print("=" * 60)

    # ── Stage 1: Ingest ───────────────────────────────────────────
    if not args.only_normalize:
        if not args.skip_github:
            print("\n[1/5] GitHub ingest")
            from ingest.github_ingest import run as github_run
            github_run()
        else:
            print("\n[1/5] GitHub ingest — SKIPPED")

        if not args.skip_claude_ai:
            print("\n[2/5] Claude.ai export ingest")
            from ingest.claude_ai_ingest import run as claude_ai_run
            claude_ai_run()
        else:
            print("\n[2/5] Claude.ai ingest — SKIPPED")

        if not args.skip_claude_code:
            print("\n[3/5] Claude Code session ingest")
            from ingest.claude_code_ingest import run as claude_code_run
            claude_code_run()
        else:
            print("\n[3/5] Claude Code ingest — SKIPPED")
    else:
        print("\n[1-3/5] Ingest stages — SKIPPED (--only-normalize)")

    # ── Stage 2: Normalize & Merge ────────────────────────────────
    print("\n[4/5] Normalize + merge + quality filter")
    from normalize.normalize import run as normalize_run
    normalize_run()

    # ── Stage 3: Report ──────────────────────────────────────────
    print("\n[5/5] Stats report")
    _print_report()

    # ── Optional: Push to HuggingFace ────────────────────────────
    if args.push:
        print("\n[+] Pushing to HuggingFace")
        try:
            from push_to_hub import run as push_run
            push_run()
        except ImportError:
            print("  push_to_hub.py not yet built — see CLAUDE.md Priority 3")

    # ── Optional: Thought-token augmentation ─────────────────────
    if args.augment:
        print("\n[+] Thought-token augmentation")
        try:
            from augment.thought_token_wrap import run as thought_run
            thought_run()
        except ImportError:
            print("  augment/thought_token_wrap.py not yet built — see CLAUDE.md Priority 4")

    print("\n✓ Pipeline complete.")


def _print_report():
    from config import MERGED_OUT, GITHUB_RAW_OUT, CLAUDE_AI_RAW_OUT, CLAUDE_CODE_RAW_OUT
    import json

    def count_lines(p):
        p = Path(p)
        if not p.exists():
            return 0
        with p.open() as f:
            return sum(1 for line in f if line.strip())

    def estimate_tokens(p):
        p = Path(p)
        if not p.exists():
            return 0
        chars = p.stat().st_size
        return chars // 4  # rough estimate

    print(f"\n  Source breakdown:")
    print(f"    GitHub raw:      {count_lines(GITHUB_RAW_OUT):>6} examples")
    print(f"    Claude.ai raw:   {count_lines(CLAUDE_AI_RAW_OUT):>6} examples")
    print(f"    Claude Code raw: {count_lines(CLAUDE_CODE_RAW_OUT):>6} examples")
    print(f"\n  Final merged dataset:")
    print(f"    Examples:  {count_lines(MERGED_OUT):>6}")
    print(f"    Est tokens: ~{estimate_tokens(MERGED_OUT):,}")
    print(f"    Location:  {MERGED_OUT}")


if __name__ == "__main__":
    main()
