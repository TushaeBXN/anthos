"""
report.py — Stats on the final training dataset.

Usage:
    python report.py
    python report.py --file output/merged/anthos-code-training-v1-thought.jsonl
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from config import MERGED_OUT, THOUGHT_OUT, GITHUB_RAW_OUT, CLAUDE_AI_RAW_OUT, CLAUDE_CODE_RAW_OUT


DOMAIN_KEYWORDS = {
    "ML / AI":       ["torch", "tensorflow", "neural", "model", "train", "dataset", "embedding", "gradient"],
    "Frontend":      ["react", "vue", "html", "css", "component", "jsx", "tsx", "dom", "tailwind"],
    "Backend / API": ["fastapi", "flask", "django", "express", "endpoint", "route", "rest", "graphql"],
    "Security":      ["exploit", "vulnerability", "cve", "pentest", "payload", "injection", "xss", "csrf"],
    "Infra / DevOps":["docker", "kubernetes", "terraform", "ansible", "nginx", "k8s", "ci/cd", "deploy"],
    "Data / DB":     ["sql", "postgres", "mongo", "pandas", "dataframe", "schema", "query", "index"],
    "Systems / C":   ["pointer", "malloc", "syscall", "kernel", "memory", "buffer", "assembly", "c++"],
}


def classify_domain(text: str) -> str:
    lower = text.lower()
    scores = {d: sum(1 for kw in kws if kw in lower) for d, kws in DOMAIN_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "General"


def count_tokens_est(text: str) -> int:
    return len(text) // 4


def report_file(path: str):
    path = Path(path)
    if not path.exists():
        print(f"  [missing] {path}")
        return

    total, has_thought, domains, token_est = 0, 0, Counter(), 0

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            turns = record.get("conversations", [])
            full_text = " ".join(t.get("value", "") for t in turns)
            token_est += count_tokens_est(full_text)
            if "<|thought|>" in full_text:
                has_thought += 1
            domains[classify_domain(full_text)] += 1

    if total == 0:
        print(f"  [empty] {path}")
        return

    print(f"\n{'='*55}")
    print(f"File: {path.name}")
    print(f"{'='*55}")
    print(f"  Examples:       {total:>10,}")
    print(f"  With <thought>: {has_thought:>10,}  ({100*has_thought//total}%)")
    print(f"  Token est:      {token_est:>10,}  (~{token_est/1e6:.0f}M tokens)")
    print(f"\n  Domain breakdown:")
    for domain, count in domains.most_common():
        bar = "█" * (count * 20 // total)
        print(f"    {domain:<20} {count:>6,}  {bar}")


def report_sources():
    print(f"\n{'='*55}")
    print("Source raw files")
    print(f"{'='*55}")
    for label, path in [
        ("GitHub repos",     GITHUB_RAW_OUT),
        ("Claude.ai export", CLAUDE_AI_RAW_OUT),
        ("Claude Code logs", CLAUDE_CODE_RAW_OUT),
    ]:
        p = Path(path)
        if p.exists():
            n = sum(1 for l in p.open() if l.strip())
            print(f"  {label:<22} {n:>8,} examples  ({p.stat().st_size//1024}KB)")
        else:
            print(f"  {label:<22} [not generated yet]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=None, help="Specific file to report on")
    args = parser.parse_args()

    report_sources()

    if args.file:
        report_file(args.file)
    else:
        report_file(str(MERGED_OUT))
        if Path(THOUGHT_OUT).exists():
            report_file(str(THOUGHT_OUT))
