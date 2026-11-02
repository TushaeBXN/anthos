#!/usr/bin/env python3
"""
generate_awesome_training_data.py — Turn awesome-* lists into Anthos training data.

Fetches curated awesome lists from GitHub, parses library entries, and generates
Q&A training pairs teaching Anthos the real ecosystem (no hallucinated libraries).

Usage:
    python3 generate_awesome_training_data.py
    python3 generate_awesome_training_data.py --n 5000 --out data/awesome_knowledge.jsonl

Output: data/awesome_knowledge.jsonl (ShareGPT format, ready for train.py)
"""

import argparse
import json
import random
import re
import urllib.request
from pathlib import Path

SYSTEM = (
    "You are Anthos, a coding AI created by Brian Tushae Thomas. "
    "You have deep knowledge of programming ecosystems across Python, JavaScript, "
    "Go, Rust, and security tooling. You recommend real, well-maintained libraries "
    "and tools. You are direct and specific. No filler. No made-up libraries."
)

# Verified awesome lists — raw GitHub URLs
AWESOME_LISTS = [
    # High-level languages
    ("Python",          "https://raw.githubusercontent.com/vinta/awesome-python/master/README.md"),
    ("JavaScript",      "https://raw.githubusercontent.com/sorrycc/awesome-javascript/master/README.md"),
    ("Node.js",         "https://raw.githubusercontent.com/sindresorhus/awesome-nodejs/main/readme.md"),
    ("Go",              "https://raw.githubusercontent.com/avelino/awesome-go/main/README.md"),
    ("Rust",            "https://raw.githubusercontent.com/rust-unofficial/awesome-rust/main/README.md"),
    ("TypeScript",      "https://raw.githubusercontent.com/dzharii/awesome-typescript/master/README.md"),
    # Systems / low-level (ODYSSEUS-7 gap fix)
    ("C",               "https://raw.githubusercontent.com/oz123/awesome-c/master/README.md"),
    ("C++",             "https://raw.githubusercontent.com/fffaraz/awesome-cpp/master/README.md"),
    ("Assembly",        "https://raw.githubusercontent.com/0xAX/asm/master/README.md"),
    ("Embedded",        "https://raw.githubusercontent.com/nhivp/Awesome-Embedded/master/README.md"),
    ("Reversing",       "https://raw.githubusercontent.com/ReversingID/Awesome-Reversing/master/README.md"),
    ("Binary Exploit",  "https://raw.githubusercontent.com/wtsxDev/Exploit-Development/master/README.md"),
    ("Malware Analysis","https://raw.githubusercontent.com/rshipp/awesome-malware-analysis/master/README.md"),
    # Security (fixed URL)
    ("Security",        "https://raw.githubusercontent.com/sbilly/awesome-security/master/README.md"),
    ("Hacking",         "https://raw.githubusercontent.com/carpedm20/awesome-hacking/master/README.md"),
    ("CTF",             "https://raw.githubusercontent.com/apsdehal/awesome-ctf/master/README.md"),
    ("Pentest",         "https://raw.githubusercontent.com/enaqx/awesome-pentest/master/README.md"),
    ("Fuzzing",         "https://raw.githubusercontent.com/cpuu/awesome-fuzzing/master/README.md"),
    # DevOps / Infra
    ("Docker",          "https://raw.githubusercontent.com/veggiemonk/awesome-docker/master/README.md"),
    ("Shell",           "https://raw.githubusercontent.com/alebcay/awesome-shell/master/README.md"),
    ("Kubernetes",      "https://raw.githubusercontent.com/ramitsurana/awesome-kubernetes/master/docs/README.md"),
]

# Q&A templates per entry: (question_template, answer_template)
# {lang} = language, {category} = category, {name} = library name, {desc} = description
QA_TEMPLATES = [
    (
        "What {lang} library should I use for {category}?",
        "`{name}` — {desc}",
    ),
    (
        "What is {name} in {lang}?",
        "{name} is a {lang} library for {category}. {desc}",
    ),
    (
        "Recommend a {lang} tool for {category}.",
        "Use `{name}`. {desc}",
    ),
    (
        "I need to do {category} in {lang}. What are my options?",
        "The standard choice is `{name}`. {desc} There may be alternatives depending on your specific needs, but this is the most commonly recommended option.",
    ),
    (
        "What does the {name} library do?",
        "`{name}` is a {lang} library: {desc}",
    ),
]


def fetch(url: str) -> str | None:
    req = urllib.request.Request(url, headers={"User-Agent": "anthos-trainer/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  [skip] {url}: {e}")
        return None


def parse_entries(markdown: str, lang: str) -> list[dict]:
    """Extract library entries from an awesome-* README."""
    entries = []
    current_category = "General"

    for line in markdown.splitlines():
        # Detect category headers (## or ###)
        h = re.match(r"^#{2,3}\s+(.+)", line)
        if h:
            current_category = h.group(1).strip()
            # Clean up common suffixes
            current_category = re.sub(r"\s*-\s*(Also see|See also).*", "", current_category, flags=re.I)
            continue

        # Match list items with a link: - [name](url) - description
        m = re.match(r"^\s*[-*]\s+\[([^\]]+)\]\([^)]+\)\s*[-–—]\s*(.+)", line)
        if not m:
            # Also try without the dash: - [name](url) — description
            m = re.match(r"^\s*[-*]\s+\[([^\]]+)\]\([^)]+\)\s*(.+)", line)
        if m:
            name = m.group(1).strip()
            desc = m.group(2).strip()
            # Remove trailing links like "Also see [...]"
            desc = re.sub(r"\s*[\(\[](Also see|See also)[^\)]*[\)\]].*", "", desc, flags=re.I)
            desc = re.sub(r"\s*-\s*\[awesome-\w+\].*", "", desc, flags=re.I)
            desc = desc.strip().rstrip(".")
            if name and desc and len(desc) > 10:
                entries.append({
                    "lang": lang,
                    "category": current_category,
                    "name": name,
                    "desc": desc,
                })

    return entries


def entry_to_records(entry: dict) -> list[dict]:
    """Generate multiple Q&A training records from one library entry."""
    records = []
    for q_tmpl, a_tmpl in QA_TEMPLATES:
        try:
            question = q_tmpl.format(**entry)
            answer   = a_tmpl.format(**entry)
        except KeyError:
            continue

        records.append({
            "conversations": [
                {"from": "system", "value": SYSTEM},
                {"from": "human",  "value": question},
                {"from": "gpt",    "value": answer},
            ]
        })
    return records


def make_category_summary(lang: str, category: str, entries: list[dict]) -> dict | None:
    """Generate one record summarizing all libs in a category."""
    if len(entries) < 2:
        return None
    lines = "\n".join(f"- `{e['name']}` — {e['desc']}" for e in entries[:10])
    return {
        "conversations": [
            {"from": "system", "value": SYSTEM},
            {"from": "human",  "value": f"What are the best {lang} libraries for {category}?"},
            {"from": "gpt",    "value": f"Here are the top options for {category} in {lang}:\n\n{lines}"},
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",   type=int, default=0,
                        help="Max examples to write (0 = all)")
    parser.add_argument("--out", type=str, default="data/awesome_knowledge.jsonl")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []

    for lang, url in AWESOME_LISTS:
        print(f"Fetching {lang} list...")
        md = fetch(url)
        if not md:
            continue

        entries = parse_entries(md, lang)
        print(f"  Parsed {len(entries)} entries")

        # Per-entry Q&A records
        for entry in entries:
            all_records.extend(entry_to_records(entry))

        # Per-category summary records
        by_cat: dict[str, list] = {}
        for e in entries:
            by_cat.setdefault(e["category"], []).append(e)
        for cat, cat_entries in by_cat.items():
            rec = make_category_summary(lang, cat, cat_entries)
            if rec:
                all_records.append(rec)

    random.shuffle(all_records)

    if args.n and args.n < len(all_records):
        all_records = all_records[:args.n]

    with out_path.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n✅ {len(all_records):,} training examples → {out_path}")
    print("\nMerge into code training:")
    print("  cat data/code_teacher.jsonl data/code_eval.jsonl data/code_magicoder.jsonl data/awesome_knowledge.jsonl > data/code_combined.jsonl")
    print("  python3 train.py --tier code --steps 15000")


if __name__ == "__main__":
    main()
