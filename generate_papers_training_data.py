#!/usr/bin/env python3
"""
generate_papers_training_data.py — Academic paper knowledge for Anthos.

Two sources:
  1. arXiv API (free, no key) — abstracts from CS categories
  2. papers-we-love GitHub — topic README files with curated paper descriptions

Teaches Anthos the theory behind code: algorithms, distributed systems,
security concepts, PL theory, ML techniques — at the depth a senior engineer
would know, not just the names.

Usage:
    python3 generate_papers_training_data.py
    python3 generate_papers_training_data.py --arxiv-n 3000 --out data/papers_knowledge.jsonl

Output: data/papers_knowledge.jsonl (ShareGPT format, ready for train.py)
"""

import argparse
import json
import random
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

SYSTEM = (
    "You are Anthos, a coding AI created by Brian Tushae Thomas. "
    "You have deep knowledge of computer science theory, algorithms, distributed systems, "
    "security, programming language design, and machine learning. "
    "You explain concepts clearly, connect theory to practical implementation, "
    "and can discuss academic ideas at the level of a senior engineer or researcher. "
    "You are precise. No filler."
)

# ── arXiv categories to pull ──────────────────────────────────────────────────
# Format: (category_id, human_readable_name)
ARXIV_CATEGORIES = [
    ("cs.SE",  "Software Engineering"),
    ("cs.PL",  "Programming Languages"),
    ("cs.CR",  "Cryptography and Security"),
    ("cs.DS",  "Data Structures and Algorithms"),
    ("cs.DC",  "Distributed, Parallel, and Cluster Computing"),
    ("cs.LG",  "Machine Learning"),
    ("cs.AI",  "Artificial Intelligence"),
    ("cs.DB",  "Databases"),
    ("cs.NI",  "Networking and Internet Architecture"),
    ("cs.OS",  "Operating Systems"),
]

ARXIV_NS = "http://www.w3.org/2005/Atom"

# ── papers-we-love topic README URLs ─────────────────────────────────────────
PWL_TOPICS = [
    ("Algorithms",              "https://raw.githubusercontent.com/papers-we-love/papers-we-love/main/algorithms/README.md"),
    ("Distributed Systems",     "https://raw.githubusercontent.com/papers-we-love/papers-we-love/main/distributed_systems/README.md"),
    ("Machine Learning",        "https://raw.githubusercontent.com/papers-we-love/papers-we-love/main/machine_learning/README.md"),
    ("Security",                "https://raw.githubusercontent.com/papers-we-love/papers-we-love/main/security/README.md"),
    ("Databases",               "https://raw.githubusercontent.com/papers-we-love/papers-we-love/main/databases/README.md"),
    ("Networking",              "https://raw.githubusercontent.com/papers-we-love/papers-we-love/main/networking/README.md"),
    ("Programming Languages",   "https://raw.githubusercontent.com/papers-we-love/papers-we-love/main/programming-languages/README.md"),
    ("Operating Systems",       "https://raw.githubusercontent.com/papers-we-love/papers-we-love/main/operating-systems/README.md"),
    ("Compilers",               "https://raw.githubusercontent.com/papers-we-love/papers-we-love/main/compilers/README.md"),
    ("Cryptography",            "https://raw.githubusercontent.com/papers-we-love/papers-we-love/main/cryptography/README.md"),
    ("Computer Architecture",   "https://raw.githubusercontent.com/papers-we-love/papers-we-love/main/computer-architecture/README.md"),
    ("Type Theory",             "https://raw.githubusercontent.com/papers-we-love/papers-we-love/main/type-theory/README.md"),
    ("Concurrency",             "https://raw.githubusercontent.com/papers-we-love/papers-we-love/main/concurrency/README.md"),
    ("Testing",                 "https://raw.githubusercontent.com/papers-we-love/papers-we-love/main/testing/README.md"),
]


def fetch(url: str, delay: float = 0.0) -> str | None:
    if delay:
        time.sleep(delay)
    req = urllib.request.Request(url, headers={"User-Agent": "anthos-trainer/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  [skip] {url}: {e}")
        return None


# ── arXiv ─────────────────────────────────────────────────────────────────────

def fetch_arxiv_abstracts(category: str, max_results: int = 300) -> list[dict]:
    """Pull paper titles + abstracts from arXiv for a given CS category."""
    results = []
    batch = 100
    start = 0

    while len(results) < max_results:
        n = min(batch, max_results - len(results))
        params = urllib.parse.urlencode({
            "search_query": f"cat:{category}",
            "start": start,
            "max_results": n,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        })
        url = f"https://export.arxiv.org/api/query?{params}"
        xml_text = fetch(url, delay=1.0)  # arXiv rate limit: 1 req/sec
        if not xml_text:
            break

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            print(f"  [xml error] {e}")
            break

        entries = root.findall(f"{{{ARXIV_NS}}}entry")
        if not entries:
            break

        for entry in entries:
            title = (entry.findtext(f"{{{ARXIV_NS}}}title") or "").strip().replace("\n", " ")
            abstract = (entry.findtext(f"{{{ARXIV_NS}}}summary") or "").strip().replace("\n", " ")
            # Trim very long abstracts
            if len(abstract) > 1200:
                abstract = abstract[:1200].rsplit(" ", 1)[0] + "..."
            if title and abstract and len(abstract) > 80:
                results.append({"title": title, "abstract": abstract, "category": category})

        start += len(entries)
        if len(entries) < n:
            break

    return results


def arxiv_to_records(paper: dict) -> list[dict]:
    """Turn one arXiv paper into multiple Q&A training records."""
    title    = paper["title"]
    abstract = paper["abstract"]
    cat      = paper["category"]

    qa_pairs = [
        (
            f"What is the paper \"{title}\" about?",
            abstract,
        ),
        (
            f"Explain the key ideas from \"{title}\" in plain terms.",
            f"This paper ({cat}) covers: {abstract}",
        ),
        (
            f"What problem does \"{title}\" address?",
            f"The paper addresses: {abstract}",
        ),
    ]

    records = []
    for question, answer in qa_pairs:
        records.append({
            "conversations": [
                {"from": "system", "value": SYSTEM},
                {"from": "human",  "value": question},
                {"from": "gpt",    "value": answer},
            ]
        })
    return records


# ── papers-we-love ────────────────────────────────────────────────────────────

def parse_pwl_readme(markdown: str, topic: str) -> list[dict]:
    """
    Extract paper entries from a papers-we-love README.
    Entries look like: - [Paper Title](url) — description
    or just: - [Paper Title](url)
    """
    entries = []
    for line in markdown.splitlines():
        m = re.match(r"^\s*[-*]\s+\[([^\]]+)\]\([^)]+\)\s*[-–—]?\s*(.*)", line)
        if not m:
            continue
        title = m.group(1).strip()
        desc  = m.group(2).strip()
        # Skip navigation/meta links
        if any(x in title.lower() for x in ["back to", "readme", "contributing", "code of conduct"]):
            continue
        if len(title) < 5:
            continue
        entries.append({"title": title, "desc": desc, "topic": topic})
    return entries


def pwl_to_records(entry: dict) -> list[dict]:
    title = entry["title"]
    desc  = entry["desc"]
    topic = entry["topic"]

    if desc:
        answer = f"{title}: {desc}"
    else:
        answer = (
            f"\"{title}\" is a foundational paper in {topic}. "
            f"It appears in the papers-we-love curated list for {topic}, "
            f"which collects influential academic work that practitioners should know."
        )

    qa_pairs = [
        (
            f"What is \"{title}\"?",
            answer,
        ),
        (
            f"Why is \"{title}\" important in {topic}?",
            answer + (f" It's considered essential reading for understanding {topic}." if not desc else ""),
        ),
    ]

    records = []
    for question, ans in qa_pairs:
        records.append({
            "conversations": [
                {"from": "system", "value": SYSTEM},
                {"from": "human",  "value": question},
                {"from": "gpt",    "value": ans},
            ]
        })
    return records


def pwl_topic_summary(topic: str, entries: list[dict]) -> dict | None:
    if len(entries) < 3:
        return None
    names = [e["title"] for e in entries[:12]]
    bullet_list = "\n".join(f"- \"{t}\"" for t in names)
    return {
        "conversations": [
            {"from": "system", "value": SYSTEM},
            {"from": "human",  "value": f"What are the most important papers to read in {topic}?"},
            {"from": "gpt",    "value": f"Key papers in {topic} (from papers-we-love):\n\n{bullet_list}"},
        ]
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arxiv-n", type=int, default=300,
                        help="Max abstracts per arXiv category (default 300)")
    parser.add_argument("--out",     type=str, default="data/papers_knowledge.jsonl")
    parser.add_argument("--no-arxiv", action="store_true", help="Skip arXiv (faster)")
    parser.add_argument("--no-pwl",   action="store_true", help="Skip papers-we-love")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []

    # ── arXiv ────────────────────────────────────────────────────────────────
    if not args.no_arxiv:
        print("\n=== arXiv abstracts ===")
        for cat_id, cat_name in ARXIV_CATEGORIES:
            print(f"Fetching {cat_name} ({cat_id})... ", end="", flush=True)
            papers = fetch_arxiv_abstracts(cat_id, max_results=args.arxiv_n)
            print(f"{len(papers)} papers")
            for paper in papers:
                all_records.extend(arxiv_to_records(paper))

    # ── papers-we-love ───────────────────────────────────────────────────────
    if not args.no_pwl:
        print("\n=== papers-we-love ===")
        for topic, url in PWL_TOPICS:
            print(f"Fetching {topic}... ", end="", flush=True)
            md = fetch(url)
            if not md:
                print("skip")
                continue
            entries = parse_pwl_readme(md, topic)
            print(f"{len(entries)} papers")
            for e in entries:
                all_records.extend(pwl_to_records(e))
            summary = pwl_topic_summary(topic, entries)
            if summary:
                all_records.append(summary)

    random.shuffle(all_records)

    with out_path.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n✅ {len(all_records):,} training examples → {out_path}")
    print("\nMerge into training data:")
    print("  cat data/code_teacher.jsonl data/code_eval.jsonl data/code_magicoder.jsonl \\")
    print("      data/awesome_knowledge.jsonl data/papers_knowledge.jsonl > data/code_combined.jsonl")
    print("  python3 train.py --tier code --steps 15000")


if __name__ == "__main__":
    main()
