#!/usr/bin/env python3
"""
generate_wikipedia_training_data.py — PhD-breadth knowledge for Anthos via Wikipedia.

Pulls from three tiers:
  1. Wikipedia Featured Articles (~6,700) — expert-reviewed, highest quality
  2. Wikipedia Good Articles (~37,000) — editor-reviewed, solid accuracy
  3. Specific academic discipline categories — targeted PhD-domain depth

Covers: physics, chemistry, biology, mathematics, medicine, law, philosophy,
history, linguistics, economics, psychology, sociology, political science,
neuroscience, engineering, astronomy, art history, music theory,
environmental science, anthropology, and more.

Usage:
    python3 generate_wikipedia_training_data.py
    python3 generate_wikipedia_training_data.py --max-articles 10000 --out data/wiki_knowledge.jsonl

Output: data/wiki_knowledge.jsonl (ShareGPT format, ready for train.py)
"""

import argparse
import json
import random
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

SYSTEM = (
    "You are Anthos, a coding AI created by Brian Tushae Thomas. "
    "You have encyclopedic knowledge spanning all academic disciplines — "
    "physics, chemistry, biology, mathematics, medicine, law, philosophy, history, "
    "linguistics, economics, psychology, neuroscience, engineering, astronomy, art, "
    "music, environmental science, political science, sociology, and more. "
    "You explain concepts clearly and accurately, connecting ideas across disciplines. "
    "You are precise. No filler. No fabrication."
)

WIKI_API = "https://en.wikipedia.org/w/api.php"

# ── PhD-domain Wikipedia categories to harvest from ──────────────────────────
# Each entry: (display_name, wikipedia_category_title)
PHD_CATEGORIES = [
    # Natural Sciences
    ("Physics",                   "Category:Physics"),
    ("Quantum Mechanics",         "Category:Quantum mechanics"),
    ("Thermodynamics",            "Category:Thermodynamics"),
    ("Electromagnetism",          "Category:Electromagnetism"),
    ("Relativity",                "Category:Theory of relativity"),
    ("Astrophysics",              "Category:Astrophysics"),
    ("Particle Physics",          "Category:Particle physics"),
    ("Condensed Matter Physics",  "Category:Condensed matter physics"),
    ("Chemistry",                 "Category:Chemistry"),
    ("Organic Chemistry",         "Category:Organic chemistry"),
    ("Biochemistry",              "Category:Biochemistry"),
    ("Physical Chemistry",        "Category:Physical chemistry"),
    ("Materials Science",         "Category:Materials science"),
    ("Biology",                   "Category:Biology"),
    ("Molecular Biology",         "Category:Molecular biology"),
    ("Genetics",                  "Category:Genetics"),
    ("Cell Biology",              "Category:Cell biology"),
    ("Evolutionary Biology",      "Category:Evolutionary biology"),
    ("Ecology",                   "Category:Ecology"),
    ("Neuroscience",              "Category:Neuroscience"),
    ("Microbiology",              "Category:Microbiology"),
    ("Astronomy",                 "Category:Astronomy"),
    ("Cosmology",                 "Category:Physical cosmology"),
    ("Earth Science",             "Category:Earth sciences"),
    ("Geology",                   "Category:Geology"),
    ("Climatology",               "Category:Climatology"),
    # Mathematics
    ("Mathematics",               "Category:Mathematics"),
    ("Calculus",                  "Category:Calculus"),
    ("Linear Algebra",            "Category:Linear algebra"),
    ("Abstract Algebra",          "Category:Abstract algebra"),
    ("Topology",                  "Category:Topology"),
    ("Number Theory",             "Category:Number theory"),
    ("Probability Theory",        "Category:Probability theory"),
    ("Statistics",                "Category:Statistics"),
    ("Mathematical Logic",        "Category:Mathematical logic"),
    ("Geometry",                  "Category:Geometry"),
    ("Differential Equations",    "Category:Differential equations"),
    ("Combinatorics",             "Category:Combinatorics"),
    ("Graph Theory",              "Category:Graph theory"),
    ("Numerical Analysis",        "Category:Numerical analysis"),
    ("Information Theory",        "Category:Information theory"),
    # Medicine & Health
    ("Medicine",                  "Category:Medicine"),
    ("Pharmacology",              "Category:Pharmacology"),
    ("Immunology",                "Category:Immunology"),
    ("Cardiology",                "Category:Cardiology"),
    ("Oncology",                  "Category:Oncology"),
    ("Psychiatry",                "Category:Psychiatry"),
    ("Neurology",                 "Category:Neurology"),
    ("Anatomy",                   "Category:Anatomy"),
    ("Physiology",                "Category:Physiology"),
    ("Epidemiology",              "Category:Epidemiology"),
    # Social Sciences
    ("Economics",                 "Category:Economics"),
    ("Microeconomics",            "Category:Microeconomics"),
    ("Macroeconomics",            "Category:Macroeconomics"),
    ("Game Theory",               "Category:Game theory"),
    ("Behavioral Economics",      "Category:Behavioral economics"),
    ("Psychology",                "Category:Psychology"),
    ("Cognitive Science",         "Category:Cognitive science"),
    ("Social Psychology",         "Category:Social psychology"),
    ("Developmental Psychology",  "Category:Developmental psychology"),
    ("Sociology",                 "Category:Sociology"),
    ("Anthropology",              "Category:Anthropology"),
    ("Political Science",         "Category:Political science"),
    ("International Relations",   "Category:International relations"),
    ("Political Philosophy",      "Category:Political philosophy"),
    # Humanities
    ("Philosophy",                "Category:Philosophy"),
    ("Logic",                     "Category:Logic"),
    ("Ethics",                    "Category:Ethics"),
    ("Epistemology",              "Category:Epistemology"),
    ("Metaphysics",               "Category:Metaphysics"),
    ("Philosophy of Mind",        "Category:Philosophy of mind"),
    ("Philosophy of Science",     "Category:Philosophy of science"),
    ("History",                   "Category:History"),
    ("Ancient History",           "Category:Ancient history"),
    ("Medieval History",          "Category:Medieval history"),
    ("History of Science",        "Category:History of science"),
    ("Linguistics",               "Category:Linguistics"),
    ("Syntax",                    "Category:Syntax"),
    ("Semantics",                 "Category:Semantics"),
    ("Phonology",                 "Category:Phonology"),
    ("Computational Linguistics", "Category:Computational linguistics"),
    ("Law",                       "Category:Law"),
    ("Constitutional Law",        "Category:Constitutional law"),
    ("International Law",         "Category:International law"),
    ("Criminal Law",              "Category:Criminal law"),
    ("Intellectual Property",     "Category:Intellectual property law"),
    # Engineering
    ("Electrical Engineering",    "Category:Electrical engineering"),
    ("Mechanical Engineering",    "Category:Mechanical engineering"),
    ("Civil Engineering",         "Category:Civil engineering"),
    ("Chemical Engineering",      "Category:Chemical engineering"),
    ("Aerospace Engineering",     "Category:Aerospace engineering"),
    ("Biomedical Engineering",    "Category:Biomedical engineering"),
    ("Control Theory",            "Category:Control theory"),
    ("Signal Processing",         "Category:Signal processing"),
    # Arts & Humanities
    ("Music Theory",              "Category:Music theory"),
    ("Art History",               "Category:Art history"),
    ("Literary Theory",           "Category:Literary theory"),
    ("Architecture",              "Category:Architecture"),
    ("Film Theory",               "Category:Film theory"),
    # Environment & Interdisciplinary
    ("Environmental Science",     "Category:Environmental science"),
    ("Climate Change",            "Category:Climate change"),
    ("Bioinformatics",            "Category:Bioinformatics"),
    ("Systems Biology",           "Category:Systems biology"),
    ("Cognitive Neuroscience",    "Category:Cognitive neuroscience"),
]

# High-quality article lists — pull these first (best quality)
QUALITY_CATEGORIES = [
    "Category:Featured_articles",
    "Category:Good_articles",
]


def wiki_api(params: dict, delay: float = 0.1) -> dict | None:
    """Call the Wikipedia API. Returns parsed JSON or None."""
    if delay:
        time.sleep(delay)
    params["format"] = "json"
    params["formatversion"] = "2"
    url = WIKI_API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "anthos-trainer/1.0 (brian.thomas.t@gmail.com)"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as e:
        print(f"  [wiki error] {e}")
        return None


def clean_text(text: str) -> str:
    """Strip HTML tags and normalize whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()
    return text


def get_category_members(category: str, limit: int = 500) -> list[str]:
    """Get article titles in a Wikipedia category."""
    titles = []
    cmcontinue = None
    while len(titles) < limit:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmlimit": min(500, limit - len(titles)),
            "cmtype": "page",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue

        data = wiki_api(params)
        if not data:
            break

        members = data.get("query", {}).get("categorymembers", [])
        titles.extend(m["title"] for m in members if m.get("ns") == 0)

        cont = data.get("continue", {})
        cmcontinue = cont.get("cmcontinue")
        if not cmcontinue or len(members) == 0:
            break

    return titles


def get_article_extract(title: str) -> dict | None:
    """Fetch title + intro extract + section headings for one article."""
    params = {
        "action": "query",
        "prop": "extracts|categories",
        "exintro": True,         # intro only (before first heading)
        "explaintext": True,     # plain text, no HTML
        "exsectionformat": "plain",
        "titles": title,
        "redirects": True,
    }
    data = wiki_api(params, delay=0.05)
    if not data:
        return None

    pages = data.get("query", {}).get("pages", [])
    if not pages:
        return None
    page = pages[0]
    if page.get("missing"):
        return None

    extract = clean_text(page.get("extract") or "")
    if len(extract) < 100:
        return None

    # Cap length — we don't need the full article, just a rich intro
    if len(extract) > 2000:
        extract = extract[:2000].rsplit(". ", 1)[0] + "."

    return {"title": page.get("title", title), "extract": extract}


def article_to_records(article: dict, field: str = "") -> list[dict]:
    """Convert a Wikipedia article into Q&A training records."""
    title   = article["title"]
    extract = article["extract"]
    field_clause = f" in {field}" if field else ""

    qa_pairs = [
        (
            f"What is {title}?",
            extract,
        ),
        (
            f"Explain {title}.",
            extract,
        ),
        (
            f"Give me an overview of {title}{field_clause}.",
            extract,
        ),
    ]

    # For shorter extracts, also produce a "brief" version
    sentences = extract.split(". ")
    if len(sentences) >= 3:
        brief = ". ".join(sentences[:2]) + "."
        qa_pairs.append((
            f"Briefly, what is {title}?",
            brief,
        ))

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-articles",  type=int, default=50000,
                        help="Max total articles to process (default 50000)")
    parser.add_argument("--per-category",  type=int, default=200,
                        help="Max articles per PhD category (default 200)")
    parser.add_argument("--featured-limit", type=int, default=5000,
                        help="Max featured/good articles to pull (default 5000)")
    parser.add_argument("--out",           type=str, default="data/wiki_knowledge.jsonl")
    parser.add_argument("--skip-featured", action="store_true")
    parser.add_argument("--skip-phd",      action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen_titles: set[str] = set()
    all_records: list[dict] = []
    total_articles = 0

    def process_titles(titles: list[str], field: str = ""):
        nonlocal total_articles
        for title in titles:
            if total_articles >= args.max_articles:
                return
            if title in seen_titles:
                continue
            seen_titles.add(title)

            article = get_article_extract(title)
            if not article:
                continue

            records = article_to_records(article, field)
            all_records.extend(records)
            total_articles += 1

            if total_articles % 100 == 0:
                print(f"  {total_articles} articles processed ({len(all_records)} records)...")

    # ── Tier 1: Featured + Good Articles ─────────────────────────────────────
    if not args.skip_featured:
        print("\n=== Featured + Good Articles (highest quality) ===")
        for cat in QUALITY_CATEGORIES:
            if total_articles >= args.max_articles:
                break
            print(f"Fetching {cat}...")
            titles = get_category_members(cat, limit=args.featured_limit)
            random.shuffle(titles)
            process_titles(titles)

    # ── Tier 2: PhD domain categories ────────────────────────────────────────
    if not args.skip_phd:
        print(f"\n=== PhD Domain Categories ({len(PHD_CATEGORIES)} fields) ===")
        random.shuffle(PHD_CATEGORIES)
        for field_name, category in PHD_CATEGORIES:
            if total_articles >= args.max_articles:
                break
            print(f"Fetching {field_name}...")
            titles = get_category_members(category, limit=args.per_category)
            process_titles(titles, field=field_name)

    random.shuffle(all_records)

    with out_path.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n✅ {total_articles:,} articles → {len(all_records):,} training records → {out_path}")
    print("\nMerge into training data:")
    print("  cat data/code_teacher.jsonl data/code_eval.jsonl data/code_magicoder.jsonl \\")
    print("      data/awesome_knowledge.jsonl data/papers_knowledge.jsonl data/wiki_knowledge.jsonl \\")
    print("      > data/code_combined.jsonl")
    print("  python3 train.py --tier code --steps 20000")


if __name__ == "__main__":
    main()
