#!/usr/bin/env python3
"""
ingest_art_media.py — Art, Media, Finance Text, News Articles

Covers:
  - allenai/art         (ART abductive reasoning: obs1+obs2+hyp1+hyp2+label)
  - GEM/ART             (GEM NLG task: input+target)
  - artemis13fowl/imdb  (IMDB reviews: text+label → sentiment Q&A)
  - Artificio/WikiArt   (art history metadata: artist+title+style+genre+description)
  - hugginglearners/russia-ukraine-conflict-articles (news → media literacy)
  - adameubanks/filtered_articles_by_year (news articles)
  - artefactory/Argimi-Ardian-Finance-10k-text (finance text)
  - artefactory/ledger-long-context-multi-kpi (finance KPI documents)

Skipped from batch:
  - All ARTPARK-IISc/Vaani* (Indian speech audio)
  - All artur-muratov/multilingual-speech-commands* (speech)
  - All ArtificialAnalysis/* (AI benchmarks)
  - minnesotanlp/LLM-Artifacts (LLM output artifacts)
  - sayakpaul/diffusers-qa-chatbot-artifacts (image gen)
  - Fhrozen/relaion-art (image dataset)
  - mondk/ASCII-art-video (video)
  - maxwellinked/time-lapse-artifacts (video)
  - brandonyang/artem-fold-towel (robotics)
  - bigscience-data/roots_id_indonesian_news_articles_2017 (Indonesian)
  - cy0307/ropedia-xperience-10m-task-suite-artifacts (task artifacts)
  - pietrolesci/anchoral-paper-artefacts (research metadata)
  - artplus/PrismLayersPro (art tool)
  - Arthur12137/SoftVTBench (benchmark)
  - huggan/few-shot-art-painting (image only, no text Q&A)

Output: data/art_media_sft.jsonl
Merge:  cat data/sft_master.jsonl data/art_media_sft.jsonl > /tmp/m.jsonl && mv /tmp/m.jsonl data/sft_master.jsonl
"""

import json, re
from pathlib import Path
from datasets import load_dataset

OUTPUT = Path("data/art_media_sft.jsonl")

SYS = {
    "reasoning": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You think carefully and reason through complex questions step by step, "
        "giving clear, well-supported answers."
    ),
    "arts": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You have broad knowledge of art, culture, and creative works. "
        "You explain artistic movements, styles, and works in ways that are accessible and insightful."
    ),
    "media": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You help people think critically about media, news, and information. "
        "You explain context, identify bias, and help people understand complex events clearly."
    ),
    "finance": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You explain financial concepts, documents, and data clearly, "
        "making finance accessible to people who've been shut out of this knowledge."
    ),
    "general": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You give clear, accurate, helpful answers in plain language so anyone can understand."
    ),
}


def mc(q, a, domain="general"):
    q, a = q.strip(), a.strip()[:4000]
    if not q or not a or len(q) < 8 or len(a) < 15:
        return None
    return {"conversations": [
        {"from": "system", "value": SYS.get(domain, SYS["general"])},
        {"from": "human",  "value": q},
        {"from": "gpt",    "value": a},
    ]}


def is_english(text, threshold=0.75):
    sample = text[:300]
    if not sample:
        return False
    return sum(1 for c in sample if ord(c) < 128) / len(sample) >= threshold


def strip_think(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


# ── allenai/art — Abductive Reasoning in Text ──────────────────────────────────

def conv_art_abductive(row):
    """obs1+obs2+hyp1+hyp2+label → which hypothesis better explains?"""
    obs1 = str(row.get("obs1", row.get("observation_1", "")) or "").strip()
    obs2 = str(row.get("obs2", row.get("observation_2", "")) or "").strip()
    hyp1 = str(row.get("hyp1", row.get("hypothesis_1", "")) or "").strip()
    hyp2 = str(row.get("hyp2", row.get("hypothesis_2", "")) or "").strip()
    label = row.get("label", row.get("answer", None))
    if not obs1 or not obs2 or not hyp1 or not hyp2:
        return None
    q = (
        f"Read these two observations and two possible explanations, then decide which explanation "
        f"better connects them.\n\n"
        f"Observation 1: {obs1}\n"
        f"Observation 2: {obs2}\n\n"
        f"Hypothesis 1: {hyp1}\n"
        f"Hypothesis 2: {hyp2}\n\n"
        f"Which hypothesis (1 or 2) better explains the connection between the two observations? Explain your reasoning."
    )
    if label is not None:
        try:
            chosen = int(label) + 1  # label 0→Hyp1, 1→Hyp2
            chosen_text = hyp1 if chosen == 1 else hyp2
            a = (
                f"Hypothesis {chosen} better explains the observations.\n\n"
                f"Reasoning: Given that '{obs1}' happened, and then '{obs2}' happened, "
                f"the most logical connecting explanation is: '{chosen_text}'. "
                f"This creates a coherent causal chain between the two observations, "
                f"while the other hypothesis creates a less plausible or consistent narrative."
            )
        except (ValueError, TypeError):
            chosen_text = str(label)
            a = f"The better explanation is: {chosen_text}\n\nThis hypothesis creates a more coherent narrative connecting the two observations."
    else:
        a = (
            f"Looking at the two observations, I need to find which hypothesis creates "
            f"a better causal story.\n\nHypothesis 1: {hyp1}\n\nHypothesis 2: {hyp2}\n\n"
            f"The more plausible explanation is the one that fits naturally between the observations as a logical cause."
        )
    return mc(q, a, "reasoning")


# ── GEM/ART ────────────────────────────────────────────────────────────────────

def conv_gem_art(row):
    """GEM task format: typically input+target or similar."""
    for qf, af in [("input", "target"), ("source", "target"), ("premise", "hypothesis"),
                   ("obs1", "hyp1"), ("context", "continuation")]:
        q = str(row.get(qf, "") or "").strip()
        a = str(row.get(af, "") or "").strip()
        if q and a and len(q) > 10 and len(a) > 10 and is_english(q):
            return mc(q, a, "reasoning")
    # Try concatenated references
    refs = row.get("references", row.get("targets", []))
    src = str(row.get("input", row.get("source", row.get("premise", ""))) or "").strip()
    if src and isinstance(refs, list) and refs:
        return mc(f"Complete or respond to: {src}", str(refs[0])[:2000], "reasoning")
    return None


# ── IMDB reviews ───────────────────────────────────────────────────────────────

SENTIMENT = {0: "negative", 1: "positive"}

def conv_imdb(row):
    """text+label → sentiment analysis Q&A."""
    text = str(row.get("text", row.get("review", row.get("content", ""))) or "").strip()
    label = row.get("label", row.get("sentiment", None))
    if not text or len(text) < 50 or not is_english(text):
        return None
    excerpt = text[:600].rsplit(".", 1)[0] + "." if "." in text[:600] else text[:600]
    if label is not None:
        try:
            sentiment = SENTIMENT.get(int(label), str(label))
        except (ValueError, TypeError):
            sentiment = str(label).lower()
        q = f"Read this movie review and explain the sentiment:\n\n{excerpt}"
        a = (
            f"This is a {sentiment} review. "
            f"The reviewer {'appreciates' if sentiment == 'positive' else 'criticizes'} the film, "
            f"as shown by the tone and word choice throughout the review. "
            f"Key signals: {text[600:900] if len(text) > 600 else text[-200:]}"
        )
    else:
        q = f"Analyze the sentiment and opinion expressed in this movie review:\n\n{excerpt}"
        a = f"The reviewer expresses their opinion about this film through: {text[:800]}"
    return mc(q, a, "media")


# ── WikiArt ────────────────────────────────────────────────────────────────────

def conv_wikiart(row):
    """Art metadata: artist+title+style+genre+description/caption."""
    artist = str(row.get("artist", row.get("Artist", row.get("artist_name", ""))) or "").strip()
    title = str(row.get("title", row.get("Title", row.get("filename", ""))) or "").strip()
    style = str(row.get("style", row.get("Style", row.get("art_style", ""))) or "").strip()
    genre = str(row.get("genre", row.get("Genre", row.get("content", ""))) or "").strip()
    desc = str(row.get("description", row.get("caption", row.get("text", ""))) or "").strip()

    if not artist and not title:
        return None
    parts = []
    if title and artist:
        q = f'Tell me about the artwork "{title}" by {artist}.'
        parts.append(f'"{title}" is a work by {artist}.')
    elif title:
        q = f'Tell me about the artwork "{title}".'
        parts.append(f'"{title}" is a notable artwork.')
    else:
        q = f"Tell me about the art of {artist}."
        parts.append(f"{artist} is an artist known for their distinctive work.")
    if style:
        parts.append(f"Style: {style}.")
    if genre:
        parts.append(f"Genre: {genre}.")
    if desc and len(desc) > 20:
        parts.append(desc[:1500])
    if len(parts) < 2:
        return None
    return mc(q, " ".join(parts), "arts")


# ── News / Conflict Articles ───────────────────────────────────────────────────

def conv_news_article(row, domain="media", context_label=""):
    """Generic news article converter: title+text or title+content."""
    title = str(row.get("title", row.get("Title", row.get("headline", ""))) or "").strip()
    text = str(row.get("text", row.get("content", row.get("body", row.get("article", ""))) or "")).strip()
    if not text and not title:
        return None
    if not is_english((title + " " + text)[:300]):
        return None
    body = text or title
    if len(body) < 80:
        return None
    excerpt = body[:2500]
    if title:
        q = f"Summarize and explain this news article: {title}"
        a = excerpt
    else:
        lines = [l.strip() for l in body.splitlines() if l.strip()]
        q = f"Summarize and explain this news article:\n\n{lines[0][:200]}"
        a = "\n".join(lines[1:])[:2500] or excerpt
    if context_label:
        q = f"{context_label}\n\n{q}"
    return mc(q, a, domain)


def conv_conflict_article(row):
    """Russia-Ukraine conflict articles — treat as current events / media literacy."""
    return conv_news_article(row, "media")


# ── Finance Text ───────────────────────────────────────────────────────────────

def conv_finance_text(row):
    """artefactory/Argimi-Ardian-Finance-10k-text: text field."""
    text = str(row.get("text", row.get("content", row.get("body", ""))) or "").strip()
    title = str(row.get("title", row.get("company", row.get("ticker", ""))) or "").strip()
    if not text or len(text) < 100 or not is_english(text):
        return None
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    topic = title or (lines[0][:150] if lines else "this financial document")
    q = f"Explain the key financial information in this document: {topic}"
    return mc(q, text[:3000], "finance")


def conv_ledger_kpi(row):
    """artefactory/ledger-long-context-multi-kpi: KPI extraction from financial docs."""
    text = str(row.get("text", row.get("document", row.get("content", ""))) or "").strip()
    kpis = row.get("kpis", row.get("labels", row.get("answer", None)))
    question = str(row.get("question", row.get("query", "")) or "").strip()
    if not text or len(text) < 80 or not is_english(text):
        return None
    if question:
        q = question
        a = str(kpis or text[:2000])
    else:
        q = f"Identify and explain the key performance indicators (KPIs) in this financial document:\n\n{text[:400]}"
        a = str(kpis) if kpis else text[:2500]
    return mc(q, a[:3000], "finance")


# ── Dataset config ─────────────────────────────────────────────────────────────

DATASETS = [
    # Reasoning
    {"name": "allenai_art_abductive",   "id": "allenai/art",         "split": "validation", "max": 5000, "conv": conv_art_abductive},
    {"name": "gem_art",                 "id": "GEM/ART",             "split": "train",      "max": 5000, "conv": conv_gem_art},
    # Media / sentiment
    {"name": "imdb_reviews",            "id": "artemis13fowl/imdb",  "split": "train",      "max": 5000, "conv": conv_imdb},
    # Art history
    {"name": "wikiart",                 "id": "Artificio/WikiArt",   "split": "train",      "max": 5000, "conv": conv_wikiart},
    # News / current events
    {"name": "conflict_articles",       "id": "hugginglearners/russia-ukraine-conflict-articles", "split": "train", "max": 3000, "conv": conv_conflict_article},
    {"name": "filtered_articles",       "id": "adameubanks/filtered_articles_by_year",            "split": "train", "max": 5000, "conv": lambda r: conv_news_article(r, "media")},
    # Finance
    {"name": "finance_10k_text",        "id": "artefactory/Argimi-Ardian-Finance-10k-text",       "split": "train", "max": 5000, "conv": conv_finance_text},
    {"name": "ledger_kpi",              "id": "artefactory/ledger-long-context-multi-kpi",         "split": "train", "max": 3000, "conv": conv_ledger_kpi},
]


def main():
    all_pairs = []

    for cfg in DATASETS:
        name = cfg["name"]
        ds_id = cfg["id"]
        split = cfg["split"]
        max_pairs = cfg["max"]
        conv_fn = cfg["conv"]

        print(f"\n[{name}]  {ds_id}")
        loaded = False
        first_err = ""
        for try_split in [split, "train", "test", "validation"]:
            try:
                ds = load_dataset(ds_id, split=try_split, streaming=True)
                pairs, skipped = [], 0
                for row in ds:
                    if len(pairs) >= max_pairs:
                        break
                    try:
                        p = conv_fn(row)
                        if p:
                            pairs.append(p)
                        else:
                            skipped += 1
                    except Exception:
                        skipped += 1
                suffix = f" [split={try_split}]" if try_split != split else ""
                print(f"  {len(pairs):,} pairs  ({skipped} skipped){suffix}")
                all_pairs.extend(pairs)
                loaded = True
                break
            except Exception as e:
                if not first_err:
                    first_err = str(e)[:100]
        if not loaded:
            print(f"  SKIP — {first_err}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        for p in all_pairs:
            f.write(json.dumps(p) + "\n")

    print(f"\n{'═'*60}")
    print(f"Total: {len(all_pairs):,} pairs → {OUTPUT}")
    print(f"\nMerge:")
    print(f"  cat data/sft_master.jsonl {OUTPUT} > /tmp/m.jsonl && mv /tmp/m.jsonl data/sft_master.jsonl")


if __name__ == "__main__":
    main()
