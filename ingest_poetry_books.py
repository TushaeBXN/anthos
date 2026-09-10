#!/usr/bin/env python3
"""
ingest_poetry_books.py — Poetry, Books, Drama, Literary Reasoning

Covers (English only):
  Poetry:
    - biglam/gutenberg-poetry-corpus      (line+author+title → recite/explain)
    - suayptalha/Poetry-Foundation-Poems  (poem+poet+tags → analyze)
  Literary Reasoning / Q&A:
    - agentlans/literary-reasoning                      (Q&A format)
    - bestofbothworldsenjoyer/literary-reasoning-filtered (Q&A format)
    - Hananguyen12/shakespeare-QA-plays                 (Q&A)
    - NuBerea/literary-analogy                          (analogy format)
    - laion/llama-nemotron-science-reasoning-on-canonical-think-full (think→answer)
  Book Summaries:
    - ubaada/booksum-complete-cleaned    (chapter → summary)
    - LeverageX/book-summarization       (title+author → summary)
  Books (prose text):
    - common-pile/pre_1929_books         (public domain)
    - stevez80/Sci-Fi-Books-gutenberg    (sci-fi)
    - AlekseyKorshuk/fairy-tale-books
    - AlekseyKorshuk/fantasy-books
    - AlekseyKorshuk/fiction-books
    - AlekseyKorshuk/thriller-books
    - AlekseyKorshuk/mystery-crime-books
    - AlekseyKorshuk/drama-books
    - AlekseyKorshuk/romance-books
  Plays:
    - AlephFunk/storyworld-plays
    - alst10/beckett-dramatic-works-cpt

Skipped:
  - All non-English poetry (Persian, Pashto, Spanish, Multilingual classification)
  - PoetryMTEB/* (retrieval/classification tasks, no prose Q&A)
  - Tokenized datasets (pg_books-tokenized, books3-SmolLM2-sorted)
  - open-phi/programming_books_llama (programming, not literature)
  - chcaa/kb-books (Danish)
  - biglam/gallica_literary_fictions (French)
  - crazyjeannot/fr_literary_dataset_base (French)
  - Helsinki-NLP/opus_books (multilingual parallel)
  - Amba/mt5-small-finetuned-amazon-en-es_books_dataset (bilingual)
  - pfaha/goodreads-books (metadata only)
  - applied-ai-018/pretraining_v1-omega_books (raw pretraining, no structure)
  - autoevaluate/autoeval-staging-eval-* (eval harness artifact)
  - dataset-rewriter/CudyPokemonAdventures (off-topic)
  - PCNTechnologyEnterprise/Books (check format; fallback to text converter)
  - emozilla/pg_books-tokenized (tokenized IDs only)
  - aklein4/books3-SmolLM2-sorted (tokenized)

Output: data/poetry_books_sft.jsonl
Merge:  cat data/sft_master.jsonl data/poetry_books_sft.jsonl > /tmp/m.jsonl && mv /tmp/m.jsonl data/sft_master.jsonl
"""

import json, re
from pathlib import Path
from datasets import load_dataset

OUTPUT = Path("data/poetry_books_sft.jsonl")

SYS = {
    "poetry": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You have a deep love of poetry — its imagery, rhythm, and emotional truth. "
        "You can recite, analyze, explain, and discuss poems in ways that open them up for anyone."
    ),
    "literature": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You engage with literature, storytelling, and the written word with depth and enthusiasm. "
        "You explain themes, context, and meaning in ways that make great works accessible to everyone."
    ),
    "reasoning": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You think carefully and reason through complex questions step by step. "
        "You give clear, well-supported answers that help people understand and learn."
    ),
    "science": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You explain scientific ideas and research clearly and accurately, "
        "making them accessible without losing the substance."
    ),
}


def mc(q, a, domain="literature"):
    q, a = q.strip(), a.strip()[:4000]
    if not q or not a or len(q) < 8 or len(a) < 15:
        return None
    return {"conversations": [
        {"from": "system", "value": SYS.get(domain, SYS["literature"])},
        {"from": "human",  "value": q},
        {"from": "gpt",    "value": a},
    ]}


def is_english(text, threshold=0.75):
    sample = text[:300]
    if not sample:
        return False
    ratio = sum(1 for c in sample if ord(c) < 128) / len(sample)
    return ratio >= threshold


def strip_think(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


# ── Poetry ────────────────────────────────────────────────────────────────────

def conv_gutenberg_poetry(row):
    """biglam/gutenberg-poetry-corpus: line+author+title fields."""
    line = str(row.get("line", "") or "").strip()
    title = str(row.get("title", "") or "").strip()
    author = str(row.get("author", "") or "").strip()
    if not line or len(line) < 20 or not is_english(line):
        return None
    if author and title:
        q = f'Share and explain a line from "{title}" by {author}.'
        a = f'Here is a line from "{title}" by {author}:\n\n{line}\n\nThis line captures a moment of poetic expression, using language to convey emotion, imagery, or thought in a concentrated form.'
    elif title:
        q = f'Share a line from the poem "{title}".'
        a = f'From "{title}":\n\n{line}'
    else:
        q = "Share a line of poetry and explain its meaning."
        a = f'{line}\n\nThis poetic line uses careful word choice and imagery to express a deeper truth or feeling.'
    return mc(q, a, "poetry")


def conv_poetry_foundation(row):
    """suayptalha/Poetry-Foundation-Poems: poem+poet+title+tags fields."""
    poem = str(row.get("poem", row.get("Poem", row.get("content", ""))) or "").strip()
    title = str(row.get("title", row.get("Title", "")) or "").strip()
    poet = str(row.get("poet", row.get("Poet", row.get("author", ""))) or "").strip()
    tags = row.get("tags", row.get("Tags", ""))
    if not poem or len(poem) < 40 or not is_english(poem):
        return None
    tag_str = ""
    if isinstance(tags, list):
        tag_str = ", ".join(str(t) for t in tags if t)
    elif isinstance(tags, str) and tags:
        tag_str = tags
    if title and poet:
        q = f'Recite and analyze the poem "{title}" by {poet}.'
    elif title:
        q = f'Recite and explain the poem "{title}".'
    else:
        q = "Share a poem and explain its meaning."
    answer_parts = []
    if title and poet:
        answer_parts.append(f'"{title}" by {poet}\n')
    answer_parts.append(poem[:2000])
    if tag_str:
        answer_parts.append(f'\nThemes: {tag_str}')
    return mc(q, "\n".join(answer_parts), "poetry")


# ── Literary Reasoning / Q&A ──────────────────────────────────────────────────

def conv_literary_reasoning(row):
    """agentlans/literary-reasoning and filtered variant: Q&A or instruction/response."""
    for qf, af in [("question", "answer"), ("instruction", "response"),
                   ("input", "output"), ("prompt", "completion")]:
        q = str(row.get(qf, "") or "").strip()
        a = str(row.get(af, "") or "").strip()
        if q and a and len(a) > 20:
            return mc(q, a, "reasoning")
    # Try messages list
    msgs = row.get("messages", row.get("conversations"))
    if isinstance(msgs, list):
        q = a = ""
        for m in msgs:
            role = str(m.get("role", m.get("from", ""))).lower()
            content = str(m.get("content", m.get("value", ""))).strip()
            if role in ("user", "human") and not q:
                q = content
            elif role in ("assistant", "gpt") and q and not a:
                a = strip_think(content)
        if q and a:
            return mc(q, a, "reasoning")
    return None


def conv_shakespeare_qa(row):
    """Hananguyen12/shakespeare-QA-plays: question+answer or context+question+answer."""
    q = str(row.get("question", row.get("Question", "")) or "").strip()
    a = str(row.get("answer", row.get("Answer", "")) or "").strip()
    context = str(row.get("context", row.get("Context", "")) or "").strip()
    if not q or not a or len(a) < 10:
        return None
    if context and len(context) > 20:
        full_q = f"Context from Shakespeare:\n\n{context[:500]}\n\nQuestion: {q}"
    else:
        full_q = q
    return mc(full_q, a, "literature")


def conv_literary_analogy(row):
    """NuBerea/literary-analogy: analogy-style entries."""
    for qf, af in [("question", "answer"), ("prompt", "completion"),
                   ("analogy", "explanation"), ("input", "output")]:
        q = str(row.get(qf, "") or "").strip()
        a = str(row.get(af, "") or "").strip()
        if q and a and len(a) > 10:
            return mc(q, a, "literature")
    text = str(row.get("text", "") or "").strip()
    if text and len(text) > 50:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) >= 2:
            return mc(f"Complete or explain this literary analogy: {lines[0]}", "\n".join(lines[1:])[:1500], "literature")
    return None


def conv_nemotron_reasoning(row):
    """laion/llama-nemotron-science-reasoning: conversations with think tags."""
    convs = row.get("conversations", row.get("messages", []))
    if not isinstance(convs, list):
        return None
    q = a = ""
    for m in convs:
        role = str(m.get("from", m.get("role", ""))).lower()
        content = str(m.get("value", m.get("content", ""))).strip()
        if role in ("human", "user") and not q:
            q = content
        elif role in ("gpt", "assistant") and q and not a:
            a = strip_think(content)
    if q and a and len(a) > 20:
        return mc(q, a, "science")
    return None


# ── Book Summaries ─────────────────────────────────────────────────────────────

def conv_booksum(row):
    """ubaada/booksum-complete-cleaned: chapter+summary or title+summary."""
    summary = str(row.get("summary", row.get("Summary", "")) or "").strip()
    chapter = str(row.get("chapter", row.get("Chapter", row.get("text", ""))) or "").strip()
    title = str(row.get("title", row.get("Title", row.get("book_id", ""))) or "").strip()
    if not summary or len(summary) < 30 or not is_english(summary):
        return None
    if title:
        q = f'Summarize this passage from "{title}".'
    elif chapter:
        q = f"Summarize the following passage from a book:\n\n{chapter[:500]}"
    else:
        q = "Provide a summary of this literary passage."
    return mc(q, summary[:3000], "literature")


def conv_book_summarization(row):
    """LeverageX/book-summarization: title+author+summary or similar."""
    summary = str(row.get("summary", row.get("Summary", row.get("abstract", ""))) or "").strip()
    title = str(row.get("title", row.get("Title", row.get("book", ""))) or "").strip()
    author = str(row.get("author", row.get("Author", "")) or "").strip()
    if not summary or len(summary) < 30 or not is_english(summary):
        return None
    if title and author:
        q = f'Give me a summary of "{title}" by {author}.'
    elif title:
        q = f'Summarize the book "{title}".'
    else:
        q = "Summarize this book."
    return mc(q, summary[:3000], "literature")


# ── Prose Text (books / plays) ────────────────────────────────────────────────

def conv_prose_text(row, domain="literature", genre="literary work"):
    """Generic text-field converter for prose books/plays."""
    text = str(row.get("text", row.get("content", row.get("body", ""))) or "").strip()
    if len(text) < 200 or not is_english(text):
        return None
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return None
    topic = lines[0][:200].lstrip('#').strip()
    if len(topic) < 10 and len(lines) > 1:
        topic = " ".join(lines[:2])[:200]
    if len(topic) < 10:
        return None
    q = f"Discuss the following excerpt from a {genre}:\n\n{topic}"
    return mc(q, text[:3500], domain)


def conv_beckett(row):
    return conv_prose_text(row, "literature", "dramatic work by Samuel Beckett")


def conv_storyworld(row):
    text = str(row.get("text", row.get("play", row.get("script", ""))) or "").strip()
    title = str(row.get("title", row.get("name", "")) or "").strip()
    if len(text) < 200 or not is_english(text):
        return None
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    topic = lines[0][:150] if lines else title or "this play"
    q = f"Read and discuss this excerpt from a play: {topic}"
    return mc(q, text[:3500], "literature")


DATASETS = [
    # Poetry
    {"name": "gutenberg_poetry",       "id": "biglam/gutenberg-poetry-corpus",         "split": "train", "max": 5000, "conv": conv_gutenberg_poetry},
    {"name": "poetry_foundation",      "id": "suayptalha/Poetry-Foundation-Poems",      "split": "train", "max": 5000, "conv": conv_poetry_foundation},
    # Literary reasoning / Q&A
    {"name": "literary_reasoning",     "id": "agentlans/literary-reasoning",            "split": "train", "max": 5000, "conv": conv_literary_reasoning},
    {"name": "literary_reasoning_flt", "id": "bestofbothworldsenjoyer/literary-reasoning-filtered", "split": "train", "max": 5000, "conv": conv_literary_reasoning},
    {"name": "shakespeare_qa",         "id": "Hananguyen12/shakespeare-QA-plays",       "split": "train", "max": 5000, "conv": conv_shakespeare_qa},
    {"name": "literary_analogy",       "id": "NuBerea/literary-analogy",                "split": "train", "max": 3000, "conv": conv_literary_analogy},
    {"name": "nemotron_reasoning",     "id": "laion/llama-nemotron-science-reasoning-on-canonical-think-full", "split": "train", "max": 5000, "conv": conv_nemotron_reasoning},
    # Book summaries
    {"name": "booksum",                "id": "ubaada/booksum-complete-cleaned",         "split": "train", "max": 5000, "conv": conv_booksum},
    {"name": "book_summarization",     "id": "LeverageX/book-summarization",            "split": "train", "max": 5000, "conv": conv_book_summarization},
    # Books — prose text
    {"name": "pre_1929_books",         "id": "common-pile/pre_1929_books",              "split": "train", "max": 5000, "conv": lambda r: conv_prose_text(r, "literature", "classic public domain book")},
    {"name": "scifi_books",            "id": "stevez80/Sci-Fi-Books-gutenberg",         "split": "train", "max": 3000, "conv": lambda r: conv_prose_text(r, "literature", "science fiction novel")},
    {"name": "fairy_tale_books",       "id": "AlekseyKorshuk/fairy-tale-books",         "split": "train", "max": 3000, "conv": lambda r: conv_prose_text(r, "literature", "fairy tale")},
    {"name": "fantasy_books",          "id": "AlekseyKorshuk/fantasy-books",            "split": "train", "max": 3000, "conv": lambda r: conv_prose_text(r, "literature", "fantasy novel")},
    {"name": "fiction_books",          "id": "AlekseyKorshuk/fiction-books",            "split": "train", "max": 5000, "conv": lambda r: conv_prose_text(r, "literature", "fiction novel")},
    {"name": "thriller_books",         "id": "AlekseyKorshuk/thriller-books",           "split": "train", "max": 3000, "conv": lambda r: conv_prose_text(r, "literature", "thriller novel")},
    {"name": "mystery_books",          "id": "AlekseyKorshuk/mystery-crime-books",      "split": "train", "max": 3000, "conv": lambda r: conv_prose_text(r, "literature", "mystery novel")},
    {"name": "drama_books",            "id": "AlekseyKorshuk/drama-books",              "split": "train", "max": 3000, "conv": lambda r: conv_prose_text(r, "literature", "drama")},
    {"name": "romance_books",          "id": "AlekseyKorshuk/romance-books",            "split": "train", "max": 3000, "conv": lambda r: conv_prose_text(r, "literature", "romance novel")},
    # Plays
    {"name": "storyworld_plays",       "id": "AlephFunk/storyworld-plays",              "split": "train", "max": 3000, "conv": conv_storyworld},
    {"name": "beckett_drama",          "id": "alst10/beckett-dramatic-works-cpt",       "split": "train", "max": 2000, "conv": conv_beckett},
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
