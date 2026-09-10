#!/usr/bin/env python3
"""
ingest_music_cinema.py — Music Theory, Music Captions, Cinema

Covers (text-usable, English):
  Music Theory / Knowledge:
    - Musictheory94/Chordonomicon          (chord progressions + theory)
    - m-a-p/MusicTheoryBench              (music theory Q&A benchmark)
    - Seeker38/music_abc_notation_with_music_theory (ABC notation + theory text)
    - apple-jun/MusicPro-7k               (professional music Q&A)
  Music Description / Captions:
    - google/MusicCaps                    (audio captions → "describe this music")
    - MaggiePai/MusicCaptionGeneration_SongDescriber_test (song descriptions)
    - laion/captioned-ai-music-snippets   (captions)
    - m-a-p/MusicPile                     (music text corpus)
  Cinema / Film:
    - ShariqMukadam/cinematch-movie-dataset (movie metadata Q&A)
    - BEE-spoke-data/fineweb-cinema-100k  (cinema web text)
    - gunahkarcasper/ai-video-cinematography-prompts (cinematography)

Skipped (audio/video/images/non-English/misc):
  - All benjamin-paine/free-music-archive-* (raw audio)
  - All AstraMindAI/Music-POSTPROCESS-* (audio processing)
  - All CinematicT2vData/* (video)
  - TTS-AGI/vocal-music-whisper-demo-audio (audio)
  - drengskapur/midi-classical-music (MIDI binary)
  - Sweaterdog/music-theory-images-20k (images)
  - InfoBayAI/Music_Festival_PSD_Design_Dataset (PSD files)
  - lingbow/tiktok-trending-hashtags-music (hashtags only)
  - WaveGenAI/youtube-cc-by-music (audio)
  - arch-raven/music-fingerprint-dataset (audio fingerprints)
  - amaai-lab/MusicBench (audio benchmark)
  - amazon/music-off-policy-evaluation-benchmark (recommendation)
  - a3xrfgb/cinematic-stills (images)
  - Tsu7am1/Cinematic-DiT-Video-Dataset (video)
  - csoai/cinematic-world-stills (images)
  - sevenu777/cinema-assets (images)
  - Data-Gouv-ML/les-salles-de-cinema-en-ile-de-france (French)
  - danielritchie/cinematic-mood-palette (images)
  - Pratofeitoo/cinematic_photo_prompts_SDXL (image prompts)
  - louisbrulenaudet/code-cinema-image-animee (French)
  - flying101/cinematic-slowmotion (video)
  - akba08/ultra-realistic-cinematic-photography (images)

Output: data/music_cinema_sft.jsonl
Merge:  cat data/sft_master.jsonl data/music_cinema_sft.jsonl > /tmp/m.jsonl && mv /tmp/m.jsonl data/sft_master.jsonl
"""

import json, re
from pathlib import Path
from datasets import load_dataset

OUTPUT = Path("data/music_cinema_sft.jsonl")

SYS = {
    "music": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You have deep knowledge of music theory, composition, history, and culture. "
        "You explain music concepts clearly — from chord progressions to genre history — "
        "making music education accessible to everyone."
    ),
    "arts": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You engage with art, culture, and creative works with depth and enthusiasm. "
        "You explain themes, techniques, and context in ways that are accessible to everyone."
    ),
    "cinema": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You have broad knowledge of film, cinema history, directors, storytelling techniques, "
        "and the art of cinematography. You make film analysis accessible and engaging."
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


# ── Music Theory ───────────────────────────────────────────────────────────────

def conv_chordonomicon(row):
    """Chord progressions dataset: chord+key+mode+genre or similar fields."""
    # Try multiple possible field names
    chords = str(row.get("chord_progression", row.get("chords", row.get("progression", ""))) or "").strip()
    key = str(row.get("key", row.get("key_signature", "")) or "").strip()
    mode = str(row.get("mode", row.get("scale", "")) or "").strip()
    genre = str(row.get("genre", row.get("style", "")) or "").strip()
    desc = str(row.get("description", row.get("text", row.get("explanation", ""))) or "").strip()
    name = str(row.get("name", row.get("title", "")) or "").strip()

    if not chords and not desc:
        return None
    if not is_english((chords + " " + desc)[:200]):
        return None

    if chords:
        context = []
        if key:
            context.append(f"Key: {key}")
        if mode:
            context.append(f"Mode/Scale: {mode}")
        if genre:
            context.append(f"Genre: {genre}")
        if name:
            context.append(f"Name: {name}")
        ctx_str = " | ".join(context)
        q = f"Explain this chord progression: {chords}" + (f" ({ctx_str})" if ctx_str else "")
        a = desc if desc else (
            f"This chord progression ({chords}) "
            + (f"in {key} {mode} " if key else "")
            + (f"is commonly used in {genre} music. " if genre else "")
            + "It creates a harmonic movement that defines the mood and feel of the music. "
            + "Each chord transition contributes to the overall emotional journey of the piece."
        )
    else:
        lines = [l.strip() for l in desc.splitlines() if l.strip()]
        if not lines:
            return None
        q = f"Explain this music theory concept: {lines[0][:150]}"
        a = desc[:2000]
    return mc(q, a, "music")


def conv_music_theory_bench(row):
    """m-a-p/MusicTheoryBench: MCQ or Q&A format."""
    q = str(row.get("question", row.get("Question", row.get("query", ""))) or "").strip()
    a = str(row.get("answer", row.get("Answer", row.get("response", ""))) or "").strip()
    options = row.get("options", row.get("choices", None))
    if not q:
        return None
    if not a:
        # Try to get answer from options + answer_idx
        idx = row.get("answer_idx", row.get("label", row.get("correct", None)))
        if isinstance(options, list) and idx is not None:
            try:
                a = str(options[int(idx)])
            except (ValueError, IndexError, TypeError):
                return None
    if not a or len(a) < 2:
        return None
    if options and isinstance(options, list) and len(options) > 1:
        opts_str = "\n".join(f"  {chr(65+i)}. {opt}" for i, opt in enumerate(options))
        full_q = f"{q}\n\nOptions:\n{opts_str}"
    else:
        full_q = q
    return mc(full_q, a, "music")


def conv_abc_music_theory(row):
    """ABC notation + music theory text."""
    abc = str(row.get("abc", row.get("notation", row.get("score", ""))) or "").strip()
    theory = str(row.get("theory", row.get("text", row.get("description", row.get("explanation", "")))) or "").strip()
    title = str(row.get("title", row.get("name", "")) or "").strip()
    if not theory and not abc:
        return None
    if not is_english((theory or abc)[:200]):
        return None
    if theory and abc:
        q = f"Explain the music theory behind this piece" + (f': "{title}"' if title else "") + f":\n\n{abc[:300]}"
        a = theory[:2500]
    elif theory:
        lines = [l.strip() for l in theory.splitlines() if l.strip()]
        if not lines or len(theory) < 30:
            return None
        q = f"Explain this music theory concept: {lines[0][:150]}"
        a = theory[:2500]
    else:
        q = f"What does this ABC music notation represent?" + (f' ("{title}")' if title else "")
        a = f"This ABC notation{' for ' + title if title else ''} represents a musical piece. ABC notation is a text-based music notation system where each letter represents a note (A-G), and additional symbols indicate rhythm, octave, key signature, and other musical elements.\n\nNotation:\n{abc[:500]}"
    return mc(q, a, "music")


def conv_music_pro(row):
    """apple-jun/MusicPro-7k: professional music Q&A."""
    for qf, af in [("question", "answer"), ("instruction", "response"),
                   ("input", "output"), ("prompt", "completion"), ("query", "response")]:
        q = str(row.get(qf, "") or "").strip()
        a = str(row.get(af, "") or "").strip()
        if q and a and len(a) > 15 and is_english(q):
            a = strip_think(a)
            return mc(q, a, "music")
    # Try messages list
    msgs = row.get("messages", row.get("conversations", []))
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
            return mc(q, a, "music")
    # text field fallback
    text = str(row.get("text", "") or "").strip()
    if text and len(text) > 80 and is_english(text):
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if lines:
            return mc(f"Explain this music concept: {lines[0][:150]}", text[:2500], "music")
    return None


# ── Music Captions / Description ──────────────────────────────────────────────

def conv_musiccaps(row):
    """google/MusicCaps: ytid+caption+aspect_list+author_id."""
    caption = str(row.get("caption", row.get("description", row.get("text", ""))) or "").strip()
    aspects = row.get("aspect_list", row.get("aspects", []))
    if not caption or len(caption) < 30 or not is_english(caption):
        return None
    if isinstance(aspects, list) and aspects:
        aspect_str = ", ".join(str(a) for a in aspects[:8])
        q = f"Describe a piece of music with these characteristics: {aspect_str}"
    else:
        q = "Describe the style and feel of this piece of music."
    return mc(q, caption, "music")


def conv_song_describer(row):
    """Song description dataset: song info + description."""
    caption = str(row.get("caption", row.get("description", row.get("text", ""))) or "").strip()
    title = str(row.get("title", row.get("song_title", row.get("track", ""))) or "").strip()
    artist = str(row.get("artist", row.get("artist_name", "")) or "").strip()
    if not caption or len(caption) < 20 or not is_english(caption):
        return None
    if title and artist:
        q = f'Describe the song "{title}" by {artist}.'
    elif title:
        q = f'Describe the song "{title}".'
    else:
        q = "Describe this song and its musical qualities."
    return mc(q, caption, "music")


def conv_music_caption_generic(row):
    """Generic music caption: caption/description + optional metadata."""
    caption = str(row.get("caption", row.get("description", row.get("text", ""))) or "").strip()
    if not caption or len(caption) < 30 or not is_english(caption):
        return None
    q = "Describe this piece of music — its style, mood, instruments, and feel."
    return mc(q, caption, "music")


def conv_music_pile(row):
    """m-a-p/MusicPile: text corpus about music."""
    text = str(row.get("text", row.get("content", row.get("body", ""))) or "").strip()
    if len(text) < 150 or not is_english(text):
        return None
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return None
    topic = lines[0][:200].lstrip('#').strip()
    if len(topic) < 10 and len(lines) > 1:
        topic = " ".join(lines[:2])[:200]
    if len(topic) < 10:
        return None
    q = f"Explain this music topic: {topic}"
    return mc(q, text[:3000], "music")


# ── Cinema / Film ─────────────────────────────────────────────────────────────

def conv_movie_dataset(row):
    """ShariqMukadam/cinematch-movie-dataset: movie metadata Q&A."""
    title = str(row.get("title", row.get("Title", row.get("movie", ""))) or "").strip()
    plot = str(row.get("plot", row.get("overview", row.get("description", row.get("summary", "")))) or "").strip()
    genre = str(row.get("genre", row.get("genres", "")) or "").strip()
    director = str(row.get("director", row.get("Director", "")) or "").strip()
    year = str(row.get("year", row.get("release_year", row.get("release_date", ""))) or "").strip()[:4]
    rating = str(row.get("rating", row.get("imdb_rating", row.get("score", ""))) or "").strip()
    cast = row.get("cast", row.get("actors", row.get("stars", "")))
    if isinstance(cast, list):
        cast = ", ".join(str(c) for c in cast[:5])
    else:
        cast = str(cast or "").strip()

    if not title or not plot:
        return None
    if not is_english(plot[:200]):
        return None

    q = f'Tell me about the movie "{title}".'
    parts = []
    if title and year:
        parts.append(f'"{title}" ({year}) is a film')
    elif title:
        parts.append(f'"{title}" is a film')
    if genre:
        parts.append(f"in the {genre} genre.")
    if director:
        parts.append(f"Directed by {director}.")
    if cast:
        parts.append(f"Starring {cast}.")
    parts.append(f"\nPlot: {plot[:2000]}")
    if rating:
        parts.append(f"\nRating: {rating}")
    return mc(q, " ".join(parts), "cinema")


def conv_cinema_web(row):
    """BEE-spoke-data/fineweb-cinema-100k: web text about cinema."""
    text = str(row.get("text", row.get("content", "")) or "").strip()
    if len(text) < 150 or not is_english(text):
        return None
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return None
    topic = lines[0][:200].lstrip('#').strip()
    if len(topic) < 10 and len(lines) > 1:
        topic = " ".join(lines[:2])[:200]
    if len(topic) < 10:
        return None
    q = f"Discuss this film or cinema topic: {topic}"
    return mc(q, text[:3000], "cinema")


def conv_cinematography_prompts(row):
    """gunahkarcasper/ai-video-cinematography-prompts: prompt+explanation."""
    prompt = str(row.get("prompt", row.get("description", row.get("text", ""))) or "").strip()
    explanation = str(row.get("explanation", row.get("technique", row.get("notes", ""))) or "").strip()
    technique = str(row.get("technique", row.get("shot_type", row.get("style", ""))) or "").strip()
    if not prompt or len(prompt) < 20 or not is_english(prompt):
        return None
    if explanation:
        q = f"Explain this cinematography technique: {technique or prompt[:100]}"
        a = explanation[:2000]
    else:
        q = "Describe a cinematography approach for: " + prompt[:200]
        a = prompt[:2000]
    return mc(q, a, "cinema")


DATASETS = [
    # Music theory
    {"name": "chordonomicon",      "id": "Musictheory94/Chordonomicon",                             "split": "train", "max": 5000, "conv": conv_chordonomicon},
    {"name": "music_theory_bench", "id": "m-a-p/MusicTheoryBench",                                 "split": "train", "max": 5000, "conv": conv_music_theory_bench},
    {"name": "abc_music_theory",   "id": "Seeker38/music_abc_notation_with_music_theory",           "split": "train", "max": 3000, "conv": conv_abc_music_theory},
    {"name": "music_pro_7k",       "id": "apple-jun/MusicPro-7k",                                   "split": "train", "max": 5000, "conv": conv_music_pro},
    # Music captions
    {"name": "musiccaps",          "id": "google/MusicCaps",                                        "split": "train", "max": 5000, "conv": conv_musiccaps},
    {"name": "song_describer",     "id": "MaggiePai/MusicCaptionGeneration_SongDescriber_test",     "split": "test",  "max": 3000, "conv": conv_song_describer},
    {"name": "ai_music_captions",  "id": "laion/captioned-ai-music-snippets",                       "split": "train", "max": 3000, "conv": conv_music_caption_generic},
    {"name": "music_pile",         "id": "m-a-p/MusicPile",                                         "split": "train", "max": 5000, "conv": conv_music_pile},
    # Cinema
    {"name": "cinematch_movies",   "id": "ShariqMukadam/cinematch-movie-dataset",                   "split": "train", "max": 5000, "conv": conv_movie_dataset},
    {"name": "fineweb_cinema",     "id": "BEE-spoke-data/fineweb-cinema-100k",                      "split": "train", "max": 5000, "conv": conv_cinema_web},
    {"name": "cinematography",     "id": "gunahkarcasper/ai-video-cinematography-prompts",           "split": "train", "max": 3000, "conv": conv_cinematography_prompts},
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
