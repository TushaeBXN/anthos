"""
anthos/quality_filter.py — Data Quality Filtering Pipeline

Removes low-quality documents from pretraining corpora before training.
Quality filtering can have as large an impact as 5× more data — the
FineWeb (Penedo et al. 2024) and Dolma (Soldaini et al. 2024) papers both
show that heuristic + model-based filtering consistently improves downstream
benchmark performance.

Three filter levels:
  1. Heuristic filters (fast, O(len)):
       - Minimum/maximum document length
       - Non-letter character ratio (punctuation/digit heavy → bad)
       - Whitespace fraction
       - Stop-word coverage (low → garbled text)
       - Repeated line ratio (template/boilerplate text)
       - URL density

  2. Perplexity filter (slow, requires a reference LM):
       - Documents with perplexity > threshold are removed
       - Use a small kenlm model or Anthos itself for scoring
       - Targets outlier documents (extremely low or high perplexity)

  3. Language filter (fast with fastText):
       - Removes non-target-language documents
       - Requires fastText lid.176.bin model

Usage:
    pipeline = QualityFilterPipeline(
        min_chars=100, max_chars=100_000,
        min_stop_word_frac=0.1,
        max_repeated_line_frac=0.3,
        target_lang="en", lang_threshold=0.65,
    )
    kept = pipeline.filter_file("data/raw.jsonl", "data/clean.jsonl")
    print(f"Kept {kept} / total documents")

    # Or in-memory:
    clean_docs = pipeline.filter_texts(list_of_strings)
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Heuristic filter helpers
# ─────────────────────────────────────────────────────────────────────────────

# Common English stop words for coverage check
_EN_STOPWORDS = frozenset({
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "it",
    "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
    "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
    "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know",
    "take", "people", "into", "year", "your", "good", "some", "could",
    "them", "see", "other", "than", "then", "now", "look", "only", "come",
    "its", "over", "think", "also", "back", "after", "use", "two", "how",
    "our", "work", "first", "well", "way", "even", "new", "want", "any",
})

_URL_RE    = re.compile(r"https?://\S+|www\.\S+")
_WORD_RE   = re.compile(r"\b[a-zA-Z]{2,}\b")


def _letter_frac(text: str) -> float:
    if not text:
        return 0.0
    letters = sum(1 for c in text if unicodedata.category(c).startswith("L"))
    return letters / len(text)


def _whitespace_frac(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if c.isspace()) / len(text)


def _stop_word_frac(text: str) -> float:
    words = _WORD_RE.findall(text.lower())
    if not words:
        return 0.0
    return sum(1 for w in words if w in _EN_STOPWORDS) / len(words)


def _repeated_line_frac(text: str) -> float:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return 0.0
    return 1.0 - len(set(lines)) / len(lines)


def _url_density(text: str) -> float:
    if not text:
        return 0.0
    urls  = _URL_RE.findall(text)
    url_chars = sum(len(u) for u in urls)
    return url_chars / len(text)


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic filter
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HeuristicFilterConfig:
    min_chars:            int   = 100       # minimum document character count
    max_chars:            int   = 100_000   # maximum document character count
    min_letter_frac:      float = 0.55      # at least 55% letter characters
    max_whitespace_frac:  float = 0.30      # at most 30% whitespace
    min_stop_word_frac:   float = 0.08      # at least 8% stop words
    max_repeated_line_frac: float = 0.50    # at most 50% duplicate lines
    max_url_density:      float = 0.20      # at most 20% URL characters
    min_mean_word_len:    float = 3.0       # average word length
    max_mean_word_len:    float = 20.0      # average word length


class HeuristicFilter:
    """Fast O(len) per-document quality checks."""

    def __init__(self, cfg: HeuristicFilterConfig = HeuristicFilterConfig()):
        self.cfg = cfg

    def passes(self, text: str) -> tuple[bool, str]:
        """
        Returns (passes, reason).
        reason is empty string when passes=True.
        """
        cfg = self.cfg

        n_chars = len(text)
        if n_chars < cfg.min_chars:
            return False, f"too_short ({n_chars})"
        if n_chars > cfg.max_chars:
            return False, f"too_long ({n_chars})"

        lf = _letter_frac(text)
        if lf < cfg.min_letter_frac:
            return False, f"low_letter_frac ({lf:.2f})"

        wf = _whitespace_frac(text)
        if wf > cfg.max_whitespace_frac:
            return False, f"high_whitespace ({wf:.2f})"

        sw = _stop_word_frac(text)
        if sw < cfg.min_stop_word_frac:
            return False, f"low_stop_words ({sw:.2f})"

        rl = _repeated_line_frac(text)
        if rl > cfg.max_repeated_line_frac:
            return False, f"high_repeated_lines ({rl:.2f})"

        ud = _url_density(text)
        if ud > cfg.max_url_density:
            return False, f"high_url_density ({ud:.2f})"

        words = _WORD_RE.findall(text)
        if words:
            mean_wl = sum(len(w) for w in words) / len(words)
            if mean_wl < cfg.min_mean_word_len:
                return False, f"short_mean_word ({mean_wl:.1f})"
            if mean_wl > cfg.max_mean_word_len:
                return False, f"long_mean_word ({mean_wl:.1f})"

        return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# Language filter (fastText)
# ─────────────────────────────────────────────────────────────────────────────

class LanguageFilter:
    """
    Language identification using fastText lid.176.bin model.

    Download: https://fasttext.cc/docs/en/language-identification.html
      wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

    Requires: pip install fasttext-wheel  (or fasttext)
    """

    def __init__(
        self,
        model_path:    str   = "models/lid.176.bin",
        target_lang:   str   = "en",
        threshold:     float = 0.65,
    ):
        self.target_lang = target_lang
        self.threshold   = threshold
        self._model      = None
        self._model_path = model_path

    def _load(self):
        if self._model is None:
            try:
                import fasttext
                self._model = fasttext.load_model(self._model_path)
            except (ImportError, Exception) as e:
                raise RuntimeError(
                    f"fastText language model not available: {e}\n"
                    f"Install: pip install fasttext-wheel\n"
                    f"Download: wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
                )

    def passes(self, text: str) -> tuple[bool, str]:
        try:
            self._load()
        except RuntimeError as e:
            return True, ""   # skip filter if model unavailable

        # fastText expects single-line input
        clean = text.replace("\n", " ").strip()[:1000]
        labels, probs = self._model.predict(clean, k=1)
        lang = labels[0].replace("__label__", "")
        prob = float(probs[0])

        if lang != self.target_lang or prob < self.threshold:
            return False, f"wrong_lang ({lang} {prob:.2f})"
        return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# Perplexity filter (model-based)
# ─────────────────────────────────────────────────────────────────────────────

class PerplexityFilter:
    """
    Filter documents based on perplexity under a small reference LM.

    Very high perplexity → random/garbled text.
    Very low perplexity  → templated/repeated boilerplate.

    Uses any causal LM that exposes logits: Anthos, GPT-2, or kenlm.

    Requires: pip install kenlm   (for kenlm backend)
              or pass an Anthos model directly.
    """

    def __init__(
        self,
        min_ppl:        float          = 10.0,
        max_ppl:        float          = 1000.0,
        model=None,                            # Anthos model (optional)
        tokenizer_path: str            = "gpt2",
        device:         str            = "cpu",
        max_tokens:     int            = 256,  # cap for speed
    ):
        self.min_ppl        = min_ppl
        self.max_ppl        = max_ppl
        self._model         = model
        self._tokenizer_path = tokenizer_path
        self._device        = device
        self._max_tokens    = max_tokens
        self._tokenizer     = None

    def _get_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_path)
        return self._tokenizer

    def _perplexity(self, text: str) -> float:
        if self._model is None:
            return float("nan")

        import torch
        tok  = self._get_tokenizer()
        ids  = tok.encode(text, add_special_tokens=False)[:self._max_tokens]
        if len(ids) < 2:
            return float("nan")
        ids_t = torch.tensor([ids], dtype=torch.long, device=self._device)

        with torch.no_grad():
            try:
                logits = self._model(ids_t[:, :-1])
                if isinstance(logits, tuple):
                    logits = logits[0]
                import torch.nn.functional as F
                log_probs = F.log_softmax(logits[0], dim=-1)   # (T-1, V)
                targets   = ids_t[0, 1:]
                nll       = -log_probs.gather(1, targets.unsqueeze(1)).mean().item()
                return float(nll) * 1.4427   # nats → bits (log2e)
            except Exception:
                return float("nan")

    def passes(self, text: str) -> tuple[bool, str]:
        ppl = self._perplexity(text)
        if ppl != ppl:   # nan check
            return True, ""
        if ppl < self.min_ppl:
            return False, f"low_ppl ({ppl:.1f})"
        if ppl > self.max_ppl:
            return False, f"high_ppl ({ppl:.1f})"
        return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class QualityFilterPipeline:
    """
    Multi-stage document quality filter pipeline.

    Stages run in order; a document fails as soon as one stage rejects it.
    Stages are cheapest-first by default.

    Usage:
        pipeline = QualityFilterPipeline(
            min_chars=100, max_chars=100_000,
            target_lang="en", lang_threshold=0.65,
            lang_model_path="models/lid.176.bin",
        )
        kept = pipeline.filter_file("data/raw.jsonl", "data/clean.jsonl")

        # Or in-memory:
        clean = pipeline.filter_texts(["doc1", "doc2", ...])
    """

    def __init__(
        self,
        # Heuristic filter params
        min_chars:              int   = 100,
        max_chars:              int   = 100_000,
        min_letter_frac:        float = 0.55,
        max_whitespace_frac:    float = 0.30,
        min_stop_word_frac:     float = 0.08,
        max_repeated_line_frac: float = 0.50,
        max_url_density:        float = 0.20,
        # Language filter params
        target_lang:            str   = "en",
        lang_threshold:         float = 0.65,
        lang_model_path:        str   = "models/lid.176.bin",
        enable_lang_filter:     bool  = True,
        # Perplexity filter params
        min_ppl:                float = 10.0,
        max_ppl:                float = 1000.0,
        ppl_model=None,
        tokenizer_path:         str   = "gpt2",
        enable_ppl_filter:      bool  = False,
    ):
        heuristic_cfg = HeuristicFilterConfig(
            min_chars=min_chars, max_chars=max_chars,
            min_letter_frac=min_letter_frac,
            max_whitespace_frac=max_whitespace_frac,
            min_stop_word_frac=min_stop_word_frac,
            max_repeated_line_frac=max_repeated_line_frac,
            max_url_density=max_url_density,
        )
        self.heuristic_filter = HeuristicFilter(heuristic_cfg)
        self.lang_filter = (
            LanguageFilter(lang_model_path, target_lang, lang_threshold)
            if enable_lang_filter else None
        )
        self.ppl_filter = (
            PerplexityFilter(min_ppl, max_ppl, ppl_model, tokenizer_path)
            if enable_ppl_filter else None
        )

    def _passes(self, text: str) -> tuple[bool, str]:
        ok, reason = self.heuristic_filter.passes(text)
        if not ok:
            return False, reason
        if self.lang_filter is not None:
            ok, reason = self.lang_filter.passes(text)
            if not ok:
                return False, reason
        if self.ppl_filter is not None:
            ok, reason = self.ppl_filter.passes(text)
            if not ok:
                return False, reason
        return True, ""

    def filter_texts(self, texts: list[str], verbose: bool = False) -> list[str]:
        """Filter an in-memory list of strings. Returns only passing texts."""
        kept = []
        reasons: dict[str, int] = {}
        for text in texts:
            ok, reason = self._passes(text)
            if ok:
                kept.append(text)
            else:
                reasons[reason] = reasons.get(reason, 0) + 1
        if verbose:
            n_removed = len(texts) - len(kept)
            print(f"[quality_filter] {len(kept)}/{len(texts)} kept ({n_removed} removed)")
            for r, c in sorted(reasons.items(), key=lambda x: -x[1])[:10]:
                print(f"  {r}: {c}")
        return kept

    def filter_file(
        self,
        input_path:  str,
        output_path: str,
        text_key:    str  = "text",
        verbose:     bool = True,
    ) -> int:
        """
        Filter a JSONL file, writing passing documents to output_path.
        Returns number of documents written.
        """
        input_path  = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        n_total   = 0
        n_kept    = 0
        reasons: dict[str, int] = {}

        with open(input_path) as fin, open(output_path, "w") as fout:
            for line in fin:
                n_total += 1
                try:
                    obj  = json.loads(line)
                    text = obj.get(text_key, "") or ""
                except Exception:
                    text = line.strip()
                    obj  = {text_key: text}

                ok, reason = self._passes(text)
                if ok:
                    fout.write(json.dumps(obj) + "\n")
                    n_kept += 1
                else:
                    reasons[reason] = reasons.get(reason, 0) + 1

                if verbose and n_total % 100_000 == 0:
                    print(f"  [{n_total:,}] kept {n_kept:,} ({n_kept/n_total:.1%})", flush=True)

        n_removed = n_total - n_kept
        if verbose:
            print(f"[quality_filter] {n_kept:,}/{n_total:,} kept "
                  f"({n_removed/max(n_total,1):.1%} removed)")
            for r, c in sorted(reasons.items(), key=lambda x: -x[1])[:10]:
                print(f"  {r}: {c}")

        return n_kept
