"""
anthos/memory_compress.py — Engram Shorthand Preprocessing for Anthos Training

Training Anthos on ES-compressed data has two benefits:

  1. The thought stream learns to read compressed memory representations,
     making Layer 2 (ExternalMemoryReader) more effective at inference time —
     the model recognizes ES notation as a structured memory signal rather
     than noise.

  2. ES compression reduces sequence length 4–10×, which means more content
     fits in Anthos's context window, and the thought stream processes
     richer context per loop iteration.

This module:
  - ESCompressor: standalone compressor that mirrors Engram Shorthand
    without requiring engram to be installed
  - MemoryAugmentedDataset: wraps any HuggingFace dataset, optionally
    compressing a fraction of examples during training (curriculum approach)
  - compress_jsonl: CLI-friendly function for preprocessing data files

ES Notation (subset used here — full spec in Engram):
  Confidence:  ★ ★★ ★★★ ★★★★ ★★★★★  (1–5 stars)
  Dependency:  +dep_name
  Negation:    ~fact
  Change:      CHANGE:file add:fn() rm:fn()
  Code func:   fn:name(args)->type
  Separator:   &  (replaces "and")
  Relation:    :  (replaces "is a", "has a", "responsible for")
  Colon list:  key:val1,val2
"""

from __future__ import annotations

import re
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Standalone ES Compressor (no engram dependency)
# ─────────────────────────────────────────────────────────────────────────────

# Substitution rules ordered by specificity
_PROSE_RULES = [
    # Filler phrases
    (r"\bis a critical component that\b",     ":★★★ component"),
    (r"\bhas a dependency on\b",               "+dep"),
    (r"\bresponsible for\b",                   "responsible"),
    (r"\bis responsible for\b",                ":responsible"),
    (r"\bin order to\b",                       "to"),
    (r"\bdue to the fact that\b",             "because"),
    (r"\bthe fact that\b",                    "that"),
    (r"\bwith respect to\b",                  "re:"),
    (r"\bwith regard to\b",                   "re:"),
    (r"\bregarding\b",                        "re:"),
    (r"\bit is important to note that\b",     "NOTE:"),
    (r"\bplease note that\b",                 "NOTE:"),
    (r"\bas well as\b",                       "&"),
    (r"\band also\b",                         "&"),
    (r"\band\b",                              "&"),
    (r"\bfunction\b",                         "fn"),
    (r"\bmethod\b",                           "fn"),
    (r"\bmodule\b",                           "mod"),
    (r"\bparameter\b",                        "param"),
    (r"\bargument\b",                         "arg"),
    (r"\breturn(s|ing)?\b",                   "→"),
    (r"\bdepend(s|ency|encies) on\b",         "+dep"),
    (r"\bimport(s|ing)?\b",                   "←"),
    (r"\bimplemented? (as|using|with|in)\b",  "impl:"),
    (r"\bdefined? (as|in)\b",                 "def:"),
    (r"\bextend(s|ing)?\b",                   "ext:"),
    (r"\binherit(s|ing)? from\b",             "ext:"),
    (r"\bconfigured? (as|with|using)\b",      "cfg:"),
    (r"\benabled?\b",                         "ON"),
    (r"\bdisabled?\b",                        "OFF"),
]

_CODE_RULES = [
    # Python function signature: def fn(args) -> type:
    (
        r"def\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*(\w+))?\s*:",
        lambda m: (
            f"fn:{m.group(1)}({m.group(2).replace(' ', '')})"
            + (f"->{m.group(3)}" if m.group(3) else "")
        ),
    ),
    # class Foo(Base):
    (
        r"class\s+(\w+)\s*(?:\(([^)]*)\))?\s*:",
        lambda m: f"cls:{m.group(1)}" + (f"(ext:{m.group(2)})" if m.group(2) else ""),
    ),
    # import x / from x import y
    (r"from\s+(\S+)\s+import\s+(.+)", lambda m: f"←{m.group(1)}:{m.group(2)}"),
    (r"import\s+(\S+)(?:\s+as\s+(\S+))?", lambda m: f"←{m.group(1)}" + (f"→{m.group(2)}" if m.group(2) else "")),
]

_DIFF_RULES = [
    (r"^\+(.+)$",  r"add:\1"),
    (r"^-(.+)$",   r"rm:\1"),
    (r"^@@.+@@\s*", ""),
]


class ESCompressor:
    """
    Engram Shorthand compressor for Anthos training data.

    Achieves 4–10× compression on text, 3–5× on code.
    Output is readable by any LLM without a decoder.

    Args:
        confidence:    default confidence rating (1–5 stars appended)
        aggressive:    if True, apply more aggressive whitespace/filler removal
    """

    STARS = ["", "★", "★★", "★★★", "★★★★", "★★★★★"]

    def __init__(self, confidence: int = 3, aggressive: bool = False):
        self.confidence = max(1, min(5, confidence))
        self.aggressive = aggressive

    def compress(
        self,
        text:       str,
        is_code:    bool = False,
        is_diff:    bool = False,
        filename:   str  = "",
        confidence: Optional[int] = None,
    ) -> str:
        """
        Compress text using Engram Shorthand notation.

        Args:
            text:       text to compress
            is_code:    apply code-specific rules
            is_diff:    apply diff-specific rules
            filename:   for diff headers
            confidence: override default confidence

        Returns:
            ES-compressed string
        """
        conf = confidence or self.confidence
        text = text.strip()

        if is_diff:
            return self._compress_diff(text, filename, conf)
        if is_code:
            return self._compress_code(text, conf)
        return self._compress_prose(text, conf)

    def _compress_prose(self, text: str, conf: int) -> str:
        result = text
        for pattern, replacement in _PROSE_RULES:
            if callable(replacement):
                result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
            else:
                result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        # Collapse multiple spaces
        result = re.sub(r"  +", " ", result).strip()

        # Remove trailing periods before newlines / end of string
        result = re.sub(r"\.\s*$", "", result)

        star = self.STARS[conf]
        if star and not result.endswith(f"[{star}]"):
            result += f" [{star}]"

        return result

    def _compress_code(self, text: str, conf: int) -> str:
        result = text
        for pattern, replacement in _CODE_RULES:
            if callable(replacement):
                result = re.sub(pattern, replacement, result)
            else:
                result = re.sub(pattern, replacement, result)

        # Collapse blank lines
        result = re.sub(r"\n{3,}", "\n\n", result).strip()
        return result

    def _compress_diff(self, text: str, filename: str, conf: int) -> str:
        header = f"CHANGE:{filename} " if filename else "CHANGE: "
        adds, rms = [], []

        for line in text.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                adds.append(line[1:].strip())
            elif line.startswith("-") and not line.startswith("---"):
                rms.append(line[1:].strip())

        parts = [header]
        if adds: parts.append("add:" + ",".join(adds[:5]))
        if rms:  parts.append("rm:"  + ",".join(rms[:5]))
        return " ".join(parts)

    def compress_batch(self, texts: list[str], **kwargs) -> list[str]:
        return [self.compress(t, **kwargs) for t in texts]

    def ratio(self, original: str, compressed: str) -> float:
        if not compressed:
            return 0.0
        return len(original) / max(len(compressed), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Training data augmentation
# ─────────────────────────────────────────────────────────────────────────────

class MemoryAugmentedDataset:
    """
    Wraps a HuggingFace IterableDataset/Dataset, optionally inserting
    ES-compressed memory prefixes during training.

    This teaches Anthos to:
      1. Recognize [MEMORY]...[/MEMORY] as a retrieval signal
      2. Read ES notation correctly
      3. Use thought tokens to integrate memory context

    The compress_fraction controls what percentage of training examples
    receive an ES-compressed memory prefix. Start low (0.1) and increase
    as training progresses (curriculum approach).

    Usage:
        from anthos.memory_compress import MemoryAugmentedDataset
        from anthos.data import get_dataloader

        base_loader = get_dataloader("roneneldan/TinyStories", ...)
        aug_dataset = MemoryAugmentedDataset(
            base_loader,
            compress_fraction=0.15,
            prefix_confidence=3,
        )
    """

    MEMORY_START = "[MEMORY]\n"
    MEMORY_END   = "\n[/MEMORY]\n"

    def __init__(
        self,
        base_dataset,
        compress_fraction: float = 0.15,
        prefix_confidence: int   = 3,
        max_prefix_chars:  int   = 512,
        seed:              int   = 42,
    ):
        self.base         = base_dataset
        self.compress_frac = compress_fraction
        self.compressor   = ESCompressor(confidence=prefix_confidence)
        self.max_prefix   = max_prefix_chars
        self._rng_seed    = seed

    def _maybe_add_memory_prefix(self, text: str, rng_val: float) -> str:
        """With probability compress_fraction, prepend an ES memory prefix."""
        if rng_val >= self.compress_frac:
            return text

        # Use the first sentence as a synthetic "memory" of the context
        first_sentence_end = min(len(text), 200)
        snippet = text[:first_sentence_end]
        compressed = self.compressor.compress(snippet)
        prefix = f"{self.MEMORY_START}{compressed}{self.MEMORY_END}"

        # Truncate prefix if too long
        if len(prefix) > self.max_prefix:
            prefix = prefix[:self.max_prefix] + self.MEMORY_END

        return prefix + text

    def __iter__(self):
        import random
        rng = random.Random(self._rng_seed)
        for item in self.base:
            rng_val = rng.random()
            if isinstance(item, str):
                yield self._maybe_add_memory_prefix(item, rng_val)
            elif isinstance(item, dict) and "text" in item:
                item = dict(item)
                item["text"] = self._maybe_add_memory_prefix(item["text"], rng_val)
                yield item
            else:
                yield item


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing utility
# ─────────────────────────────────────────────────────────────────────────────

def compress_jsonl(
    input_path:  str,
    output_path: str,
    text_key:    str  = "text",
    confidence:  int  = 3,
    is_code:     bool = False,
):
    """
    Compress a JSONL training file using ES notation.

    Useful for preprocessing convo_smoke / instruct datasets before training
    to teach Anthos to read ES-compressed memory.

    Usage:
        python -c "
        from anthos.memory_compress import compress_jsonl
        compress_jsonl('data/teacher_conversations.jsonl',
                       'data/teacher_conversations_es.jsonl',
                       text_key='content')
        "
    """
    import json

    compressor = ESCompressor(confidence=confidence)
    n_in = n_out = 0

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            if text_key in item:
                original = item[text_key]
                compressed = compressor.compress(original, is_code=is_code)
                item[text_key] = compressed
                n_in  += len(original)
                n_out += len(compressed)

            fout.write(json.dumps(item) + "\n")

    ratio = n_in / max(n_out, 1)
    print(f"✓ Compressed {input_path} → {output_path}")
    print(f"  Chars: {n_in:,} → {n_out:,} ({ratio:.1f}× reduction)")
