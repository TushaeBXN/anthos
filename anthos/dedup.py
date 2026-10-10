"""
anthos/dedup.py — MinHash Near-Duplicate Deduplication

Removes near-duplicate documents from pretraining corpora using MinHash + LSH
(Locality-Sensitive Hashing). Near-duplicates in training data cause:
  - Model memorisation of training text
  - Artificially inflated benchmark scores (test set contamination)
  - Wasted compute training on semantically identical content

How it works:
  1. Tokenise each document into character n-grams (shingles)
  2. Compute a MinHash signature: k independent hash functions → k min values
  3. Group documents by their signature (LSH buckets)
  4. Within each bucket, compute exact Jaccard similarity
  5. Remove duplicates above a similarity threshold

MinHash parameters control the speed/accuracy tradeoff:
  - n_permutations: more = more accurate but slower (128 is standard)
  - ngram_size: char n-gram width (5 is standard for byte-level shingles)
  - threshold: Jaccard threshold; 0.8 removes near-dups, 0.5 removes rough dups
  - n_bands / n_rows: LSH parameters; bands × rows = n_permutations

Usage:
    deduplicator = MinHashDeduplicator(threshold=0.8, n_permutations=128)
    deduplicator.fit(["data/train.jsonl", "data/pretrain.jsonl"])
    deduplicator.write_deduped("data/train.jsonl", "data/train_deduped.jsonl")

    # Or filter in-memory:
    clean_docs = deduplicator.filter_texts(list_of_strings)
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# MinHash implementation (no external dependencies beyond numpy)
# ─────────────────────────────────────────────────────────────────────────────

_MERSENNE_PRIME = (1 << 61) - 1
_MAX_HASH       = (1 << 32) - 1


class MinHash:
    """
    MinHash signature for a single document.

    Implements the "one permutation" MinHash with k independent hash functions
    using universal hashing: h(x) = (a*x + b) mod p  where p is a Mersenne prime.
    """

    def __init__(self, n_permutations: int = 128, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.n = n_permutations
        # Coefficients for k independent hash functions
        self.a = rng.randint(1, _MERSENNE_PRIME, n_permutations, dtype=np.int64)
        self.b = rng.randint(0, _MERSENNE_PRIME, n_permutations, dtype=np.int64)
        self.hashvalues = np.full(n_permutations, _MAX_HASH, dtype=np.uint64)

    def update(self, shingles: Iterable[bytes]) -> "MinHash":
        a, b = self.a, self.b
        for s in shingles:
            # Map shingle to int using sha1 (fast, not cryptographic)
            x = int.from_bytes(hashlib.sha1(s).digest()[:4], "big")
            # Apply k hash functions
            hv = (a * x + b) % _MERSENNE_PRIME % (_MAX_HASH + 1)
            self.hashvalues = np.minimum(self.hashvalues, hv.astype(np.uint64))
        return self

    def jaccard(self, other: "MinHash") -> float:
        return float(np.mean(self.hashvalues == other.hashvalues))


def _shingle(text: str, n: int = 5) -> list[bytes]:
    """Character n-gram shingles as bytes."""
    text_b = text.encode("utf-8", errors="replace")
    return [text_b[i : i + n] for i in range(max(1, len(text_b) - n + 1))]


# ─────────────────────────────────────────────────────────────────────────────
# LSH bucketing
# ─────────────────────────────────────────────────────────────────────────────

class LSHIndex:
    """
    Locality-Sensitive Hashing index for MinHash signatures.

    Partitions the n_permutations hash values into `n_bands` bands of
    `n_rows` each. Two documents are candidate duplicates if any band's
    rows hash to the same bucket.

    The false-negative probability is approximately (1 - threshold^n_rows)^n_bands.
    For threshold=0.8, n_bands=32, n_rows=4: FN ≈ 0.002.
    """

    def __init__(self, n_bands: int = 32, n_rows: int = 4):
        self.n_bands = n_bands
        self.n_rows  = n_rows
        self.buckets: list[dict[bytes, list[int]]] = [{} for _ in range(n_bands)]
        self.signatures: list[np.ndarray] = []

    def add(self, mh: MinHash, doc_id: int) -> None:
        self.signatures.append(mh.hashvalues.copy())
        for b in range(self.n_bands):
            band_vals = mh.hashvalues[b * self.n_rows : (b + 1) * self.n_rows]
            key = struct.pack(f"{self.n_rows}Q", *band_vals.tolist())
            self.buckets[b].setdefault(key, []).append(doc_id)

    def candidates(self, doc_id: int) -> set[int]:
        """Return all doc ids that share at least one LSH bucket with doc_id."""
        sig = self.signatures[doc_id]
        result: set[int] = set()
        for b in range(self.n_bands):
            band_vals = sig[b * self.n_rows : (b + 1) * self.n_rows]
            key = struct.pack(f"{self.n_rows}Q", *band_vals.tolist())
            for cand_id in self.buckets[b].get(key, []):
                if cand_id != doc_id:
                    result.add(cand_id)
        return result

    def exact_jaccard(self, id_a: int, id_b: int) -> float:
        a, b = self.signatures[id_a], self.signatures[id_b]
        return float(np.mean(a == b))


# ─────────────────────────────────────────────────────────────────────────────
# High-level deduplicator
# ─────────────────────────────────────────────────────────────────────────────

class MinHashDeduplicator:
    """
    MinHash + LSH deduplicator for Anthos pretraining data.

    Workflow:
        1. dedup.fit(paths)           — build the index from one or more JSONL files
        2. dedup.write_deduped(...)   — write a deduped version of a file
        3. dedup.filter_texts(texts)  — filter an in-memory list of strings

    JSONL files are expected to have a "text" field (or specify text_key).
    """

    def __init__(
        self,
        threshold:      float = 0.8,
        n_permutations: int   = 128,
        ngram_size:     int   = 5,
        seed:           int   = 42,
    ):
        self.threshold      = threshold
        self.n_permutations = n_permutations
        self.ngram_size     = ngram_size
        self.seed           = seed

        n_bands = 32
        n_rows  = n_permutations // n_bands
        self.index  = LSHIndex(n_bands=n_bands, n_rows=n_rows)
        self.texts:  list[str]  = []
        self.kept:   list[bool] = []

    def _minhash(self, text: str) -> MinHash:
        mh = MinHash(self.n_permutations, self.seed)
        mh.update(_shingle(text, self.ngram_size))
        return mh

    def fit(
        self,
        paths:    list[str],
        text_key: str = "text",
        verbose:  bool = True,
    ) -> "MinHashDeduplicator":
        """
        Index all documents from a list of JSONL files.
        Call this before filter_texts or write_deduped.
        """
        self.texts = []
        for path in paths:
            with open(path) as f:
                for line in f:
                    try:
                        obj  = json.loads(line)
                        text = obj.get(text_key, "") or ""
                    except Exception:
                        text = line.strip()
                    self.texts.append(text)

        if verbose:
            print(f"[dedup] Indexing {len(self.texts):,} documents...", flush=True)

        for doc_id, text in enumerate(self.texts):
            mh = self._minhash(text)
            self.index.add(mh, doc_id)
            if verbose and (doc_id + 1) % 50_000 == 0:
                print(f"  indexed {doc_id+1:,}/{len(self.texts):,}", flush=True)

        # Greedy deduplication: process documents in order; mark as duplicate
        # if any already-kept document has Jaccard >= threshold.
        self.kept = [True] * len(self.texts)
        n_removed = 0
        for doc_id in range(len(self.texts)):
            if not self.kept[doc_id]:
                continue
            for cand_id in self.index.candidates(doc_id):
                if not self.kept[cand_id]:
                    continue
                if cand_id <= doc_id:
                    continue   # only compare forward
                j = self.index.exact_jaccard(doc_id, cand_id)
                if j >= self.threshold:
                    self.kept[cand_id] = False
                    n_removed += 1

        n_kept = sum(self.kept)
        if verbose:
            print(f"[dedup] Done. {n_kept:,} kept, {n_removed:,} removed "
                  f"({n_removed / max(len(self.texts), 1):.1%} duplication rate)")

        return self

    def filter_texts(self, texts: list[str]) -> list[str]:
        """
        Deduplicate an in-memory list of strings.
        Returns only the non-duplicate texts.
        """
        index  = LSHIndex(n_bands=32, n_rows=self.n_permutations // 32)
        kept   = [True] * len(texts)
        for doc_id, text in enumerate(texts):
            mh = self._minhash(text)
            index.add(mh, doc_id)

        for doc_id in range(len(texts)):
            if not kept[doc_id]:
                continue
            for cand_id in index.candidates(doc_id):
                if not kept[cand_id] or cand_id <= doc_id:
                    continue
                if index.exact_jaccard(doc_id, cand_id) >= self.threshold:
                    kept[cand_id] = False

        return [t for t, k in zip(texts, kept) if k]

    def write_deduped(
        self,
        input_path:  str,
        output_path: str,
        text_key:    str = "text",
        verbose:     bool = True,
    ) -> int:
        """
        Write a deduplicated version of a JSONL file.
        fit() must have been called first with this file included.

        Returns the number of kept documents written.
        """
        if not self.texts:
            raise RuntimeError("Call fit() before write_deduped()")

        input_path  = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        n_written = 0
        doc_id    = 0
        with open(input_path)  as fin, \
             open(output_path, "w") as fout:
            for line in fin:
                if doc_id < len(self.kept) and self.kept[doc_id]:
                    fout.write(line)
                    n_written += 1
                doc_id += 1

        if verbose:
            print(f"[dedup] Wrote {n_written:,} docs → {output_path}")
        return n_written
