"""
anthos/multipack.py — Multipack Sequence Packing

Eliminates padding waste by binning variable-length sequences into fixed-length
chunks using a first-fit-decreasing bin-packing algorithm.

Why this matters for Anthos:
  With only 6 markdown files, naive batching produces heavily padded sequences.
  Every padding token is a wasted forward + backward pass.  Multipack bins
  sequences together so each batch chunk is ~100% utilised.

  The key challenge: attention must not leak across packed sequences.
  We solve this with a per-sample block-diagonal attention mask injected
  into the AnthosRecurrentBlock's combined_mask.  Sequence tokens see their
  own document only; thought tokens see all thought tokens in the pack
  (they're scratch-pad, not content — cross-doc thought attention is fine).

Integration with train.py:
  Replace your dataset/dataloader with:

      from anthos.multipack import MultipackDataset, multipack_collate

      dataset = MultipackDataset(
          file_paths  = your_markdown_files,
          tokenizer   = tokenizer,
          chunk_len   = cfg.max_seq_len,         # e.g. 4096
          max_samples = cfg.max_loop_iters * 100, # optional cap
      )
      loader = DataLoader(
          dataset,
          batch_size  = 1,                       # packing handles density
          collate_fn  = multipack_collate,
          shuffle     = True,
          num_workers = 0,
      )

  Each batch item is already chunk_len tokens long with a `seq_ids` field
  that tells the attention mask builder which positions belong to which doc.
  Pass `seq_ids` to build_pack_mask() to get the block-diagonal mask.

Thought token behaviour in packed batches:
  Thought tokens prepended by AnthosRecurrentBlock see ALL positions in the
  chunk (their mask rows are all-zero).  This is intentional — thought tokens
  aggregate global reasoning state and are discarded before output, so
  cross-document thought attention does not contaminate the sequence stream.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import torch
from torch.utils.data import Dataset, Sampler


# ─────────────────────────────────────────────────────────────────────────────
# Tokenised sample
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TokenisedDoc:
    input_ids: List[int]   # raw token ids, no padding
    path:      str         # source file path (for debugging)


# ─────────────────────────────────────────────────────────────────────────────
# Packing utilities
# ─────────────────────────────────────────────────────────────────────────────

def _first_fit_decreasing(
    lengths: List[int],
    chunk_len: int,
    max_sequences_per_pack: int = 32,
) -> List[List[int]]:
    """
    Classic first-fit-decreasing bin packing.

    Args:
        lengths:  list of sequence lengths (one per document)
        chunk_len: target bin size (= model max_seq_len)

    Returns:
        List of bins, each bin is a list of document indices.
        Sequences longer than chunk_len are placed alone (will be truncated).
    """
    # Sort descending by length
    order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)

    bins:      List[List[int]] = []
    bin_sizes: List[int]       = []

    for idx in order:
        L = min(lengths[idx], chunk_len)  # we'll truncate at pack time
        placed = False
        for b_idx, b_size in enumerate(bin_sizes):
            if (b_size + L <= chunk_len
                    and len(bins[b_idx]) < max_sequences_per_pack):
                bins[b_idx].append(idx)
                bin_sizes[b_idx] += L
                placed = True
                break
        if not placed:
            bins.append([idx])
            bin_sizes.append(L)

    return bins


# ─────────────────────────────────────────────────────────────────────────────
# Attention mask for packed sequences
# ─────────────────────────────────────────────────────────────────────────────

def build_pack_mask(
    seq_ids:   torch.Tensor,   # (T,) integer, same value = same document
    n_thought: int,
    device:    torch.device,
) -> torch.Tensor:
    """
    Build the combined Anthos causal mask for a packed sequence.

    Layout: [thought₁…thoughtₙ | seq_pos₀…seq_posₜ]

    Thought rows  → attend to ALL thought tokens + ALL sequence positions
                    (thoughts are global scratch-pad, cross-doc is fine)
    Sequence rows → standard causal within same document
                  + full attention to thought tokens
                  + -inf to positions from OTHER documents

    Args:
        seq_ids:   (T,) tensor — each position's document id
        n_thought: number of thought tokens prepended
        device:    target device

    Returns:
        (1, 1, n_thought + T, n_thought + T) attention mask
    """
    T     = seq_ids.shape[0]
    total = n_thought + T
    mask  = torch.zeros(1, 1, total, total, device=device)

    # Sequence-to-sequence block
    for i in range(T):
        for j in range(T):
            if j > i:                          # future (causal)
                mask[0, 0, n_thought + i, n_thought + j] = float("-inf")
            elif seq_ids[j] != seq_ids[i]:     # different document
                mask[0, 0, n_thought + i, n_thought + j] = float("-inf")

    # Note: thought rows stay 0 (full attention everywhere)
    return mask


def build_pack_mask_fast(
    seq_ids:   torch.Tensor,   # (T,)
    n_thought: int,
    device:    torch.device,
) -> torch.Tensor:
    """
    Vectorised version of build_pack_mask — O(T²) but in tensor ops.
    Faster than the Python loop for T > 512.
    """
    T     = seq_ids.shape[0]
    total = n_thought + T
    mask  = torch.zeros(1, 1, total, total, device=device)

    # Positions: (T, T)
    i_idx = torch.arange(T, device=device).unsqueeze(1)   # (T, 1)
    j_idx = torch.arange(T, device=device).unsqueeze(0)   # (1, T)

    same_doc  = seq_ids.unsqueeze(1) == seq_ids.unsqueeze(0)  # (T, T)
    causal    = j_idx <= i_idx                                  # (T, T)
    allowed   = same_doc & causal                               # (T, T)

    seq_mask  = torch.where(allowed, torch.zeros_like(i_idx, dtype=torch.float),
                            torch.full_like(i_idx, float("-inf"), dtype=torch.float))

    mask[0, 0, n_thought:, n_thought:] = seq_mask
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class MultipackDataset(Dataset):
    """
    Packs tokenised documents into fixed-length chunks with minimal padding.

    Each __getitem__ returns a dict:
        input_ids : (chunk_len,)  padded/truncated token ids
        labels    : (chunk_len,)  same, but -100 on pad positions
        seq_ids   : (chunk_len,)  document id per position (0 = pad)
        attention_mask: (chunk_len,) 1 for real tokens, 0 for pad

    Usage:
        dataset = MultipackDataset(
            file_paths = [Path("data/new_history/essay1.md"), ...],
            tokenizer  = tokenizer,
            chunk_len  = 4096,
        )
    """

    def __init__(
        self,
        file_paths:   List[Path],
        tokenizer,
        chunk_len:    int  = 4096,
        stride:       int  = 0,        # overlap between chunks (0 = no overlap)
        seed:         int  = 42,
        max_doc_len:  Optional[int] = None,
    ):
        self.tokenizer  = tokenizer
        self.chunk_len  = chunk_len
        self.stride     = stride
        self.seed       = seed

        # Tokenise all documents
        self.docs: List[TokenisedDoc] = []
        for path in file_paths:
            text = Path(path).read_text(encoding="utf-8")
            ids  = tokenizer.encode(text)
            if max_doc_len:
                # Sliding window over long documents
                step = max_doc_len - stride if stride else max_doc_len
                for start in range(0, len(ids), step):
                    chunk = ids[start : start + max_doc_len]
                    if len(chunk) >= 16:   # skip tiny fragments
                        self.docs.append(TokenisedDoc(chunk, str(path)))
            else:
                if len(ids) >= 16:
                    self.docs.append(TokenisedDoc(ids, str(path)))

        # Pack documents into chunks
        lengths   = [len(d.input_ids) for d in self.docs]
        self.bins = _first_fit_decreasing(lengths, chunk_len)

        print(f"[MultipackDataset] {len(self.docs)} docs → {len(self.bins)} packed chunks")
        total_tokens   = sum(lengths)
        packed_capacity = len(self.bins) * chunk_len
        utilisation    = total_tokens / packed_capacity
        print(f"[MultipackDataset] Utilisation: {utilisation:.1%} "
              f"({total_tokens:,} tokens in {packed_capacity:,} capacity)")

    def __len__(self) -> int:
        return len(self.bins)

    def __getitem__(self, idx: int) -> dict:
        doc_indices = self.bins[idx]
        input_ids   = []
        seq_ids     = []
        doc_id      = 1   # 0 is reserved for padding

        for di in doc_indices:
            doc_tokens = self.docs[di].input_ids[: self.chunk_len - len(input_ids)]
            input_ids.extend(doc_tokens)
            seq_ids.extend([doc_id] * len(doc_tokens))
            doc_id += 1

        # Pad to chunk_len
        pad_len = self.chunk_len - len(input_ids)
        pad_id  = self.tokenizer.pad_token_id or 0

        input_ids_t      = torch.tensor(input_ids + [pad_id] * pad_len, dtype=torch.long)
        seq_ids_t        = torch.tensor(seq_ids   + [0]      * pad_len, dtype=torch.long)
        attention_mask_t = (seq_ids_t > 0).long()

        # Labels: shift by 1, mask padding and first token of each doc
        labels = input_ids_t.clone()
        labels[attention_mask_t == 0] = -100   # ignore pad positions

        return {
            "input_ids":      input_ids_t,
            "labels":         labels,
            "seq_ids":        seq_ids_t,
            "attention_mask": attention_mask_t,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Collate function
# ─────────────────────────────────────────────────────────────────────────────

def multipack_collate(samples: List[dict]) -> dict:
    """Stack MultipackDataset samples into a batch."""
    return {
        key: torch.stack([s[key] for s in samples])
        for key in samples[0].keys()
    }


# ─────────────────────────────────────────────────────────────────────────────
# Shuffling sampler
# ─────────────────────────────────────────────────────────────────────────────

class MultipackSampler(Sampler):
    """
    Shuffles packs (not documents) to prevent the model from memorising
    pack order.  Re-shuffles at the start of each epoch.
    """

    def __init__(self, dataset: MultipackDataset, seed: int = 42):
        self.n    = len(dataset)
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self) -> Iterator[int]:
        rng     = random.Random(self.seed + self._epoch)
        indices = list(range(self.n))
        rng.shuffle(indices)
        return iter(indices)

    def __len__(self) -> int:
        return self.n
