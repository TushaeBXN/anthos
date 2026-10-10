"""
data_mix.py — Domain-weighted data mixing for pretraining

Implements dynamic data mixing: blend multiple domain datasets with
configurable weights, with optional DoReMi-style adaptive reweighting
based on per-domain validation loss.

Two modes:
  Static mixing — fixed weights, simple and reproducible
  Adaptive mixing — weights updated every N steps via excess loss signal
                    (simplified DoReMi without the separate reference model)

Domain datasets can be:
  - JSONL files with a "text" field
  - Plain .txt files (one document per line)
  - WebDataset shards (.tar)

Usage (static):
    from anthos.data_mix import DataMixer, DomainConfig

    domains = [
        DomainConfig("data/web.jsonl",  weight=0.70, name="web"),
        DomainConfig("data/code.jsonl", weight=0.15, name="code"),
        DomainConfig("data/books.jsonl", weight=0.10, name="books"),
        DomainConfig("data/math.jsonl",  weight=0.05, name="math"),
    ]
    mixer = DataMixer(domains, tokenizer_path="data/anthos_tokenizer", seq_len=2048)
    for batch in mixer:
        # batch: (B, seq_len) token ids
        ...

Usage (adaptive):
    mixer = DataMixer(domains, ..., adaptive=True, adapt_every=1000)
    for step, batch in enumerate(mixer):
        loss_by_domain = ...   # track per-domain CE loss
        mixer.update_weights(loss_by_domain)
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import torch


@dataclass
class DomainConfig:
    path:   str
    weight: float       # initial sampling weight (will be normalized)
    name:   str  = ""
    text_key: str = "text"   # JSON field name for text content


class _DomainBuffer:
    """Reads a domain file lazily and yields tokenized chunks."""

    def __init__(self, cfg: DomainConfig, tokenizer, seq_len: int):
        self.cfg       = cfg
        self.tok       = tokenizer
        self.seq_len   = seq_len
        self._buf: list[int] = []
        self._iter = self._file_iter()

    def _file_iter(self) -> Iterator[str]:
        path = Path(self.cfg.path)
        suffix = path.suffix.lower()
        while True:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if suffix == ".jsonl":
                        try:
                            obj = json.loads(line)
                            yield obj.get(self.cfg.text_key, "")
                        except json.JSONDecodeError:
                            continue
                    else:
                        yield line

    def next_chunk(self) -> Optional[torch.Tensor]:
        while len(self._buf) < self.seq_len + 1:
            try:
                text = next(self._iter)
                self._buf.extend(self.tok.encode(text))
                self._buf.append(self.tok.eos_token_id or 0)
            except StopIteration:
                self._iter = self._file_iter()   # wrap around

        chunk = torch.tensor(self._buf[:self.seq_len + 1], dtype=torch.long)
        self._buf = self._buf[self.seq_len + 1:]
        return chunk


class DataMixer:
    """
    Yields mixed batches from multiple domain datasets.

    Args:
        domains:         list of DomainConfig objects
        tokenizer_path:  path to Anthos tokenizer (or "gpt2")
        seq_len:         token sequence length per sample
        batch_size:      samples per batch
        adaptive:        enable adaptive weight update (simplified DoReMi)
        adapt_every:     steps between weight updates
        adapt_lr:        learning rate for weight update (exponentiated gradient)
    """

    def __init__(
        self,
        domains:        list[DomainConfig],
        tokenizer_path: str,
        seq_len:        int   = 2048,
        batch_size:     int   = 8,
        adaptive:       bool  = False,
        adapt_every:    int   = 1000,
        adapt_lr:       float = 0.1,
    ):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        self.domains     = domains
        self.seq_len     = seq_len
        self.batch_size  = batch_size
        self.adaptive    = adaptive
        self.adapt_every = adapt_every
        self.adapt_lr    = adapt_lr

        total = sum(d.weight for d in domains)
        self.weights = [d.weight / total for d in domains]

        self.buffers = [_DomainBuffer(d, self.tok, seq_len) for d in domains]
        self._step   = 0
        self._domain_losses: dict[str, float] = {d.name: 0.0 for d in domains}

    def _sample_domain(self) -> int:
        r = random.random()
        cumsum = 0.0
        for i, w in enumerate(self.weights):
            cumsum += w
            if r < cumsum:
                return i
        return len(self.weights) - 1

    def __iter__(self) -> Iterator[torch.Tensor]:
        return self

    def __next__(self) -> torch.Tensor:
        chunks = []
        for _ in range(self.batch_size):
            d_idx = self._sample_domain()
            chunk = self.buffers[d_idx].next_chunk()
            chunks.append(chunk)
        self._step += 1
        return torch.stack(chunks, dim=0)   # (B, seq_len+1)

    def update_weights(self, domain_losses: dict[str, float]) -> None:
        """
        Adaptive reweighting via exponentiated gradient on excess loss.

        Args:
            domain_losses: {domain_name: mean_ce_loss} from current batch

        Call every adapt_every steps with per-domain CE losses.
        Domains with higher-than-average loss get more weight (harder).
        """
        if not self.adaptive:
            return

        names  = [d.name for d in self.domains]
        losses = [domain_losses.get(n, 0.0) for n in names]
        avg    = sum(losses) / max(len(losses), 1)

        # Excess loss = domain loss - average; positive → harder than average
        excess = [l - avg for l in losses]

        # Exponentiated gradient update on log weights
        import math
        log_w = [math.log(max(w, 1e-9)) for w in self.weights]
        log_w = [lw + self.adapt_lr * ex for lw, ex in zip(log_w, excess)]

        # Normalize
        max_lw = max(log_w)
        exp_w  = [math.exp(lw - max_lw) for lw in log_w]
        total  = sum(exp_w)
        self.weights = [ew / total for ew in exp_w]

    def get_weights(self) -> dict[str, float]:
        return {d.name: w for d, w in zip(self.domains, self.weights)}
