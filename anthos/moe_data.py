"""
anthos/moe_data.py — Interleaved Streaming Dataset for MoE Training

Streams and interleaves three domain-diverse datasets to keep the MoE router
specializing across different token types:

    50% FineWeb-Edu       — educational web text, broad knowledge foundation
    25% MetaMathQA        — step-by-step math reasoning (trains logical experts)
    25% StarCoderData     — source code across many languages (trains code experts)

MoE routers only learn to specialize when training data is domain-diverse.
Feeding a single-domain corpus causes routing collapse (all tokens go to the
same 1-2 experts, wasting the other 62). This interleaved loader prevents that.

Usage:
    from anthos.moe_data import get_moe_dataloader

    loader = get_moe_dataloader(
        tokenizer_path="gpt2",
        seq_len=2048,
        batch_size=4,
        hf_token=os.environ.get("HF_TOKEN"),
    )

    for batch in loader:
        input_ids = batch[:, :-1]
        labels    = batch[:, 1:]
        ...

Requirements:
    pip install datasets

Environment:
    HF_TOKEN — HuggingFace token (required for StarCoder; recommended for others)
"""

from __future__ import annotations

import os
from typing import Iterator, Optional

import torch
from torch.utils.data import DataLoader, IterableDataset


class InterleavedMoeStream(IterableDataset):
    """
    Streams FineWeb-Edu, MetaMathQA, and StarCoderData from HuggingFace,
    interleaved at 50/25/25 ratio into fixed-length token chunks.

    Each yielded tensor is shape (seq_len + 1,) — use input[:seq_len] as
    input_ids and input[1:] as labels for causal language modeling.

    Documents are separated by the tokenizer's EOS token so attention does
    not bleed across document boundaries in a packed sequence.
    """

    _DATASETS = {
        "web":  ("HuggingFaceFW/fineweb-edu",  "sample-10BT", "train"),
        "math": ("meta-math/MetaMathQA",        None,          "train"),
        "code": ("bigcode/starcoderdata",        None,          "train"),
    }
    _PROBABILITIES = [0.50, 0.25, 0.25]

    def __init__(
        self,
        tokenizer_path: str            = "gpt2",
        seq_len:        int            = 2048,
        hf_token:       Optional[str]  = None,
    ):
        super().__init__()
        self.seq_len   = seq_len
        self.hf_token  = hf_token or os.environ.get("HF_TOKEN")

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.eos_id    = self.tokenizer.eos_token_id or 0

    # ── text extractors per dataset ──────────────────────────────────────────

    @staticmethod
    def _text_web(example: dict) -> str:
        return example.get("text", "")

    @staticmethod
    def _text_math(example: dict) -> str:
        q = example.get("query", "")
        a = example.get("response", "")
        return f"Question: {q}\n\nSolution: {a}" if q else ""

    @staticmethod
    def _text_code(example: dict) -> str:
        return example.get("content", "")

    # ── streaming interleave ─────────────────────────────────────────────────

    def __iter__(self) -> Iterator[torch.Tensor]:
        try:
            from datasets import load_dataset, interleave_datasets
        except ImportError:
            raise ImportError(
                "pip install datasets  # required for InterleavedMoeStream"
            )

        load_kw = {"streaming": True}
        if self.hf_token:
            load_kw["token"] = self.hf_token

        streams = []
        extractors = [self._text_web, self._text_math, self._text_code]

        for key, extractor in zip(["web", "math", "code"], extractors):
            name, subset, split = self._DATASETS[key]
            kw = dict(load_kw)
            if subset:
                kw["name"] = subset
            ds = load_dataset(name, split=split, **kw)
            _ext = extractor  # capture in closure
            ds = ds.map(
                lambda ex, e=_ext: {"_text": e(ex)},
                remove_columns=ds.column_names,
            )
            streams.append(ds)

        combined = interleave_datasets(
            streams,
            probabilities=self._PROBABILITIES,
            stopping_strategy="all_exhausted",
        )

        target = self.seq_len + 1
        buffer: list[int] = []

        for example in combined:
            text = example.get("_text", "")
            if not text:
                continue
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)
            buffer.append(self.eos_id)

            while len(buffer) >= target:
                chunk  = buffer[:target]
                buffer = buffer[target:]
                yield torch.tensor(chunk, dtype=torch.long)


def get_moe_dataloader(
    tokenizer_path: str           = "gpt2",
    seq_len:        int           = 2048,
    batch_size:     int           = 4,
    num_workers:    int           = 0,
    hf_token:       Optional[str] = None,
) -> DataLoader:
    """
    Returns a DataLoader of shape (batch_size, seq_len + 1) long tensors.

    Slice [: , :-1] for input_ids and [:, 1:] for labels.

    Args:
        tokenizer_path  — HuggingFace tokenizer name or local path
        seq_len         — target sequence length (chunk size before +1 label offset)
        batch_size      — number of sequences per batch
        num_workers     — DataLoader worker processes (0 = main process, safe on Mac)
        hf_token        — HuggingFace access token for gated datasets
    """
    ds = InterleavedMoeStream(
        tokenizer_path=tokenizer_path,
        seq_len=seq_len,
        hf_token=hf_token,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(num_workers > 0),
    )
