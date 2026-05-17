"""
anthos/stream_fineweb.py — Stream FineWeb-Edu directly with zero disk usage

Usage:
    from anthos.stream_fineweb import StreamingFineWebDataset
    dataset = StreamingFineWebDataset(tokenizer, max_samples=10_000_000)
    for batch in dataset:
        train_on(batch)

Requires:
    pip install datasets
    HF_TOKEN env var (free HuggingFace account)
"""

import os
from typing import Optional
from torch.utils.data import IterableDataset


class StreamingFineWebDataset(IterableDataset):
    """
    Streams FineWeb-Edu directly from HuggingFace — no download, no disk space.
    Applies quality filtering on the fly.
    """

    # Text that indicates low-quality web content
    LOW_QUALITY_SIGNALS = [
        "lorem ipsum", "click here", "subscribe now",
        "cookie policy", "privacy policy", "terms of service",
        "all rights reserved", "404 not found",
    ]

    def __init__(
        self,
        tokenizer,
        max_seq_len:   int = 4096,
        max_samples:   Optional[int] = None,
        min_text_len:  int = 200,
        max_text_len:  int = 32000,
        hf_token:      Optional[str] = None,
    ):
        self.tokenizer    = tokenizer
        self.max_seq_len  = max_seq_len
        self.max_samples  = max_samples
        self.min_text_len = min_text_len
        self.max_text_len = max_text_len
        self.hf_token     = hf_token or os.environ.get("HF_TOKEN")

    def _load_stream(self):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets")

        kwargs = {}
        if self.hf_token:
            kwargs["token"] = self.hf_token

        return load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
            **kwargs,
        )

    def _is_quality(self, text: str) -> bool:
        if len(text) < self.min_text_len or len(text) > self.max_text_len:
            return False
        text_lower = text.lower()
        return not any(sig in text_lower for sig in self.LOW_QUALITY_SIGNALS)

    def _tokenize(self, text: str) -> dict:
        tokens = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"][0]
        return {
            "input_ids":      input_ids,
            "attention_mask": tokens["attention_mask"][0],
            "labels":         input_ids.clone(),  # causal LM
        }

    def __iter__(self):
        stream = self._load_stream()
        count  = 0

        for example in stream:
            if self.max_samples and count >= self.max_samples:
                break

            text = example.get("text", "")
            if not self._is_quality(text):
                continue

            try:
                yield self._tokenize(text)
                count += 1
            except Exception:
                continue
