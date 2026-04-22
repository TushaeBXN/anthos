"""
Anthos — Data pipeline

Streams TinyStories (smoke/proof) or FineWeb-Edu (research) through
a GPT-2 tokenizer, packs sequences to fill the full context window,
and yields (input_ids, labels) batches ready for training.

Packing strategy:
  Rather than padding short stories to seq_len (wasteful), we concatenate
  documents separated by <|endoftext|> and slice into fixed-length chunks.
  Every token in every batch is a real token — no padding masks needed.
"""

from __future__ import annotations
from typing import Iterator
import torch
from torch.utils.data import IterableDataset, DataLoader


class PackedTextDataset(IterableDataset):
    """
    Streams a HuggingFace text dataset, tokenizes with GPT-2, and yields
    packed (seq_len+1,) token tensors — no padding, no waste.

    The +1 allows input = tokens[:-1] and labels = tokens[1:] in one slice.
    """

    def __init__(
        self,
        dataset_name:  str,
        split:         str,
        seq_len:       int,
        text_field:    str = "text",
        subset:        str | None = None,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.split        = split
        self.seq_len      = seq_len
        self.text_field   = text_field
        self.subset       = subset

        # GPT-2 tokenizer — 50257 vocab, no padding needed
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained("gpt2")
        self.eos = self.tok.eos_token_id   # 50256 — document separator

    def __iter__(self) -> Iterator[torch.Tensor]:
        from datasets import load_dataset

        ds = load_dataset(
            self.dataset_name,
            self.subset,
            split=self.split,
            streaming=True,
        )

        buffer: list[int] = []
        target = self.seq_len + 1   # +1 so we can split into input/labels

        for example in ds:
            text   = example.get(self.text_field, "")
            tokens = self.tok.encode(text, add_special_tokens=False)
            buffer.extend(tokens)
            buffer.append(self.eos)   # document boundary

            while len(buffer) >= target:
                chunk  = buffer[:target]
                buffer = buffer[target:]
                yield torch.tensor(chunk, dtype=torch.long)

    def __len__(self):
        # Streaming — length unknown; return a large sentinel
        return 10_000_000


def get_dataloader(
    dataset_name: str,
    split:        str,
    seq_len:      int,
    batch_size:   int,
    num_workers:  int = 0,
    subset:       str | None = None,
) -> DataLoader:
    """
    Returns a DataLoader that yields (B, seq_len+1) tensors.
    Call input_ids = batch[:, :-1] and labels = batch[:, 1:] in the train loop.
    """
    ds = PackedTextDataset(
        dataset_name = dataset_name,
        split        = split,
        seq_len      = seq_len,
        subset       = subset,
    )
    return DataLoader(
        ds,
        batch_size  = batch_size,
        num_workers = num_workers,
        pin_memory  = False,
    )
