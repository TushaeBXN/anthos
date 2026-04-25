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


class LocalPackedTextDataset(IterableDataset):
    """
    Reads a local plain-text file where stories are separated by blank lines,
    tokenizes with GPT-2, and yields packed (seq_len+1,) tensors.

    Loops over the file indefinitely so training can run for any number of steps.

    Args:
        path     : path to the .txt file (e.g. "data/ethnic_stories.txt")
        seq_len  : sequence length (model context window)
    """

    def __init__(self, path: str, seq_len: int):
        super().__init__()
        self.path    = path
        self.seq_len = seq_len

        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained("gpt2")
        self.eos = self.tok.eos_token_id

    def __iter__(self) -> Iterator[torch.Tensor]:
        target = self.seq_len + 1
        buffer: list[int] = []

        with open(self.path, encoding="utf-8") as f:
            raw = f.read()

        # Stories separated by double newline
        stories = [s.strip() for s in raw.split("\n\n") if s.strip()]

        # Loop indefinitely so training never runs out of data
        while True:
            for story in stories:
                tokens = self.tok.encode(story, add_special_tokens=False)
                buffer.extend(tokens)
                buffer.append(self.eos)

                while len(buffer) >= target:
                    chunk  = buffer[:target]
                    buffer = buffer[target:]
                    yield torch.tensor(chunk, dtype=torch.long)

    def __len__(self):
        return 10_000_000   # streaming sentinel


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

    If dataset_name is a path to a local .txt file, uses LocalPackedTextDataset.
    Otherwise streams from HuggingFace.
    """
    import os
    if os.path.isfile(dataset_name):
        ds = LocalPackedTextDataset(path=dataset_name, seq_len=seq_len)
    else:
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
