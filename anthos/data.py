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


class LocalMarkdownDataset(IterableDataset):
    """
    Reads every .md file in a directory, tokenizes with GPT-2, and yields
    packed (seq_len+1,) tensors — same contract as LocalPackedTextDataset.

    Loops indefinitely so training can run for any MAX_STEPS.
    Files are read in sorted order each epoch so output is deterministic.

    Markdown syntax (headers, bold markers, horizontal rules) is preserved
    by default — the model learns to read your document structure.
    Pass strip_markdown=True to train on clean prose only.

    Args:
        directory:      path to folder containing .md files
        seq_len:        context window (model SEQ_LEN)
        strip_markdown: if True, remove #/*/- markdown characters
    """

    def __init__(self, directory: str, seq_len: int, strip_markdown: bool = False):
        super().__init__()
        import glob, re
        self.seq_len      = seq_len
        self.strip_md     = strip_markdown

        paths = sorted(set(
            glob.glob(f"{directory}/**/*.md", recursive=True) +
            glob.glob(f"{directory}/*.md")
        ))
        if not paths:
            raise FileNotFoundError(f"No .md files found in {directory}")
        self.paths = paths
        print(f"  [LocalMarkdownDataset] Found {len(paths)} markdown files in {directory}")

        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained("gpt2")
        self.eos = self.tok.eos_token_id

    def _clean(self, text: str) -> str:
        """Optionally strip markdown syntax, always clean up whitespace."""
        import re
        if self.strip_md:
            text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)  # headers
            text = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", text)          # bold/italic
            text = re.sub(r"^---+\s*$", "", text, flags=re.MULTILINE)   # hr
            text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)        # blockquotes
        text = re.sub(r"\n{3,}", "\n\n", text)   # collapse excess blank lines
        return text.strip()

    def __iter__(self):
        target = self.seq_len + 1
        buffer: list[int] = []

        while True:                          # loop forever
            for path in self.paths:
                with open(path, encoding="utf-8") as f:
                    raw = f.read()

                text   = self._clean(raw)
                tokens = self.tok.encode(text, add_special_tokens=False)
                buffer.extend(tokens)
                buffer.append(self.eos)      # document boundary

                while len(buffer) >= target:
                    chunk  = buffer[:target]
                    buffer = buffer[target:]
                    yield torch.tensor(chunk, dtype=torch.long)

    def __len__(self):
        return 10_000_000   # streaming sentinel


def get_markdown_dataloader(
    directory:   str,
    seq_len:     int,
    batch_size:  int,
    num_workers: int  = 0,
    strip_markdown: bool = False,
) -> DataLoader:
    """
    Returns a DataLoader over all .md files in a directory.
    Yields (B, seq_len+1) tensors — same contract as get_dataloader().
    Train loop: input_ids = batch[:, :-1], labels = batch[:, 1:]
    """
    ds = LocalMarkdownDataset(
        directory      = directory,
        seq_len        = seq_len,
        strip_markdown = strip_markdown,
    )
    return DataLoader(
        ds,
        batch_size  = batch_size,
        num_workers = num_workers,
        pin_memory  = False,
    )


class AlpacaInstructDataset(IterableDataset):
    """
    Streams tatsu-lab/alpaca (52k instruction pairs) and formats each example
    into the standard Alpaca prompt template, then tokenizes and packs.

    Template:
        Below is an instruction that describes a task. Write a response that
        appropriately completes the request.

        ### Instruction:
        {instruction}

        ### Input:          ← omitted when empty
        {input}

        ### Response:
        {output}<|endoftext|>

    During training, loss is masked on the prompt tokens so the model only
    learns to predict the response — not to memorise the instruction format.

    Args:
        seq_len        : context window length
        mask_prompt    : if True, set prompt token labels to -100 (recommended)
        split          : HuggingFace dataset split (default "train")
    """

    SYSTEM = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
    )

    def __init__(
        self,
        seq_len:     int,
        mask_prompt: bool = True,
        split:       str  = "train",
    ):
        super().__init__()
        self.seq_len     = seq_len
        self.mask_prompt = mask_prompt
        self.split       = split

        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained("gpt2")
        self.eos = self.tok.eos_token_id

    def _format(self, instruction: str, inp: str, output: str) -> tuple[str, str]:
        """Returns (prompt_text, full_text) — prompt is the part to mask."""
        if inp.strip():
            prompt = (
                f"{self.SYSTEM}\n\n"
                f"### Instruction:\n{instruction.strip()}\n\n"
                f"### Input:\n{inp.strip()}\n\n"
                f"### Response:\n"
            )
        else:
            prompt = (
                f"{self.SYSTEM}\n\n"
                f"### Instruction:\n{instruction.strip()}\n\n"
                f"### Response:\n"
            )
        full = prompt + output.strip()
        return prompt, full

    def __iter__(self) -> Iterator[torch.Tensor]:
        from datasets import load_dataset

        ds = load_dataset("tatsu-lab/alpaca", split=self.split, streaming=True)
        target = self.seq_len + 1

        # Yield individual examples (no cross-document packing for instruct)
        for example in ds:
            instruction = example.get("instruction", "")
            inp         = example.get("input", "")
            output      = example.get("output", "")

            if not instruction or not output:
                continue

            prompt, full = self._format(instruction, inp, output)

            full_ids   = self.tok.encode(full,   add_special_tokens=False) + [self.eos]
            prompt_ids = self.tok.encode(prompt, add_special_tokens=False)
            prompt_len = len(prompt_ids)

            # Truncate to seq_len+1
            full_ids = full_ids[:target]
            if len(full_ids) < 2:
                continue

            tokens = torch.tensor(full_ids, dtype=torch.long)

            if self.mask_prompt:
                # Labels: -100 for prompt tokens (masked), real ids for response
                labels = tokens.clone()
                labels[:prompt_len] = -100
                # Yield (tokens, labels) tuple — train loop must handle this
                yield tokens, labels
            else:
                yield tokens, tokens.clone()

    def __len__(self):
        return 52_000   # Alpaca has ~52k examples


def get_instruct_dataloader(
    seq_len:     int,
    batch_size:  int,
    num_workers: int  = 0,
    mask_prompt: bool = True,
    split:       str  = "train",
) -> DataLoader:
    """
    Returns a DataLoader for Alpaca instruction tuning.
    Batches are (input_ids, labels) pairs where prompt tokens are masked to -100.
    """
    ds = AlpacaInstructDataset(
        seq_len     = seq_len,
        mask_prompt = mask_prompt,
        split       = split,
    )

    def collate(batch):
        # Pad to longest in batch, labels pad with -100
        tokens_list = [b[0] for b in batch]
        labels_list = [b[1] for b in batch]
        max_len = max(t.size(0) for t in tokens_list)

        padded_tokens = torch.zeros(len(batch), max_len, dtype=torch.long)
        padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

        for i, (t, l) in enumerate(zip(tokens_list, labels_list)):
            padded_tokens[i, :t.size(0)] = t
            padded_labels[i, :l.size(0)] = l

        return padded_tokens, padded_labels

    return DataLoader(
        ds,
        batch_size  = batch_size,
        num_workers = num_workers,
        pin_memory  = False,
        collate_fn  = collate,
    )


class ChatInstructDataset(IterableDataset):
    """
    Streams Open-Orca/SlimOrca and formats conversations using Anthos
    custom chat tokens:

        <|system|>
        {system_prompt}<|end|>
        <|user|>
        {question}<|end|>
        <|thought|>
        [thought stream — model reasons here before speaking]<|end|>
        <|assistant|>
        {response}<|end|>

    Loss is masked on everything except the assistant response,
    so the model only learns to generate answers — not mimic prompts.

    The <|thought|> block is empty during SFT (the model learns to fill it
    during RLHF). Even empty, it reserves space for the thought stream.

    Args:
        tokenizer_path : path to saved Anthos tokenizer (data/anthos_tokenizer)
        seq_len        : context window
        mask_prompt    : mask system+user+thought tokens from loss (recommended)
        split          : dataset split
    """

    def __init__(
        self,
        tokenizer_path: str  = "data/anthos_tokenizer",
        seq_len:        int  = 512,
        mask_prompt:    bool = True,
        split:          str  = "train",
        max_samples:    int  = 0,    # 0 = use full dataset; N = stop after N conversations
        dataset_name:   str  = "Open-Orca/SlimOrca",  # swap to any ShareGPT-format dataset
    ):
        super().__init__()
        self.seq_len      = seq_len
        self.mask_prompt  = mask_prompt
        self.split        = split
        self.max_samples  = max_samples
        self.dataset_name = dataset_name

        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(tokenizer_path)
        self.eos = self.tok.eos_token_id

        # Special token IDs
        self.SYS   = self.tok.convert_tokens_to_ids("<|system|>")
        self.USR   = self.tok.convert_tokens_to_ids("<|user|>")
        self.THT   = self.tok.convert_tokens_to_ids("<|thought|>")
        self.AST   = self.tok.convert_tokens_to_ids("<|assistant|>")
        self.END   = self.tok.convert_tokens_to_ids("<|end|>")

    def _encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def _format(self, system: str, question: str, response: str):
        """
        Build token ids and labels. Prompt tokens get label -100.
        Returns (input_ids, labels) as lists.
        """
        # Build prompt section (masked)
        prompt_ids = (
            [self.SYS] + self._encode(system.strip())  + [self.END] +
            [self.USR] + self._encode(question.strip()) + [self.END] +
            [self.THT, self.END]   # empty thought block — model fills during RLHF
        )

        # Build response section (learned)
        response_ids = (
            [self.AST] + self._encode(response.strip()) + [self.END, self.eos]
        )

        input_ids = prompt_ids + response_ids
        labels    = (
            [-100] * len(prompt_ids) +
            response_ids
        )
        return input_ids, labels

    def __iter__(self) -> Iterator[torch.Tensor]:
        from datasets import load_dataset

        def _load():
            # Local JSONL file (teacher-generated data) or HuggingFace dataset
            if self.dataset_name.endswith(".jsonl") or self.dataset_name.endswith(".json"):
                import json as _json
                from pathlib import Path as _Path

                def _local_stream():
                    while True:   # loop forever like HF streaming
                        with open(self.dataset_name) as fh:
                            for line in fh:
                                line = line.strip()
                                if line:
                                    yield _json.loads(line)
                return _local_stream()
            else:
                return load_dataset(self.dataset_name, split=self.split, streaming=True)

        ds     = _load()
        target = self.seq_len + 1
        n_seen = 0

        for example in ds:
            convs = example.get("conversations", [])
            if len(convs) < 2:
                continue

            # Extract system / human / gpt turns (ShareGPT format)
            system   = next((c["value"] for c in convs if c["from"] == "system"),
                            "You are Anthos, a helpful and honest assistant.")
            question = next((c["value"] for c in convs if c["from"] == "human"), "")
            response = next((c["value"] for c in convs if c["from"] == "gpt"),   "")

            if not question or not response:
                continue

            input_ids, labels = self._format(system, question, response)

            # Truncate to model context window
            input_ids = input_ids[:target]
            labels    = labels[:target]
            if len(input_ids) < 4:
                continue

            yield (
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(labels,    dtype=torch.long),
            )

            n_seen += 1
            if self.max_samples and n_seen >= self.max_samples:
                # Wrap around — loop the slice indefinitely
                n_seen = 0
                ds     = _load()

    def __len__(self):
        return 517_982   # SlimOrca size


def get_chat_dataloader(
    seq_len:        int,
    batch_size:     int,
    num_workers:    int  = 0,
    tokenizer_path: str  = "data/anthos_tokenizer",
    split:          str  = "train",
    max_samples:    int  = 0,       # 0 = full dataset; N = CPU-friendly slice
    dataset_name:   str  = "Open-Orca/SlimOrca",
) -> DataLoader:
    """
    Returns a DataLoader for chat SFT (ShareGPT format).
    Batches are (input_ids, labels) with prompt tokens masked to -100.

    Run setup_tokenizer.py once before using this loader.
    Set max_samples=5000 and dataset_name="mlabonne/FineTome-100k" for convo_smoke.
    """
    ds = ChatInstructDataset(
        tokenizer_path = tokenizer_path,
        seq_len        = seq_len,
        split          = split,
        max_samples    = max_samples,
        dataset_name   = dataset_name,
    )

    def collate(batch):
        tokens_list = [b[0] for b in batch]
        labels_list = [b[1] for b in batch]
        max_len = max(t.size(0) for t in tokens_list)

        padded_tokens = torch.zeros(len(batch), max_len, dtype=torch.long)
        padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

        for i, (t, l) in enumerate(zip(tokens_list, labels_list)):
            padded_tokens[i, :t.size(0)] = t
            padded_labels[i, :l.size(0)] = l

        return padded_tokens, padded_labels

    return DataLoader(
        ds,
        batch_size  = batch_size,
        num_workers = num_workers,
        pin_memory  = False,
        collate_fn  = collate,
    )


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
