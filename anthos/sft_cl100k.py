"""Chat SFT on the cl100k pretrained checkpoint: token map, example encoding, and the binary loader.

The pretrained model uses tiktoken cl100k_base (ids 0..100276) padded to vocab 100352. The ids from
100277 up were never seen in pretraining, so they carry the chat structure and the reserved building
blocks below. `reinit_new_token_embeddings` must run after loading the base weights: the tied output
layer pushed those rows down in every pretraining context, so they start in a bad place.

Example layout (loss only on what follows <|assistant|>):

    <|system|> {system} <|end|>
    [<|grade:k|>]                     optional reward-conditioning tag, set by a checker, never by the model
    ... earlier turns: <|user|> {q} <|end|> / <|assistant|> {a} <|end|> ...
    <|user|> {q} <|end|> <|thought|> <|end|> <|assistant|>
    {answer} <|end|> <|endoftext|>
                       ^ prefix_len points here: thoughts read only the positions before it
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

BASE_VOCAB = 100277          # tiktoken cl100k_base n_vocab, ids 0..100276
EOS = 100257                 # <|endoftext|>
MODEL_VOCAB = 100352

SPECIALS = {
    # chat structure (trained by SFT)
    "<|system|>": 100277, "<|user|>": 100278, "<|thought|>": 100279,
    "<|assistant|>": 100280, "<|end|>": 100281,
    # reserved building blocks: embeddings initialised, untrained until data for them exists
    "<|tool_call|>": 100282, "<|tool_result|>": 100283,
    "<|done|>": 100284, "<|blocked|>": 100285,
    "<|grade:0|>": 100286, "<|grade:1|>": 100287, "<|grade:2|>": 100288,
    "<|grade:3|>": 100289, "<|grade:4|>": 100290,
}
GRADES = {k: SPECIALS[f"<|grade:{k}|>"] for k in range(5)}
SYS, USR, THT, AST, END = (SPECIALS[t] for t in ("<|system|>", "<|user|>", "<|thought|>", "<|assistant|>", "<|end|>"))
CALL, RES, DONE, BLOCKED = (SPECIALS[t] for t in ("<|tool_call|>", "<|tool_result|>", "<|done|>", "<|blocked|>"))
STATUS = {"done": DONE, "blocked": BLOCKED}

DEFAULT_SYSTEM = "You are Anthos, a helpful and honest assistant."


class ChatTok:
    def __init__(self):
        import tiktoken
        self.enc = tiktoken.get_encoding("cl100k_base")
        assert self.enc.n_vocab == BASE_VOCAB

    def text(self, s: str) -> list[int]:
        return self.enc.encode_ordinary(s)        # never emits special ids from raw text

    def decode(self, ids: list[int]) -> str:
        inv = {v: k for k, v in SPECIALS.items()}
        out, buf = [], []
        for i in ids:
            if i >= BASE_VOCAB or i == EOS:
                if buf: out.append(self.enc.decode(buf)); buf = []
                out.append(inv.get(i, "<|eos|>" if i == EOS else f"<|{i}|>"))
            else:
                buf.append(i)
        if buf: out.append(self.enc.decode(buf))
        return "".join(out)


def encode_prompt(tok: ChatTok, system: str, turns: list[tuple[str, str]], grade: int | None = None) -> list[int]:
    """turns: [(role, text), ...] ending with a 'human' turn; role in {'human','gpt'}. Returns ids up to <|assistant|>."""
    ids = [SYS] + tok.text(system.strip()) + [END]
    if grade is not None:
        ids.append(GRADES[grade])
    for role, text in turns:
        ids += ([USR] if role == "human" else [AST]) + tok.text(text.strip()) + [END]
    return ids + [THT, END, AST]


def encode_example(tok: ChatTok, system: str, turns: list[tuple[str, str]], grade: int | None = None,
                   max_len: int = 512):
    """turns ends with the 'gpt' turn to learn. Returns (ids, prefix_len) or None if it does not fit."""
    *ctx, (role, answer) = turns
    if role != "gpt" or not ctx or ctx[-1][0] != "human" or not answer.strip():
        return None
    prompt = encode_prompt(tok, system, ctx, grade)
    ids = prompt + tok.text(answer.strip()) + [END, EOS]
    return (ids, len(prompt)) if len(ids) <= max_len else None


def encode_trajectory(tok: ChatTok, system: str, task: str, steps: list[dict], final: str, status: str,
                      grade: int | None = None, max_len: int = 512):
    """A tool-use run as one stream with a per-token loss mask. Returns (ids, prefix_len, mask) or None if too long.

    <|assistant|> <|tool_call|>{json}<|end|> <|tool_result|>{result}<|end|> ... {final}<|end|><|done||blocked|><|endoftext|>
    Learned: every tool call, the final answer, <|end|>, the status token. Not learned: the tool results, which the
    harness writes (it also supplies <|tool_result|>, so the model stops at the <|end|> after a call).
    prefix_len ends the task statement; thoughts read nothing after it, tool results included."""
    prompt = encode_prompt(tok, system, [("human", task)], grade)
    ids, mask = list(prompt), [0] * len(prompt)
    for st in steps:
        c = [CALL] + tok.text(json.dumps(st["call"], separators=(",", ":"))) + [END]
        r = [RES] + tok.text(str(st["result"])) + [END]
        ids += c + r; mask += [1] * len(c) + [0] * len(r)
    f = tok.text(final.strip()) + [END, STATUS[status], EOS]
    ids += f; mask += [1] * len(f)
    return (ids, len(prompt), mask) if len(ids) <= max_len else None


def reinit_new_token_embeddings(model, seed: int = 0) -> int:
    """Reset rows >= BASE_VOCAB to the mean of the real-token rows plus small noise (tied head too)."""
    emb = model.embed
    w = emb.weight if hasattr(emb, "weight") else emb
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        mean = w[:BASE_VOCAB].float().mean(0)
        std = w[:BASE_VOCAB].float().std(0).mean() * 0.1
        n = MODEL_VOCAB - BASE_VOCAB
        w[BASE_VOCAB:] = (mean[None, :] + torch.randn(n, w.shape[1], generator=g) * std).to(w.dtype)
    return n


class SFTBinDataset(IterableDataset):
    """Reads tokens.bin (uint32) + index.npy (N x 3: offset, length, prefix_len); shuffled epochs, forever."""

    def __init__(self, path: str, seed: int = 0, return_prefix_len: bool = True, epochs: int | None = None):
        self.dir, self.seed, self.rp, self.epochs = Path(path), seed, return_prefix_len, epochs
        self.index = np.load(self.dir / "index.npy")

    def __len__(self):
        return len(self.index)

    def __iter__(self):
        toks = np.memmap(self.dir / "tokens.bin", dtype=np.uint32, mode="r")
        mpath = self.dir / "mask.bin"
        masks = np.memmap(mpath, dtype=np.uint8, mode="r") if mpath.exists() else None   # per-token loss mask, else prefix_len
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (wi.id if wi else 0))
        ep = 0
        while self.epochs is None or ep < self.epochs:
            order = rng.permutation(len(self.index))
            if wi:
                order = order[wi.id::wi.num_workers]
            for i in order:
                off, n, p = self.index[i]
                ids = torch.from_numpy(toks[off:off + n].astype(np.int64))
                labels = ids.clone()
                if masks is None:
                    labels[:p] = -100
                else:
                    labels[torch.from_numpy(masks[off:off + n].astype(np.int64)) == 0] = -100
                yield (ids, labels, torch.tensor(int(p))) if self.rp else (ids, labels)
            ep += 1


def get_sft_bin_dataloader(path: str, batch_size: int, seed: int = 0, num_workers: int = 0,
                           return_prefix_len: bool = True, epochs: int | None = None) -> DataLoader:
    def collate(batch):
        T = max(b[0].size(0) for b in batch)
        x = torch.zeros(len(batch), T, dtype=torch.long)
        y = torch.full((len(batch), T), -100, dtype=torch.long)
        for i, b in enumerate(batch):
            x[i, :b[0].size(0)] = b[0]; y[i, :b[1].size(0)] = b[1]
        return (x, y, torch.stack([b[2] for b in batch])) if return_prefix_len else (x, y)
    return DataLoader(SFTBinDataset(path, seed, return_prefix_len, epochs), batch_size=batch_size,
                      num_workers=num_workers, collate_fn=collate)


def write_specials_meta(out: Path):
    (out / "specials.json").write_text(json.dumps(SPECIALS, indent=1))
