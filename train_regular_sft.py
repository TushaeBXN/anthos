"""SFT fine-tuning for the Regular pretrained model.

    # Recommended: Alpaca + Dolly mix (~66k examples, ~$0.10)
    python train_regular_sft.py \\
        --init-from checkpoints/regular-fw2/step_030000.pt \\
        --run-name regular-sft \\
        --datasets alpaca dolly \\
        --steps 2000 --lr 2e-5

    # Larger: OpenHermes (1M examples, better capability)
    python train_regular_sft.py \\
        --init-from checkpoints/regular-fw2/step_030000.pt \\
        --run-name regular-sft-oh \\
        --datasets openhermes \\
        --steps 5000 --lr 2e-5

Checkpoints go to checkpoints/<run>/step_NNNNNN.pt.
Loss is computed ONLY on response tokens — instruction tokens are masked out.

Prompt format (Anthos):
    ### Instruction:
    {instruction}

    ### Response:
    {response}<|endoftext|>
"""
import argparse
import dataclasses
import json
import math
import random
import time
from pathlib import Path
from typing import Iterator

import tiktoken
import torch
import torch.nn.functional as F
from datasets import load_dataset

from regular import PRESETS, Regular, RegularConfig

PROMPT_PREFIX = "### Instruction:\n"
PROMPT_SUFFIX = "\n\n### Response:\n"
EOS_ID = 100257  # <|endoftext|> in cl100k_base


def format_alpaca(ex: dict) -> tuple[str, str] | None:
    instruction = ex.get("instruction", "").strip()
    inp = ex.get("input", "").strip()
    response = ex.get("output", "").strip()
    if not instruction or not response:
        return None
    if inp:
        instruction = f"{instruction}\n\nInput: {inp}"
    return instruction, response


def format_dolly(ex: dict) -> tuple[str, str] | None:
    instruction = ex.get("instruction", "").strip()
    context = ex.get("context", "").strip()
    response = ex.get("response", "").strip()
    if not instruction or not response:
        return None
    if context:
        instruction = f"{instruction}\n\nContext: {context}"
    return instruction, response


def format_openhermes(ex: dict) -> tuple[str, str] | None:
    convs = ex.get("conversations", [])
    user = next((c["value"] for c in convs if c.get("from") in ("human", "user")), None)
    asst = next((c["value"] for c in convs if c.get("from") in ("gpt", "assistant")), None)
    if not user or not asst:
        return None
    return user.strip(), asst.strip()


def _parse_local_line(line: str) -> tuple[str, str] | None:
    """Parse one JSONL line — handles {instruction,response} and ShareGPT {conversations} formats."""
    try:
        ex = json.loads(line)
    except Exception:
        return None
    # ShareGPT format: {conversations: [{from, value}]}
    if "conversations" in ex:
        convs = ex["conversations"]
        system = next((c.get("value", "") for c in convs if c.get("from") == "system"), "")
        human  = next((c.get("value", "") for c in convs if c.get("from") in ("human", "user")), None)
        gpt    = next((c.get("value", "") for c in convs if c.get("from") in ("gpt", "assistant")), None)
        if not human or not gpt:
            return None
        inst = (system.strip() + "\n\n" + human.strip()).strip() if system else human.strip()
        return inst, gpt.strip()
    # Flat format: {instruction, response} or {instruction, output}
    inst = ex.get("instruction", "").strip()
    resp = ex.get("response", ex.get("output", "")).strip()
    if inst and resp:
        return inst, resp
    return None


DATASET_CONFIGS = {
    "alpaca":      ("yahma/alpaca-cleaned",           "train", format_alpaca),
    "dolly":       ("databricks/databricks-dolly-15k", "train", format_dolly),
    "openhermes":  ("teknium/OpenHermes-2.5",          "train", format_openhermes),
    "distill":     None,  # local JSONL — path set via --distill-data (kept for back-compat)
    "local":       None,  # one or more local JSONL paths — set via --local-data
}


def load_examples(dataset_names: list[str], distill_data: str | None = None,
                  local_data: list[str] | None = None) -> list[tuple[str, str]]:
    examples = []
    for name in dataset_names:
        if name == "distill":
            paths = [distill_data] if distill_data else []
            if not paths:
                raise ValueError("--distill-data path required when using --datasets distill")
        elif name == "local":
            paths = local_data or []
            if not paths:
                raise ValueError("--local-data path(s) required when using --datasets local")
        else:
            paths = []

        if paths:
            for path in paths:
                print(f"Loading local JSONL from {path}...", flush=True)
                before = len(examples)
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        result = _parse_local_line(line)
                        if result:
                            examples.append(result)
                print(f"  +{len(examples) - before} examples", flush=True)
            continue

        hf_id, split, fmt = DATASET_CONFIGS[name]
        print(f"Loading {name} ({hf_id})...", flush=True)
        ds = load_dataset(hf_id, split=split)
        before = len(examples)
        for ex in ds:
            result = fmt(ex)
            if result:
                examples.append(result)
        print(f"  +{len(examples) - before} examples", flush=True)
    random.shuffle(examples)
    print(f"Total: {len(examples)} examples", flush=True)
    return examples


def tokenize(tok: tiktoken.Encoding, instruction: str, response: str, seq_len: int
             ) -> tuple[torch.Tensor, torch.Tensor] | None:
    prompt = PROMPT_PREFIX + instruction + PROMPT_SUFFIX
    prompt_ids = tok.encode_ordinary(prompt)
    response_ids = tok.encode_ordinary(response) + [EOS_ID]

    all_ids = prompt_ids + response_ids
    if len(all_ids) > seq_len + 1:
        return None  # too long — skip

    # Pad to seq_len + 1 with EOS
    pad = [EOS_ID] * (seq_len + 1 - len(all_ids))
    all_ids = all_ids + pad

    tokens = torch.tensor(all_ids, dtype=torch.long)
    x = tokens[:-1]           # input
    y = tokens[1:].clone()    # target (shifted)

    # Mask prompt tokens and padding in labels
    n_prompt = len(prompt_ids)
    y[:n_prompt - 1] = -100              # mask prompt (shifted: predict from last prompt token)
    response_end = n_prompt + len(response_ids)
    y[response_end - 1:] = -100          # mask padding after EOS

    return x, y


def batches(tok: tiktoken.Encoding, examples: list, seq_len: int, batch_size: int,
            seed: int = 0) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    rng = random.Random(seed)
    while True:
        order = list(range(len(examples)))
        rng.shuffle(order)
        xs, ys = [], []
        for i in order:
            inst, resp = examples[i]
            result = tokenize(tok, inst, resp, seq_len)
            if result is None:
                continue
            xs.append(result[0]); ys.append(result[1])
            if len(xs) == batch_size:
                yield torch.stack(xs), torch.stack(ys)
                xs, ys = [], []


def lr_at(step: int, a) -> float:
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    t = (step - a.warmup) / max(1, a.steps - a.warmup)
    return a.lr * (a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, t))))


def latest(run_dir: Path):
    ck = sorted(run_dir.glob("step_*.pt"))
    return ck[-1] if ck else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-from", required=True,
                    help="Pretrained checkpoint to fine-tune from")
    ap.add_argument("--run-name", default="regular-sft")
    ap.add_argument("--datasets", nargs="+", default=["alpaca", "dolly"],
                    choices=list(DATASET_CONFIGS.keys()),
                    help="One or more SFT datasets to mix")
    ap.add_argument("--distill-data", default=None,
                    help="Path to distillation JSONL (required when --datasets includes distill)")
    ap.add_argument("--local-data", nargs="+", default=None, metavar="PATH",
                    help="One or more local JSONL paths (required when --datasets includes local). "
                         "Supports {instruction,response} and ShareGPT {conversations} formats.")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    device = torch.device(("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
                          if a.device == "auto" else a.device)
    torch.manual_seed(a.seed)
    random.seed(a.seed)

    run_dir = Path("checkpoints") / a.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Resume from SFT checkpoint if one exists, otherwise load pretrained
    ck = latest(run_dir)
    src = ck or a.init_from
    state = torch.load(src, map_location=device, weights_only=False)
    cfg = RegularConfig(**state["cfg"])
    model = Regular(cfg).to(device)
    model.load_state_dict(state["model"])

    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.weight_decay},
                              {"params": nodecay, "weight_decay": 0.0}],
                             lr=a.lr, betas=(0.9, 0.95))

    step = 0
    if ck:
        opt.load_state_dict(state["opt"]); step = state["step"]
        print(f"resumed from {ck} at step {step}", flush=True)
    else:
        print(f"loaded pretrained weights from {a.init_from}", flush=True)
    print(f"params {model.num_params():,} on {device}", flush=True)

    tok = tiktoken.get_encoding("cl100k_base")
    examples = load_examples(a.datasets, distill_data=a.distill_data, local_data=a.local_data)
    loader = batches(tok, examples, a.seq_len, a.batch_size, seed=a.seed)
    use_amp = device.type == "cuda"
    log = open(run_dir / "train.log", "a")
    t0, resp_toks = time.time(), 0

    while step < a.steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a)
        total = 0.0
        for _ in range(a.grad_accum):
            x, y = next(loader)
            x, y = x.to(device), y.to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(x)
            loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten(),
                                   ignore_index=-100) / a.grad_accum
            loss.backward()
            total += loss.item()
            resp_toks += (y != -100).sum().item()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip).item()
        opt.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if not math.isfinite(total):
            raise SystemExit(f"loss is {total} at step {step}: stopping")
        if step % a.log_every == 0:
            rec = {"step": step, "loss": round(total, 4), "gnorm": round(gnorm, 3),
                   "lr": round(opt.param_groups[0]["lr"], 8),
                   "resp_tok_s": int(resp_toks / (time.time() - t0))}
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        if step % a.save_every == 0 or step == a.steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "cfg": dataclasses.asdict(cfg)},
                       run_dir / f"step_{step:06d}.pt")
    print("RUN_EXIT complete", flush=True); log.write("RUN_EXIT complete\n"); log.close()


if __name__ == "__main__":
    main()
