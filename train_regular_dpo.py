"""DPO (Direct Preference Optimization) fine-tuning for the Regular model.

Trains the model to prefer direct, helpful responses over refusals/over-hedged answers.

    # Recommended: Orca DPO pairs (direct answers vs refusals, ~13k pairs)
    python train_regular_dpo.py \\
        --init-from checkpoints/regular-sft/step_002000.pt \\
        --run-name regular-dpo \\
        --datasets orca_dpo \\
        --steps 1000 --lr 5e-6

    # Larger mix
    python train_regular_dpo.py \\
        --init-from checkpoints/regular-sft/step_002000.pt \\
        --run-name regular-dpo \\
        --datasets orca_dpo argilla_dpo \\
        --steps 2000 --lr 5e-6

Checkpoints go to checkpoints/<run>/step_NNNNNN.pt.

DPO loss trains on (instruction, chosen_response, rejected_response) triples.
The reference model (frozen SFT checkpoint) anchors the KL penalty via --beta.
Loss is computed only on response tokens; prompts are masked.
"""
import argparse
import copy
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

from regular import Regular, RegularConfig

PROMPT_PREFIX = "### Instruction:\n"
PROMPT_SUFFIX = "\n\n### Response:\n"
EOS_ID = 100257  # <|endoftext|> in cl100k_base


# ── dataset formatters ────────────────────────────────────────────────────────

def _extract_text(value) -> str:
    """Pull assistant text out of a string or a list of message dicts."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        for m in reversed(value):
            role = m.get("role", m.get("from", ""))
            if role in ("assistant", "gpt"):
                return m.get("content", m.get("value", "")).strip()
    return ""


def format_orca_dpo(ex: dict) -> tuple[str, str, str] | None:
    system = ex.get("system", "").strip()
    question = ex.get("question", "").strip()
    instruction = f"{system}\n\n{question}" if system else question
    chosen   = _extract_text(ex.get("chosen", ""))
    rejected = _extract_text(ex.get("rejected", ""))
    if not instruction or not chosen or not rejected or chosen == rejected:
        return None
    return instruction, chosen, rejected


def format_argilla_dpo(ex: dict) -> tuple[str, str, str] | None:
    system      = ex.get("system", "").strip()
    instruction = ex.get("instruction", "").strip()
    if system:
        instruction = f"{system}\n\n{instruction}"
    chosen   = _extract_text(ex.get("chosen", ""))
    rejected = _extract_text(ex.get("rejected", ""))
    if not instruction or not chosen or not rejected or chosen == rejected:
        return None
    return instruction, chosen, rejected


def format_orpo_mix(ex: dict) -> tuple[str, str, str] | None:
    prompt = ex.get("prompt", "")
    if isinstance(prompt, list):
        # extract last user message
        prompt = next((m.get("content", "") for m in reversed(prompt)
                       if m.get("role") == "user"), "")
    instruction = prompt.strip() if isinstance(prompt, str) else ""
    chosen   = _extract_text(ex.get("chosen", ""))
    rejected = _extract_text(ex.get("rejected", ""))
    if not instruction or not chosen or not rejected or chosen == rejected:
        return None
    return instruction, chosen, rejected


DATASET_CONFIGS = {
    "orca_dpo":   ("Intel/orca_dpo_pairs",        "train", format_orca_dpo),
    "argilla_dpo":("argilla/dpo-mix-7k",           "train", format_argilla_dpo),
    "orpo_mix":   ("kaist-ai/orpo-dpo-mix-40k",    "train", format_orpo_mix),
}


def load_examples(dataset_names: list[str]) -> list[tuple[str, str, str]]:
    examples = []
    for name in dataset_names:
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
    print(f"Total: {len(examples)} DPO pairs", flush=True)
    return examples


# ── tokenisation ──────────────────────────────────────────────────────────────

def tokenize_pair(tok: tiktoken.Encoding, instruction: str,
                  chosen: str, rejected: str, seq_len: int
                  ) -> tuple[torch.Tensor, torch.Tensor,
                              torch.Tensor, torch.Tensor] | None:
    prompt      = PROMPT_PREFIX + instruction + PROMPT_SUFFIX
    prompt_ids  = tok.encode_ordinary(prompt)

    def encode_response(response: str) -> tuple[torch.Tensor, torch.Tensor] | None:
        resp_ids = tok.encode_ordinary(response) + [EOS_ID]
        all_ids  = prompt_ids + resp_ids
        if len(all_ids) > seq_len + 1:
            return None
        pad = [EOS_ID] * (seq_len + 1 - len(all_ids))
        ids = torch.tensor(all_ids + pad, dtype=torch.long)
        x   = ids[:-1]
        y   = ids[1:].clone()
        n_p           = len(prompt_ids)
        resp_end      = n_p + len(resp_ids)
        y[:n_p - 1]   = -100   # mask prompt
        y[resp_end - 1:] = -100  # mask padding
        return x, y

    r_chosen   = encode_response(chosen)
    r_rejected = encode_response(rejected)
    if r_chosen is None or r_rejected is None:
        return None
    return r_chosen[0], r_chosen[1], r_rejected[0], r_rejected[1]


def batches(tok: tiktoken.Encoding, examples: list, seq_len: int, batch_size: int,
            seed: int = 0) -> Iterator[tuple]:
    rng = random.Random(seed)
    while True:
        order = list(range(len(examples)))
        rng.shuffle(order)
        xc_buf, yc_buf, xr_buf, yr_buf = [], [], [], []
        for i in order:
            inst, chosen, rejected = examples[i]
            result = tokenize_pair(tok, inst, chosen, rejected, seq_len)
            if result is None:
                continue
            xc, yc, xr, yr = result
            xc_buf.append(xc); yc_buf.append(yc)
            xr_buf.append(xr); yr_buf.append(yr)
            if len(xc_buf) == batch_size:
                yield (torch.stack(xc_buf), torch.stack(yc_buf),
                       torch.stack(xr_buf), torch.stack(yr_buf))
                xc_buf, yc_buf, xr_buf, yr_buf = [], [], [], []


# ── DPO loss ──────────────────────────────────────────────────────────────────

def compute_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean log-prob over non-masked positions (per example in batch)."""
    log_probs = F.log_softmax(logits, dim=-1)          # stay in bfloat16; avoid 3 GB float32 cast
    mask      = labels != -100
    tgt       = labels.clone(); tgt[~mask] = 0
    token_lp  = log_probs.gather(2, tgt.unsqueeze(-1)).squeeze(-1).float()
    return (token_lp * mask).sum(-1) / mask.sum(-1).clamp(min=1)


def dpo_loss(policy_chosen_lp, policy_rejected_lp,
             ref_chosen_lp,    ref_rejected_lp, beta: float):
    chosen_rewards   = beta * (policy_chosen_lp   - ref_chosen_lp)
    rejected_rewards = beta * (policy_rejected_lp - ref_rejected_lp)
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
    acc  = (chosen_rewards > rejected_rewards).float().mean().item()
    return loss, chosen_rewards.mean().item(), rejected_rewards.mean().item(), acc


# ── LR schedule ──────────────────────────────────────────────────────────────

def lr_at(step: int, a) -> float:
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    t = (step - a.warmup) / max(1, a.steps - a.warmup)
    return a.lr * (a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, t))))


def latest(run_dir: Path):
    ck = sorted(run_dir.glob("step_*.pt"))
    return ck[-1] if ck else None


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-from", required=True,
                    help="SFT checkpoint to start DPO from (also used as reference model)")
    ap.add_argument("--run-name", default="regular-dpo")
    ap.add_argument("--datasets", nargs="+", default=["orca_dpo"],
                    choices=list(DATASET_CONFIGS.keys()))
    ap.add_argument("--steps",       type=int,   default=1000)
    ap.add_argument("--batch-size",  type=int,   default=8)
    ap.add_argument("--grad-accum",  type=int,   default=4)
    ap.add_argument("--seq-len",     type=int,   default=512)
    ap.add_argument("--lr",          type=float, default=5e-6)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--warmup",      type=int,   default=50)
    ap.add_argument("--beta",        type=float, default=0.1,
                    help="KL penalty coefficient; lower = less constrained from reference")
    ap.add_argument("--weight-decay",type=float, default=0.0)
    ap.add_argument("--clip",        type=float, default=1.0)
    ap.add_argument("--save-every",  type=int,   default=250)
    ap.add_argument("--log-every",   type=int,   default=10)
    ap.add_argument("--seed",        type=int,   default=0)
    ap.add_argument("--device",      default="auto")
    a = ap.parse_args()

    device = torch.device(("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
                          if a.device == "auto" else a.device)
    torch.manual_seed(a.seed)
    random.seed(a.seed)

    run_dir = Path("checkpoints") / a.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load policy model (trainable)
    ck = latest(run_dir)
    src = ck or a.init_from
    state = torch.load(src, map_location=device, weights_only=False)
    cfg = RegularConfig(**state["cfg"])
    model = Regular(cfg).to(device)
    model.load_state_dict(state["model"])

    # Reference model: frozen copy of the SFT checkpoint
    ref_state = torch.load(a.init_from, map_location=device, weights_only=False)
    ref_model = Regular(cfg).to(device)
    ref_model.load_state_dict(ref_state["model"])
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    decay   = [p for n, p in model.named_parameters() if p.ndim >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay,   "weight_decay": a.weight_decay},
                              {"params": nodecay, "weight_decay": 0.0}],
                             lr=a.lr, betas=(0.9, 0.95))

    step = 0
    if ck:
        opt.load_state_dict(state["opt"]); step = state["step"]
        print(f"resumed from {ck} at step {step}", flush=True)
    else:
        print(f"loaded SFT weights from {a.init_from}", flush=True)
    print(f"params {model.num_params():,} on {device} | beta={a.beta}", flush=True)

    tok      = tiktoken.get_encoding("cl100k_base")
    examples = load_examples(a.datasets)
    loader   = batches(tok, examples, a.seq_len, a.batch_size, seed=a.seed)
    use_amp  = device.type == "cuda"
    log      = open(run_dir / "train.log", "a")
    t0       = time.time()

    while step < a.steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a)

        total_loss = 0.0
        total_chosen_r = total_rejected_r = total_acc = 0.0

        for _ in range(a.grad_accum):
            xc, yc, xr, yr = next(loader)
            xc, yc = xc.to(device), yc.to(device)
            xr, yr = xr.to(device), yr.to(device)

            # Concat chosen+rejected for single policy forward pass
            x_all = torch.cat([xc, xr], dim=0)
            y_all = torch.cat([yc, yr], dim=0)

            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits_all = model(x_all)
            lp_all = compute_logps(logits_all, y_all)
            policy_chosen_lp, policy_rejected_lp = lp_all.chunk(2, dim=0)

            # Reference forward (no grad)
            with torch.no_grad():
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                    ref_logits_all = ref_model(x_all)
            ref_lp_all = compute_logps(ref_logits_all, y_all)
            ref_chosen_lp, ref_rejected_lp = ref_lp_all.chunk(2, dim=0)

            loss, cr, rr, acc = dpo_loss(
                policy_chosen_lp, policy_rejected_lp,
                ref_chosen_lp,    ref_rejected_lp, a.beta)
            (loss / a.grad_accum).backward()

            total_loss      += loss.item()
            total_chosen_r  += cr
            total_rejected_r+= rr
            total_acc       += acc

        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip).item()
        opt.step(); opt.zero_grad(set_to_none=True)
        step += 1

        if not math.isfinite(total_loss):
            raise SystemExit(f"loss is {total_loss} at step {step}: stopping")

        if step % a.log_every == 0:
            rec = {
                "step":          step,
                "loss":          round(total_loss      / a.grad_accum, 4),
                "chosen_r":      round(total_chosen_r  / a.grad_accum, 4),
                "rejected_r":    round(total_rejected_r/ a.grad_accum, 4),
                "reward_acc":    round(total_acc        / a.grad_accum, 4),
                "gnorm":         round(gnorm, 3),
                "lr":            round(opt.param_groups[0]["lr"], 10),
            }
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()

        if step % a.save_every == 0 or step == a.steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "cfg": dataclasses.asdict(cfg)},
                       run_dir / f"step_{step:06d}.pt")

    print("RUN_EXIT complete", flush=True)
    log.write("RUN_EXIT complete\n"); log.close()


if __name__ == "__main__":
    main()
