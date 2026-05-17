"""
Anthos — Sovereign Training Script (Mansa Edition)
Think in Streams.
"""

import os
import sys
import math
import time
import argparse
import json
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.optim import AdamW

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent))

from anthos.main    import Anthos
from anthos.configs import get_training_config
from anthos.data    import get_dataloader, get_instruct_dataloader, get_chat_dataloader, get_markdown_dataloader
from anthos.sasft         import RepetitionPenaltyLoss, ThoughtDiversityLoss
from anthos.steering      import ActivationCollector
from anthos.memory_compress import MemoryAugmentedDataset
from anthos.distill       import DistillConfig, DistillationLoss, TeacherLabelDataset
from anthos.eaft          import EAFTLoss

# ─────────────────────────────────────────────────────────────────────────────
# MANSA CONFIGURATION (HARD-CODED FOR STABILITY)
# ─────────────────────────────────────────────────────────────────────────────
MAX_STEPS    = 10000
MAX_LR       = 1.0e-4
MIN_LR       = 1.0e-5
WARMUP_STEPS = 2000
PHASE1_STEPS = 3000
PHASE1_LOOPS = 4
PHASE2_LOOPS = 16
SEQ_LEN      = 512
LOG_EVERY    = 100
SAVE_EVERY   = 1000

# ─────────────────────────────────────────────────────────────────────────────
# Logic Functions
# ─────────────────────────────────────────────────────────────────────────────

def get_lr(step: int) -> float:
    if step < WARMUP_STEPS:
        return MAX_LR * (step + 1) / WARMUP_STEPS
    if step >= MAX_STEPS:
        return MIN_LR
    progress = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    return MIN_LR + 0.5 * (MAX_LR - MIN_LR) * (1 + math.cos(math.pi * progress))

def get_n_loops(step: int) -> int:
    return PHASE1_LOOPS if step < PHASE1_STEPS else PHASE2_LOOPS

@torch.no_grad()
def generate_samples(
    model: Anthos,
    device: str,
    n_loops: int,
    n_samples: int = 3,
    tokenizer_path: str = "gpt2",
) -> list[str]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    tok.model_max_length = SEQ_LEN
    model.eval()
    prompts = ["Once upon a time", "The small robot looked at", "In a world where"][:n_samples]
    samples = []
    for prompt in prompts:
        enc = tok.encode(prompt, truncation=True, max_length=SEQ_LEN)
        ids = torch.tensor([enc], dtype=torch.long).to(device)
        out = model.generate(ids, max_new_tokens=80, n_loops=n_loops, temperature=0.8, top_k=40)
        samples.append(tok.decode(out[0].tolist()))
    model.train()
    return samples

def save_checkpoint(path: Path, model: Anthos, optimizer: AdamW, step: int, loss: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step":      step,
        "loss":      loss,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)
    print(f"  ✓ Saved checkpoint → {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train(tier: str = "proof", resume: str | None = None, teacher_labels: str | None = None, max_steps: int | None = None):
    global MAX_STEPS, MAX_LR, MIN_LR, WARMUP_STEPS, SEQ_LEN, LOG_EVERY, SAVE_EVERY

    model_cfg, train_cfg = get_training_config(tier)
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = Path("checkpoints/mansa_sovereign")

    # ── Tier-specific overrides ───────────────────────────────────────────────
    if tier in ("sft", "instruct"):
        MAX_STEPS    = 500
        MAX_LR       = 3e-5
        MIN_LR       = 3e-6
        WARMUP_STEPS = 50
    elif tier == "convo_smoke":
        MAX_STEPS    = 10_000
        MAX_LR       = 5e-5
        MIN_LR       = 5e-6
        WARMUP_STEPS = 500
        SEQ_LEN      = 256
    elif tier == "identity_hardening":
        MAX_STEPS    = 10_000
        MAX_LR       = 1e-4
        MIN_LR       = 1e-5
        WARMUP_STEPS = 500
        SEQ_LEN      = 256
        LOG_EVERY    = 100
        SAVE_EVERY   = 1000
    elif tier == "history":
        MAX_STEPS    = 5_000
        MAX_LR       = 3e-5
        MIN_LR       = 3e-6
        WARMUP_STEPS = 100
        SEQ_LEN      = 256
        LOG_EVERY    = 10
        SAVE_EVERY   = 500
    elif tier == "distill":
        MAX_STEPS    = 10_000
        MAX_LR       = 2e-4
        MIN_LR       = 2e-5
        WARMUP_STEPS = 500

    # ── CLI override (always applied last) ───────────────────────────────────
    if max_steps is not None:
        MAX_STEPS = max_steps

    model = Anthos(model_cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())

    # ── Loss functions ────────────────────────────────────────────────────────
    eaft_criterion = EAFTLoss(
        vocab_size      = model_cfg.vocab_size,
        top_k           = 50,
        focal_gamma     = 1.0,   # entropy weight: uncertain positions get more gradient
        act_gamma       = 0.5,   # extra weight for positions that used more loops
        max_loops       = model_cfg.max_loop_iters,
        label_smoothing = 0.1,
    )
    rep_loss_fn   = RepetitionPenaltyLoss(ngram_size=4, penalty=0.3)
    div_loss_fn   = ThoughtDiversityLoss(coeff=0.05)
    thought_collector = ActivationCollector(
        model, stream="thought",
        n_thought_tokens=model_cfg.n_thought_tokens,
    )
    thought_collector.attach()

    print(f"\n{'─'*60}")
    print(f"  Anthos — Sovereign Training (Mansa Edition)")
    print(f"  Parameters: {total_params:,}")
    print(f"  Max Steps:  {MAX_STEPS:,} | Warmup: {WARMUP_STEPS}")
    print(f"  Max LR:     {MAX_LR} | Loops: {PHASE1_LOOPS}->{PHASE2_LOOPS}")
    print(f"{'─'*60}\n")

    use_fused = True if device == "cuda" else False
    optimizer = AdamW(model.parameters(), lr=MAX_LR, betas=(0.9, 0.95), fused=use_fused)

    start_step = 0
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing:
            print(f"  ℹ New params (randomly init): {len(missing)} tensors — e.g. {missing[0]}")
        if tier in ("sft", "instruct", "convo_smoke", "history"):
            start_step = 0
            print(f"  ✓ {tier} mode: optimizer state reset (fresh Adam at {MAX_LR})")
        else:
            optimizer.load_state_dict(ckpt["optimizer"])
            start_step = ckpt["step"]

    is_sft      = (tier in ("sft", "instruct", "convo_smoke", "identity_hardening"))
    is_distill  = (tier == "distill")
    is_history  = (tier == "history")

    if is_sft and Path("data/anthos_tokenizer").exists():
        tok_path = "data/anthos_tokenizer"
    else:
        tok_path = "gpt2"

    if is_history:
        history_dir = "data/new_history"
        print(f"  ✓ History tier: training on markdown essays in {history_dir}/")
        loader  = get_markdown_dataloader(
            directory  = history_dir,
            seq_len    = SEQ_LEN,
            batch_size = train_cfg.batch_size,
            num_workers= 0,
        )
        tok_path = "gpt2"

    elif is_distill:
        labels_path = teacher_labels or "data/teacher_labels.jsonl"
        if not Path(labels_path).exists():
            raise FileNotFoundError(
                f"Teacher labels not found: {labels_path}\n"
                f"Run: python generate_teacher_labels.py --teacher <model> --out {labels_path}"
            )
        print(f"  ✓ Distill mode: loading teacher labels from {labels_path}")
        from torch.utils.data import DataLoader

        def _distill_collate(batch):
            ids_list  = [x[0] for x in batch]
            tlog_list = [x[1] for x in batch]
            max_T = max(t.shape[0] for t in ids_list)
            V = tlog_list[0].shape[-1]
            ids_pad  = torch.zeros(len(batch), max_T, dtype=torch.long)
            tlog_pad = torch.full((len(batch), max_T, V), float("-inf"))
            for i, (ids, tlog) in enumerate(zip(ids_list, tlog_list)):
                T = ids.shape[0]
                ids_pad[i, :T]      = ids
                tlog_pad[i, :T, :V] = tlog[:, :V]
            return ids_pad, tlog_pad

        teacher_dataset = TeacherLabelDataset(
            path=labels_path,
            student_vocab_size=model_cfg.vocab_size,
            teacher_vocab_size=model_cfg.vocab_size,
        )
        loader = DataLoader(
            teacher_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=True,
            collate_fn=_distill_collate,
            num_workers=0,
        )
        distill_loss_fn = DistillationLoss(DistillConfig(temperature=4.0, alpha=0.3))
        tok_path = "gpt2"

    elif is_sft:
        if tier == "identity_hardening":
            local_data = "data/phase2_train.jsonl"
            if not Path(local_data).exists():
                raise FileNotFoundError(
                    f"{local_data} not found.\n"
                    "Run: python build_phase2_dataset.py"
                )
            dataset_name = local_data
            max_samples  = 0
            print(f"  ✓ Identity hardening: loading {local_data}")
            print(f"    Creator: Brian Tushae Thomas | Model: Anthos")
        elif tier == "convo_smoke":
            local_data = "data/teacher_conversations.jsonl"
            if Path(local_data).exists():
                dataset_name = local_data
                max_samples  = 0
                print(f"  ✓ Using local teacher data: {local_data}")
            else:
                dataset_name = "mlabonne/FineTome-100k"
                max_samples  = 5000
                print(f"  ✓ Using FineTome-100k (run generate_teacher_data.py for better results)")
        else:
            max_samples  = 0
            dataset_name = "Open-Orca/SlimOrca"

        n_workers = 0 if tier in ("convo_smoke", "identity_hardening") else 1
        loader = get_chat_dataloader(
            seq_len        = SEQ_LEN,
            batch_size     = train_cfg.batch_size,
            num_workers    = n_workers,
            tokenizer_path = tok_path,
            max_samples    = max_samples,
            dataset_name   = dataset_name,
        )
    else:
        loader = get_dataloader(
            dataset_name = train_cfg.dataset,
            split        = "train",
            seq_len      = SEQ_LEN,
            batch_size   = train_cfg.batch_size,
            num_workers  = 1,
        )

    if tier in ("proof", "sft", "instruct"):
        loader = MemoryAugmentedDataset(
            loader,
            compress_fraction=0.15,
            prefix_confidence=3,
        )

    data_iter = iter(loader)
    model.train()
    step       = start_step
    loss_accum = aux_accum = 0.0
    avg_loss   = 0.0
    t0         = time.time()

    print(f"  🔄 Training loop started — logging every {LOG_EVERY} steps, saving every {SAVE_EVERY}", flush=True)
    print(f"  ⏳ First log will appear after step {LOG_EVERY} (may take 1-2 min on CPU)...", flush=True)

    while step < MAX_STEPS:
        lr     = get_lr(step)
        for pg in optimizer.param_groups: pg["lr"] = lr
        n_loops = get_n_loops(step)

        optimizer.zero_grad()
        for _ in range(train_cfg.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch     = next(data_iter)

            if is_distill:
                input_ids       = batch[0].to(device)[:, :SEQ_LEN]
                teacher_logits  = batch[1].to(device)[:, :SEQ_LEN, :]
                labels          = input_ids[:, 1:]
                input_ids_in    = input_ids[:, :-1]
            elif is_sft:
                input_ids       = batch[0].to(device)[:, :SEQ_LEN]
                labels          = batch[1].to(device)[:, :SEQ_LEN]
                input_ids_in    = input_ids
            else:
                batch        = batch.to(device)
                input_ids_in = batch[:, :SEQ_LEN-1]
                labels       = batch[:, 1:SEQ_LEN]

            _dtype = torch.bfloat16 if device == "cuda" else torch.float32
            with torch.amp.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=_dtype):
                logits, aux = model(input_ids_in, n_loops=n_loops, return_aux=True)

                if is_distill:
                    t_log = teacher_logits[:, :-1, :]
                    loss_kd, kd_info = distill_loss_fn(
                        student_logits=logits,
                        teacher_logits=t_log,
                        labels=labels,
                    )
                    ce  = torch.tensor(kd_info["ce_loss"], device=device)
                    loss = (loss_kd + aux) / train_cfg.grad_accum
                else:
                    # EAFT: entropy-aware focal loss — uncertain positions (high
                    # entropy) and positions that used more ACT loops get more gradient
                    loops_used = getattr(model, "_last_loops_used", None)
                    ce         = eaft_criterion(logits, labels, loops_used=loops_used)
                    rep_pen = rep_loss_fn(logits)
                    thought_acts = thought_collector.flat_activations().to(device)
                    n_thought    = model_cfg.n_thought_tokens
                    # ── FIX: trim thought_acts to a clean multiple of (n_thought * dim)
                    # before reshape. Raw activations can be any size depending on
                    # batch/seq combinations; the view requires exact divisibility.
                    chunk = n_thought * model_cfg.dim
                    if thought_acts.shape[0] >= chunk:
                        trimmed = thought_acts[: (thought_acts.shape[0] // chunk) * chunk]
                        div_pen = div_loss_fn(
                            trimmed.view(-1, n_thought, model_cfg.dim)
                        )
                    else:
                        div_pen = torch.tensor(0.0, device=device)
                    thought_collector.clear()
                    loss = (ce + aux + rep_pen + div_pen) / train_cfg.grad_accum

            loss.backward()
            loss_accum += ce.item()
            aux_accum  += aux.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1

        if step % LOG_EVERY == 0:
            t1       = time.time()
            avg_loss = loss_accum / (LOG_EVERY * train_cfg.grad_accum)
            avg_aux  = aux_accum  / (LOG_EVERY * train_cfg.grad_accum)
            tok_sec  = (LOG_EVERY * train_cfg.batch_size * train_cfg.grad_accum * SEQ_LEN) / (t1 - t0)
            print(f"step {step:6d} | loss {avg_loss:.4f} | ponder {avg_aux:.5f} | loops {n_loops} | lr {lr:.2e} | {tok_sec:,.0f} tok/s", flush=True)
            loss_accum = aux_accum = 0.0
            t0 = t1

        if step % SAVE_EVERY == 0:
            save_checkpoint(ckpt_dir / f"step_{step:06d}.pt", model, optimizer, step, avg_loss)
            print("\n── Sample outputs ─────────────────────────────────────")
            try:
                for sample in generate_samples(model, device, n_loops, tokenizer_path=tok_path):
                    print(f"  {sample[:200]}\n")
            except Exception as e:
                print(f"  ⚠ Sample generation failed: {e}")
            print("────────────────────────────────────────────────────────\n")

    print(f"\n✓ Sovereign Training complete — {step} steps")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier",   type=str, default="proof",
                        choices=["smoke", "proof", "research", "ethnic", "instruct", "sft", "convo_smoke", "distill", "history", "identity_hardening"])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--steps",  type=int, default=None,
                        help="Override MAX_STEPS for this run (e.g. --steps 150000)")
    parser.add_argument("--teacher_labels", type=str, default=None,
                        help="Path to teacher soft-label JSONL (required for --tier distill)")
    args = parser.parse_args()
    train(tier=args.tier, resume=args.resume, teacher_labels=args.teacher_labels, max_steps=args.steps)
