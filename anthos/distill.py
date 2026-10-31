"""
anthos/distill.py — Teacher-Student Knowledge Distillation for Anthos

Goal: make a small Anthos punch well above its parameter count by training
it to mimic the output distributions of a large teacher model.

Two distillation strategies:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Strategy 1 — Offline Distillation (recommended to start)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Pre-generate teacher soft labels (top-k logprobs) and save to disk.
Train Anthos on saved labels — teacher never runs during student training.
Teacher can be a much larger model (Qwen3-32B, LLaMA-3.1-70B, etc.) that
you run once via Unsloth Studio or vLLM to generate the label dataset.

Workflow:
  1. python generate_teacher_labels.py  → data/teacher_labels.jsonl
  2. python train.py --tier distill      → trains student on saved labels

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Strategy 2 — Online Distillation (higher quality, requires both models in VRAM)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Teacher runs alongside student each step. Student sees fresh teacher
distributions per batch. Requires teacher and student both fit in RAM/VRAM.
Feasible on M1 Max 64GB: Qwen3-7B teacher (Q4: ~4GB) + Anthos-1B student (~2GB).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Loss formula (Hinton et al. 2015, adapted):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  L_total = α · L_CE(student, hard_labels)
          + (1-α) · T² · L_KL(student/T || teacher/T)

  Where:
    T     = temperature (default 4.0 — higher = softer, more info transferred)
    α     = hard label weight (default 0.3 — lean on teacher)
    L_KL  = KL divergence (forward: student learns teacher's distribution)
    T²    = rescaling factor to maintain loss magnitude across temperatures

  Anthos bonus — ThoughtStream distillation:
  If teacher also has thought tokens (another Anthos instance), also distill
  the intermediate thought activations via cosine similarity loss.
  This teaches the student HOW to reason, not just WHAT to output.

Usage — Offline (recommended first):
    from anthos.distill import DistillationLoss, DistillConfig

    cfg  = DistillConfig(temperature=4.0, alpha=0.3)
    loss_fn = DistillationLoss(cfg)

    # In training loop:
    logits, aux = student(input_ids, n_loops=8, return_aux=True)
    loss, info  = loss_fn(
        student_logits=logits,
        teacher_logits=teacher_logits,   # loaded from saved labels
        labels=input_ids[:, 1:],
    )

Usage — Online:
    from anthos.distill import OnlineDistiller

    distiller = OnlineDistiller(
        student=anthos_model,
        teacher=qwen_model,             # any HF model
        cfg=DistillConfig(temperature=4.0),
    )
    loss, info = distiller.step(input_ids, n_loops=8)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DistillConfig:
    temperature:      float = 4.0    # Softens distributions — higher = more transfer
    alpha:            float = 0.3    # Weight on hard CE loss (1-alpha goes to KL)
    top_k_distill:    int   = 0      # If > 0, only distill top-k teacher tokens (memory efficient)
    thought_distill:  bool  = False  # Distill thought stream activations (Anthos→Anthos only)
    thought_coeff:    float = 0.1    # Weight on thought-stream cosine loss
    ignore_index:     int   = -100   # Padding token to ignore in CE loss


# ─────────────────────────────────────────────────────────────────────────────
# Core Distillation Loss
# ─────────────────────────────────────────────────────────────────────────────

class DistillationLoss(nn.Module):
    """
    Hinton-style knowledge distillation loss.

    Combines hard cross-entropy (ground truth) with soft KL divergence
    (teacher distribution). The T² rescaling keeps loss magnitude stable
    across different temperature values.

    Handles vocabulary mismatch: if teacher and student have different
    vocab sizes, projects to the intersection (common tokens only).
    """

    def __init__(self, cfg: DistillConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        student_logits:  torch.Tensor,           # [B, T, V_student]
        teacher_logits:  torch.Tensor,           # [B, T, V_teacher]
        labels:          torch.Tensor,           # [B, T] hard labels
        student_thought: Optional[torch.Tensor] = None,  # [B, N_t, D]
        teacher_thought: Optional[torch.Tensor] = None,  # [B, N_t, D]
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns:
            total_loss: scalar
            info:       dict with component losses for logging
        """
        T, α = self.cfg.temperature, self.cfg.alpha

        # ── Hard CE loss ──────────────────────────────────────────────────
        V_s = student_logits.shape[-1]
        ce_loss = F.cross_entropy(
            student_logits.reshape(-1, V_s),
            labels.reshape(-1),
            ignore_index=self.cfg.ignore_index,
        )

        # ── Soft KL loss ──────────────────────────────────────────────────
        # Align vocab sizes — distill on the smaller vocab
        V_t = teacher_logits.shape[-1]
        V   = min(V_s, V_t)

        s_log_probs = F.log_softmax(student_logits[..., :V] / T, dim=-1)  # [B, T, V]
        t_probs     = F.softmax(teacher_logits[..., :V]     / T, dim=-1)  # [B, T, V]

        if self.cfg.top_k_distill > 0:
            # Memory-efficient: only distill on teacher's top-k tokens
            top_vals, top_idx = t_probs.topk(self.cfg.top_k_distill, dim=-1)
            # Renormalize teacher distribution over top-k
            t_probs_sparse = torch.zeros_like(t_probs)
            t_probs_sparse.scatter_(-1, top_idx, top_vals)
            t_probs_sparse = t_probs_sparse / t_probs_sparse.sum(-1, keepdim=True).clamp(min=1e-8)
            t_probs = t_probs_sparse

        # T² rescaling (Hinton et al.)
        kl_loss = F.kl_div(
            s_log_probs.reshape(-1, V),
            t_probs.reshape(-1, V),
            reduction="batchmean",
        ) * (T ** 2)

        # ── Thought stream distillation (Anthos → Anthos only) ──────────
        thought_loss = torch.tensor(0.0, device=student_logits.device)
        if (self.cfg.thought_distill
                and student_thought is not None
                and teacher_thought is not None):
            # Cosine similarity loss — student thought tokens should align
            # with teacher thought tokens positionally
            s_norm = F.normalize(student_thought, dim=-1)  # [B, N_t, D]
            t_norm = F.normalize(teacher_thought, dim=-1)  # [B, N_t, D]
            cos_sim = (s_norm * t_norm).sum(-1)            # [B, N_t]
            thought_loss = (1 - cos_sim).mean() * self.cfg.thought_coeff

        # ── Combine ───────────────────────────────────────────────────────
        total = α * ce_loss + (1 - α) * kl_loss + thought_loss

        return total, {
            "ce_loss":      ce_loss.item(),
            "kl_loss":      kl_loss.item(),
            "thought_loss": thought_loss.item(),
            "total_loss":   total.item(),
            "temperature":  T,
            "alpha":        α,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Teacher Label Generator (offline distillation — run once)
# ─────────────────────────────────────────────────────────────────────────────

class TeacherLabelGenerator:
    """
    Run teacher model over a dataset and save soft labels to disk.

    Saves top-k logprobs per token position to avoid storing full vocab
    distributions (which are huge). Student reconstructs approximate
    distribution from top-k during training.

    Designed for running a large teacher model (Qwen3-14B, LLaMA-3.1-70B)
    through Unsloth or HuggingFace Transformers once, offline.

    Usage:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from anthos.distill import TeacherLabelGenerator

        teacher  = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-14B", ...)
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")

        gen = TeacherLabelGenerator(teacher, tokenizer, top_k=64)
        gen.generate(
            dataset_name="roneneldan/TinyStories",
            output_path="data/teacher_labels_qwen14b.jsonl",
            n_samples=50_000,
        )
    """

    def __init__(self, teacher_model, tokenizer, top_k: int = 64, device: str = "cpu"):
        self.teacher   = teacher_model
        self.tokenizer = tokenizer
        self.top_k     = top_k
        self.device    = device

    @torch.no_grad()
    def generate_for_batch(
        self,
        input_ids: torch.Tensor,    # [B, T]
        seq_len:   int = 512,
    ) -> dict:
        """
        Returns dict:
            top_k_ids:    [B, T, K] — token ids of top-k teacher predictions
            top_k_logits: [B, T, K] — corresponding logits (not softmaxed)
        """
        input_ids = input_ids[:, :seq_len].to(self.device)
        outputs   = self.teacher(input_ids)
        logits    = outputs.logits                  # [B, T, V]

        top_vals, top_idx = logits.topk(self.top_k, dim=-1)

        return {
            "top_k_ids":    top_idx.cpu().tolist(),
            "top_k_logits": top_vals.cpu().tolist(),
        }

    def generate(
        self,
        dataset_name: str,
        output_path:  str,
        n_samples:    int = 50_000,
        batch_size:   int = 4,
        seq_len:      int = 512,
    ):
        """Generate and save teacher labels for a full dataset."""
        import json
        from pathlib import Path
        from datasets import load_dataset

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        dataset = load_dataset(dataset_name, split="train", streaming=True)
        self.teacher.eval()

        n_written = 0
        batch_ids_accum = []

        with open(output_path, "w") as fout:
            for sample in dataset:
                if n_written >= n_samples:
                    break

                text    = sample.get("text", sample.get("content", ""))
                enc     = self.tokenizer(text, truncation=True,
                                         max_length=seq_len, return_tensors="pt")
                input_ids = enc["input_ids"]

                labels = self.generate_for_batch(input_ids, seq_len)
                labels["input_ids"] = input_ids[0].tolist()

                fout.write(json.dumps(labels) + "\n")
                n_written += 1

                if n_written % 1000 == 0:
                    print(f"  Generated {n_written:,}/{n_samples:,} teacher label sequences")

        print(f"✓ Teacher labels saved → {output_path} ({n_written:,} sequences)")


# ─────────────────────────────────────────────────────────────────────────────
# Teacher Label Dataset (load saved labels for student training)
# ─────────────────────────────────────────────────────────────────────────────

class TeacherLabelDataset:
    """
    Loads pre-saved teacher labels for use in student distillation training.

    Reconstructs full-vocabulary teacher logit tensors from saved top-k
    (non-top-k positions set to -inf so they don't contribute to KL loss).
    """

    def __init__(self, path: str, student_vocab_size: int, teacher_vocab_size: int):
        import json
        self.path = path
        self.V_student = student_vocab_size
        self.V_teacher = teacher_vocab_size
        self._data = []
        with open(path) as f:
            for line in f:
                self._data.append(json.loads(line))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            input_ids:      [T] student token ids
            teacher_logits: [T, V] reconstructed teacher logits (sparse)
        """
        item       = self._data[idx]
        input_ids  = torch.tensor(item["input_ids"], dtype=torch.long)
        T          = len(input_ids)
        top_k_ids  = item["top_k_ids"]
        top_k_vals = item["top_k_logits"]

        V = min(self.V_student, self.V_teacher)
        teacher_logits = torch.full((T, V), float("-inf"))

        for t in range(T):
            for tok_id, logit in zip(top_k_ids[t], top_k_vals[t]):
                if tok_id < V:
                    teacher_logits[t, tok_id] = logit

        return input_ids, teacher_logits


# ─────────────────────────────────────────────────────────────────────────────
# Online Distiller (teacher + student run together)
# ─────────────────────────────────────────────────────────────────────────────

class OnlineDistiller:
    """
    Runs teacher and student forward passes together each training step.

    Feasible on M1 Max 64GB:
        Qwen3-7B (Q4_K_M via llama.cpp ≈ 4GB) as teacher
        Anthos-1B (bfloat16 ≈ 2GB) as student
        Total: ~6GB — fits with room for gradients and activations

    The teacher runs in inference mode (no gradient), so its memory cost
    is just the model weights + KV cache for one batch.

    Usage:
        distiller = OnlineDistiller(student=anthos, teacher=qwen, cfg=cfg)
        for batch in loader:
            loss, info = distiller.step(batch, n_loops=8)
            loss.backward()
            optimizer.step()
    """

    def __init__(self, student, teacher, cfg: DistillConfig, device: str = "cpu"):
        self.student = student
        self.teacher = teacher
        self.cfg     = cfg
        self.device  = device
        self.loss_fn = DistillationLoss(cfg)

    @torch.no_grad()
    def _teacher_forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get teacher logits without gradient computation."""
        self.teacher.eval()
        out = self.teacher(input_ids)
        # Handle both HuggingFace CausalLM and raw logit outputs
        if hasattr(out, "logits"):
            return out.logits
        return out

    def step(
        self,
        input_ids: torch.Tensor,  # [B, T]
        n_loops:   int = 8,
    ) -> tuple[torch.Tensor, dict]:
        """
        One distillation step.
        Returns loss (has grad) and info dict for logging.
        """
        input_ids = input_ids.to(self.device)
        labels    = input_ids[:, 1:]

        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_logits = self._teacher_forward(input_ids)
            # Align to sequence positions: drop last teacher logit
            teacher_logits = teacher_logits[:, :-1, :]

        # Student forward (with grad)
        self.student.train()
        student_logits, aux = self.student(input_ids[:, :-1], n_loops=n_loops, return_aux=True)

        loss, info = self.loss_fn(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
        )

        # Add Anthos auxiliary loss (MoE load balancing + ACT penalty)
        total = loss + aux
        info["aux_loss"]   = aux.item()
        info["total_loss"] = total.item()

        return total, info
