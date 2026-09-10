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
Teacher can be a much larger model (Llama-3.1-70B, LLaMA-3.1-70B, etc.) that
you run once via Unsloth Studio or vLLM to generate the label dataset.

Workflow:
  1. python generate_teacher_labels.py  → data/teacher_labels.jsonl
  2. python train.py --tier distill      → trains student on saved labels

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Strategy 2 — Online Distillation (higher quality, requires both models in VRAM)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Teacher runs alongside student each step. Student sees fresh teacher
distributions per batch. Requires teacher and student both fit in RAM/VRAM.
Feasible on M1 Max 64GB: Llama-3.2-3B teacher (Q4: ~4GB) + Anthos-1B student (~2GB).

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
        teacher=teacher_model,             # any HF model
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

    Designed for running a large teacher model (Llama-3.1-8B, LLaMA-3.1-70B)
    through Unsloth or HuggingFace Transformers once, offline.

    Usage:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from anthos.distill import TeacherLabelGenerator

        teacher  = AutoModelForCausalLM.from_pretrained("Qwen/Llama-3.1-8B", ...)
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Llama-3.1-8B")

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
        Llama-3.2-3B (Q4_K_M via llama.cpp ≈ 4GB) as teacher
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


# ─────────────────────────────────────────────────────────────────────────────
# Int4-Recovery LoRA  (Chimera pipeline — same trace, two outputs)
# ─────────────────────────────────────────────────────────────────────────────
#
# Problem: int4 quantization of Anthos (or any base model) introduces
# rounding error in every linear layer.  A small LoRA delta trained on the
# same teacher-trace data the SFT run uses can recover most of that error
# without duplicating data generation or running the teacher twice.
#
# Design:
#   1. Load base checkpoint in int4 (requires bitsandbytes ≥ 0.43 on GPU;
#      falls back to float32 when unavailable — same code path, lower fidelity).
#   2. Inject LoRALinear wrappers around every nn.Linear in the base model;
#      only the LoRA delta weights (A, B) are trainable.
#   3. ChimeraTrainer shares one run_id and one DataLoader between the SFT
#      student and the recovery LoRA model — the batch is forwarded through
#      both in the same step, so the teacher is queried only once per batch.
#
# Saving:
#   checkpoints/<run_id>/sft/step_XXXXXX.pt        — SFT student checkpoint
#   checkpoints/<run_id>/lora/step_XXXXXX_lora.pt  — LoRA delta weights only
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RecoveryLoRAConfig:
    rank:          int   = 16       # LoRA rank for all injected adapters
    alpha:         float = 32.0     # LoRA scaling: delta = (alpha/rank) * B @ A
    target_modules: list = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    dropout:       float = 0.05
    use_int4:      bool  = True     # Load base weights in int4 (requires bitsandbytes)
    loss_coef:     float = 1.0      # Weight on recovery LoRA loss relative to SFT loss


@dataclass
class ChimeraConfig:
    """Bundles SFT distillation + int4-recovery LoRA under one run ID."""
    distill:       DistillConfig      = field(default_factory=DistillConfig)
    recovery:      RecoveryLoRAConfig = field(default_factory=RecoveryLoRAConfig)
    run_id:        str                = "chimera_run"
    checkpoint_dir: str              = "checkpoints"
    save_every:    int                = 500


class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that adds a trainable low-rank delta.

    output = frozen_linear(x) + (alpha/rank) * x @ A^T @ B^T

    Frozen base weights stay at their original dtype (int4 or float32).
    A and B are always float32 for numerical stability during training.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        in_f, out_f = base.in_features, base.out_features
        self.base    = base                    # frozen — do not register as parameter
        self.rank    = rank
        self.scale   = alpha / rank
        self.lora_A  = nn.Parameter(torch.empty(rank, in_f))
        self.lora_B  = nn.Parameter(torch.zeros(out_f, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B zeros → delta is 0 at init, preserving base model behavior

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + self.scale * lora_out.to(base_out.dtype)


def add_recovery_lora(model: nn.Module, cfg: RecoveryLoRAConfig) -> nn.Module:
    """
    Walk every nn.Linear in model whose name matches cfg.target_modules,
    replace it with a LoRALinear wrapper, and freeze the base weights.

    Returns the modified model (in-place mutation + return for convenience).
    """
    for module_name, module in list(model.named_modules()):
        parent_name, _, child_name = module_name.rpartition(".")
        if not isinstance(module, nn.Linear):
            continue
        if not any(t in child_name for t in cfg.target_modules):
            continue

        # Freeze base weights
        module.weight.requires_grad_(False)
        if module.bias is not None:
            module.bias.requires_grad_(False)

        # Replace with LoRALinear wrapper
        lora_layer = LoRALinear(module, cfg.rank, cfg.alpha, cfg.dropout)
        parent     = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, lora_layer)

    return model


def load_base_int4(model: nn.Module, checkpoint_path: str, cfg: RecoveryLoRAConfig) -> nn.Module:
    """
    Load a checkpoint into model in int4 (if bitsandbytes is available) or
    float32 (fallback).  Returns the model with base weights frozen.

    When use_int4=False or bitsandbytes is absent, weights are loaded in
    float32 and all base parameters are frozen — recovery LoRA still works.
    """
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # Unwrap trainer state dicts that wrap the model under a key
    for key in ("model", "model_state_dict"):
        if key in state:
            state = state[key]
            break
    model.load_state_dict(state, strict=False)

    if cfg.use_int4:
        try:
            import bitsandbytes as bnb  # type: ignore
            # Replace nn.Linear with bitsandbytes Int8 / Int4 Linear8bitLt layers
            # bitsandbytes.functional.quantize_4bit handles in-place replacement
            # when called through replace_linear_with_target (bnb utility).
            # We do a manual pass here to stay dependency-light.
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and module.weight.requires_grad:
                    module.weight.data = bnb.functional.quantize_4bit(
                        module.weight.data.cuda().half()
                    )[0].cpu()
                    module.weight.requires_grad_(False)
        except (ImportError, AttributeError):
            # bitsandbytes absent or version mismatch — continue in float32
            for p in model.parameters():
                p.requires_grad_(False)
    else:
        for p in model.parameters():
            p.requires_grad_(False)

    return model


class ChimeraTrainer:
    """
    Trains SFT student and int4-recovery LoRA in parallel from the same
    teacher-trace DataLoader.  Both outputs share one run_id.

    The teacher is queried once per batch; both loss computations reuse
    the resulting teacher logits — no duplicated teacher inference.

    Usage:
        cfg = ChimeraConfig(run_id="mansa_sovereign_v2")
        trainer = ChimeraTrainer(
            student=anthos_model,
            recovery_base=anthos_int4,   # base model, already loaded + frozen
            teacher=deepseek_v4,
            cfg=cfg,
        )
        for batch in loader:
            sft_loss, lora_loss, info = trainer.step(batch, n_loops=8)
            # losses are already backward()'d; just step the optimizers
            trainer.step_optimizers()

    Saving checkpoints:
        trainer.save(step)
        # → checkpoints/<run_id>/sft/step_005000.pt
        # → checkpoints/<run_id>/lora/step_005000_lora.pt
    """

    def __init__(
        self,
        student,
        recovery_base: nn.Module,
        teacher,
        cfg:            ChimeraConfig,
        sft_optimizer:  Optional[torch.optim.Optimizer]  = None,
        lora_optimizer: Optional[torch.optim.Optimizer]  = None,
        device:         str                              = "cpu",
    ):
        self.student        = student
        self.teacher        = teacher
        self.cfg            = cfg
        self.device         = device
        self.loss_fn        = DistillationLoss(cfg.distill)

        # Inject LoRA into recovery base; only LoRA weights require grad
        self.recovery_model = add_recovery_lora(recovery_base, cfg.recovery)

        # Default optimizers if not provided: AdamW on each set of trainable params
        self.sft_optimizer  = sft_optimizer or torch.optim.AdamW(
            [p for p in student.parameters() if p.requires_grad], lr=1e-4
        )
        self.lora_params = [p for p in self.recovery_model.parameters() if p.requires_grad]
        self.lora_optimizer = lora_optimizer or torch.optim.AdamW(self.lora_params, lr=2e-4)

        self._last_sft_info  = {}
        self._last_lora_info = {}

    @torch.no_grad()
    def _teacher_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.teacher.eval()
        out = self.teacher(input_ids.to(self.device))
        if hasattr(out, "logits"):
            return out.logits
        return out

    def step(
        self,
        input_ids: torch.Tensor,
        n_loops:   int = 8,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        One Chimera step.

        1. Run teacher once → teacher_logits (no grad, shared by both pipelines)
        2. SFT student forward + backward
        3. Recovery LoRA forward + backward
        4. Return (sft_loss, lora_loss, combined_info)
           — callers call step_optimizers() after this to update weights
        """
        input_ids = input_ids.to(self.device)
        labels    = input_ids[:, 1:]

        # ── 1. Teacher (single inference) ────────────────────────────────
        teacher_logits = self._teacher_logits(input_ids)[:, :-1, :]  # align positions

        # ── 2. SFT student ────────────────────────────────────────────────
        self.student.train()
        self.sft_optimizer.zero_grad()

        sft_logits, sft_aux = self.student(input_ids[:, :-1], n_loops=n_loops, return_aux=True)
        sft_loss, sft_info  = self.loss_fn(sft_logits, teacher_logits, labels)
        sft_total           = sft_loss + sft_aux
        sft_total.backward()

        # ── 3. Recovery LoRA ──────────────────────────────────────────────
        self.recovery_model.train()
        self.lora_optimizer.zero_grad()

        lora_out = self.recovery_model(input_ids[:, :-1])
        # recovery_model may be an Anthos or a HF CausalLM — unpack accordingly
        if isinstance(lora_out, tuple):
            lora_logits = lora_out[0]
        elif hasattr(lora_out, "logits"):
            lora_logits = lora_out.logits
        else:
            lora_logits = lora_out

        lora_loss, lora_info = self.loss_fn(lora_logits, teacher_logits, labels)
        (lora_loss * self.cfg.recovery.loss_coef).backward()

        self._last_sft_info  = {f"sft/{k}":  v for k, v in sft_info.items()}
        self._last_lora_info = {f"lora/{k}": v for k, v in lora_info.items()}

        return sft_total, lora_loss, {**self._last_sft_info, **self._last_lora_info}

    def step_optimizers(
        self,
        grad_clip: float = 1.0,
    ) -> None:
        """Clip gradients and step both optimizers. Call after step()."""
        nn.utils.clip_grad_norm_(self.student.parameters(),        grad_clip)
        nn.utils.clip_grad_norm_(self.lora_params,                 grad_clip)
        self.sft_optimizer.step()
        self.lora_optimizer.step()

    def save(self, step: int) -> tuple[str, str]:
        """
        Save SFT student and LoRA-only delta to separate subdirs under run_id.

        Returns (sft_path, lora_path).
        """
        import json
        from pathlib import Path

        run_dir  = Path(self.cfg.checkpoint_dir) / self.cfg.run_id
        sft_dir  = run_dir / "sft"
        lora_dir = run_dir / "lora"
        sft_dir.mkdir(parents=True, exist_ok=True)
        lora_dir.mkdir(parents=True, exist_ok=True)

        sft_path  = sft_dir  / f"step_{step:06d}.pt"
        lora_path = lora_dir / f"step_{step:06d}_lora.pt"

        torch.save(self.student.state_dict(),                sft_path)
        torch.save(
            {k: v for k, v in self.recovery_model.state_dict().items()
             if "lora_A" in k or "lora_B" in k},
            lora_path,
        )

        meta = {
            "run_id":     self.cfg.run_id,
            "step":       step,
            "sft_path":   str(sft_path),
            "lora_path":  str(lora_path),
            "lora_rank":  self.cfg.recovery.rank,
            "lora_alpha": self.cfg.recovery.alpha,
        }
        (run_dir / f"step_{step:06d}_meta.json").write_text(
            json.dumps(meta, indent=2)
        )

        return str(sft_path), str(lora_path)
