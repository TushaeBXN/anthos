"""
train_on_runpod.py — Complete RunPod training script for Anthos

Optimized for A100 / H100 single or multi-GPU pods.
Streams FineWeb-Edu directly — no disk download needed.
Generates Claude Haiku teacher data on-demand.
Saves signed checkpoints every N steps.

Usage:
    # Single GPU
    python train_on_runpod.py

    # Multi-GPU (4 GPUs)
    torchrun --nproc_per_node=4 train_on_runpod.py

Required env vars:
    ANTHROPIC_API_KEY  — for teacher data generation
    HF_TOKEN           — for FineWeb-Edu streaming
    WANDB_API_KEY      — optional, for training dashboard

Cost estimate on RunPod:
    A100 40GB: ~$1.89/hr, ~3-5 days = ~$135-$225 for 200k steps
    H100:      ~$3.89/hr, ~2-3 days = ~$190-$280 for 200k steps
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, IterableDataset
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer
from pathlib import Path
from tqdm import tqdm
import json
import random

from anthos import Anthos, AnthosConfig
from anthos.identity_hardening import AnthosWithIdentityLock, CheckpointSigner
from anthos.data_pipeline import AnthosDataPipeline, _to_conversation
from anthos.stream_fineweb import StreamingFineWebDataset

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
RUNPOD_CONFIG = {
    # Model
    "size":            "1B",
    "dim":             2048,
    "n_heads":         16,
    "n_kv_heads":      8,
    "n_experts":       64,
    "n_thought_tokens": 16,
    "max_loop_iters":  16,
    "max_seq_len":     4096,
    "prelude_layers":  2,
    "coda_layers":     2,

    # Training
    "batch_size":          8,       # per GPU — effective batch with grad_accum = 32
    "learning_rate":       3e-4,
    "warmup_steps":        5000,
    "total_steps":         200000,
    "grad_accum_steps":    4,
    "grad_clip":           1.0,
    "checkpoint_steps":    5000,
    "identity_freeze_steps": 5000,

    # Data
    "fineweb_limit":    2_000_000,  # stream up to 2M examples
    "teacher_examples": 50_000,
    "identity_repeat":  5_000,
}


# ─────────────────────────────────────────────────────────────────────────────
# Simple list dataset for teacher + identity data
# ─────────────────────────────────────────────────────────────────────────────
class ConversationDataset(IterableDataset):
    """Wraps a list of conversation dicts into a PyTorch dataset."""

    def __init__(self, examples: list, tokenizer, max_seq_len: int = 4096):
        self.examples    = examples
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len

    def _format_text(self, example: dict) -> str:
        convs = example.get("conversations", [])
        parts = []
        for c in convs:
            role = c.get("from", "")
            val  = c.get("value", "")
            if role == "system":
                parts.append(f"<|system|>{val}<|end|>")
            elif role == "human":
                parts.append(f"<|user|>{val}<|end|>")
            elif role == "gpt":
                parts.append(f"<|assistant|>{val}<|end|>")
        return "\n".join(parts)

    def _tokenize(self, text: str) -> dict:
        tokens = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        ids = tokens["input_ids"][0]
        return {
            "input_ids":      ids,
            "attention_mask": tokens["attention_mask"][0],
            "labels":         ids.clone(),
        }

    def __iter__(self):
        shuffled = self.examples[:]
        random.shuffle(shuffled)
        for ex in shuffled:
            try:
                text = self._format_text(ex)
                if len(text) < 20:
                    continue
                yield self._tokenize(text)
            except Exception:
                continue


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────
class RunPodTrainer:
    """Optimized for RunPod A100 / H100 instances."""

    def __init__(self, config: dict):
        self.config       = config
        self.global_step  = 0
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler       = GradScaler()
        self.signer       = CheckpointSigner()
        self.is_ddp       = dist.is_available() and dist.is_initialized()
        self.is_main      = (not self.is_ddp) or (dist.get_rank() == 0)

        # Enable TF32 on A100/H100 (2× matmul throughput)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32        = True

        # Tokenizer
        tokenizer_name = os.environ.get("TOKENIZER", "gpt2")
        print(f"  Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model
        self.model = self._build_model()
        self.model.to(self.device)

        # Multi-GPU via DDP
        if self.is_ddp:
            local_rank  = dist.get_rank() % torch.cuda.device_count()
            self.model  = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[local_rank]
            )
            if self.is_main:
                print(f"  DDP: {dist.get_world_size()} GPUs")
        elif torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            print(f"  DataParallel: {torch.cuda.device_count()} GPUs")

        # Optimizer — try 8-bit Adam, fall back to standard AdamW
        try:
            from bitsandbytes.optim import Adam8bit
            self.optimizer = Adam8bit(
                self.model.parameters(),
                lr=config["learning_rate"],
                weight_decay=0.1,
            )
            if self.is_main:
                print("  Optimizer: 8-bit Adam (bitsandbytes)")
        except ImportError:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=config["learning_rate"],
                weight_decay=0.1,
            )
            if self.is_main:
                print("  Optimizer: AdamW (bitsandbytes not installed)")

        # LR scheduler
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config["warmup_steps"],
            num_training_steps=config["total_steps"],
        )

        # WandB — optional
        self.wandb_enabled = False
        if os.environ.get("WANDB_API_KEY") and self.is_main:
            try:
                import wandb
                wandb.init(
                    project="anthos-runpod",
                    name=f"anthos_{config['size']}",
                    config=config,
                )
                self.wandb_enabled = True
                print("  WandB: enabled")
            except ImportError:
                print("  WandB: not installed (pip install wandb)")

    def _build_model(self) -> nn.Module:
        cfg = AnthosConfig(
            vocab_size        = 32008,
            dim               = self.config["dim"],
            n_heads           = self.config["n_heads"],
            n_kv_heads        = self.config["n_kv_heads"],
            n_experts         = self.config["n_experts"],
            n_thought_tokens  = self.config["n_thought_tokens"],
            max_loop_iters    = self.config["max_loop_iters"],
            max_seq_len       = self.config["max_seq_len"],
            prelude_layers    = self.config["prelude_layers"],
            coda_layers       = self.config["coda_layers"],
        )

        base_model = Anthos(cfg)
        model = AnthosWithIdentityLock(
            base_model,
            hidden_dim         = cfg.dim,
            freeze_after_steps = self.config["identity_freeze_steps"],
        )

        params = sum(p.numel() for p in model.parameters()) / 1e9
        if self.is_main:
            print(f"  Model: {params:.2f}B parameters")

        return model

    def _build_dataloaders(self) -> dict:
        """Build all three dataloaders."""
        pipeline = AnthosDataPipeline(max_seq_len=self.config["max_seq_len"])

        # FineWeb — streaming, no disk
        fineweb_ds = StreamingFineWebDataset(
            tokenizer   = self.tokenizer,
            max_seq_len = self.config["max_seq_len"],
            max_samples = self.config["fineweb_limit"],
            hf_token    = os.environ.get("HF_TOKEN"),
        )
        fineweb_loader = DataLoader(
            fineweb_ds,
            batch_size  = self.config["batch_size"],
            num_workers = 4,
            pin_memory  = True,
        )

        # Teacher + identity — list-based
        data_mix = pipeline.create_training_mix(
            fineweb_limit    = 0,                               # already handled above
            teacher_examples = self.config["teacher_examples"],
            identity_repeat  = self.config["identity_repeat"],
        )

        teacher_ds = ConversationDataset(
            data_mix["teacher"], self.tokenizer, self.config["max_seq_len"]
        )
        identity_ds = ConversationDataset(
            data_mix["identity"], self.tokenizer, self.config["max_seq_len"]
        )

        teacher_loader  = DataLoader(teacher_ds,  batch_size=self.config["batch_size"],
                                     num_workers=4, pin_memory=True)
        identity_loader = DataLoader(identity_ds, batch_size=self.config["batch_size"],
                                     num_workers=2, pin_memory=True)

        return {
            "fineweb":  (fineweb_loader,  0.50),
            "teacher":  (teacher_loader,  0.40),
            "identity": (identity_loader, 0.10),
        }

    def _train_step(self, batch: dict) -> tuple[float, float, float]:
        input_ids = batch["input_ids"].to(self.device)
        labels    = batch.get("labels", input_ids.clone()).to(self.device)

        with autocast(dtype=torch.bfloat16):
            total_loss, ce_loss, id_loss = self.model(
                input_ids,
                labels               = labels,
                return_identity_loss = True,
            )

        self.scaler.scale(total_loss / self.config["grad_accum_steps"]).backward()
        return total_loss.item(), ce_loss.item(), id_loss.item()

    def train(self):
        """Main training loop."""
        if self.is_main:
            print(f"\n  Starting training — {self.config['total_steps']:,} steps")

        loaders = self._build_dataloaders()
        self.model.train()

        # Build interleaved iterator based on weights
        def interleaved_batches():
            iters = {
                name: iter(loader)
                for name, (loader, _) in loaders.items()
            }
            weights = [w for _, (_, w) in loaders.items()]
            names   = list(loaders.keys())

            while True:
                name = random.choices(names, weights=weights, k=1)[0]
                try:
                    batch = next(iters[name])
                    yield name, batch
                except StopIteration:
                    iters[name] = iter(loaders[name][0])
                    try:
                        batch = next(iters[name])
                        yield name, batch
                    except StopIteration:
                        break

        accum_loss = accum_ce = accum_id = 0.0
        accum_step = 0
        pbar = tqdm(total=self.config["total_steps"], desc="Anthos RunPod") if self.is_main else None

        for name, batch in interleaved_batches():
            if self.global_step >= self.config["total_steps"]:
                break

            loss, ce, id_l = self._train_step(batch)
            accum_loss += loss
            accum_ce   += ce
            accum_id   += id_l
            accum_step += 1

            if accum_step >= self.config["grad_accum_steps"]:
                # Gradient clip + optimizer step
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["grad_clip"]
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()

                self.global_step += 1
                accum_step = 0

                # Logging
                if self.is_main:
                    avg_loss = accum_loss / self.config["grad_accum_steps"]
                    avg_ce   = accum_ce   / self.config["grad_accum_steps"]
                    avg_id   = accum_id   / self.config["grad_accum_steps"]
                    accum_loss = accum_ce = accum_id = 0.0

                    if pbar:
                        pbar.set_postfix(loss=f"{avg_loss:.4f}", src=name)
                        pbar.update(1)

                    if self.wandb_enabled:
                        import wandb
                        wandb.log({
                            "loss":    avg_loss,
                            "ce_loss": avg_ce,
                            "id_loss": avg_id,
                            "lr":      self.scheduler.get_last_lr()[0],
                            "step":    self.global_step,
                            "source":  name,
                        })

                    # Checkpoint
                    if self.global_step % self.config["checkpoint_steps"] == 0:
                        self.save_checkpoint(
                            f"checkpoints/anthos-runpod/step_{self.global_step:06d}.pt"
                        )

        if pbar:
            pbar.close()

        if self.is_main:
            self.save_checkpoint("checkpoints/anthos-runpod/final.pt")
            print("\n  Training complete!")

    def save_checkpoint(self, path: str):
        """Save a cryptographically signed checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        raw = self.model
        if hasattr(raw, "module"):
            raw = raw.module   # unwrap DDP / DataParallel

        state = raw.state_dict()
        checkpoint = self.signer.sign(state, {
            "step":   self.global_step,
            "config": self.config,
        })
        torch.save(checkpoint, path)
        print(f"  Checkpoint saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Multi-GPU init
    if torch.cuda.device_count() > 1 and "LOCAL_RANK" in os.environ:
        dist.init_process_group("nccl")

    trainer = RunPodTrainer(RUNPOD_CONFIG)
    trainer.train()
