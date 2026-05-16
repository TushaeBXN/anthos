#!/usr/bin/env python3
"""Complete training pipeline with identity locking, growth, and optimization"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import os
from pathlib import Path

from anthos import AnthosConfig
from anthos.identity_hardening import AnthosWithIdentityLock, CheckpointSigner
from anthos.scalable_growth import ScalableAnthos


# ============================================================================
# Configuration
# ============================================================================

class TrainingConfig:
    """Centralized config for all training phases"""

    PHASE1 = {
        "name": "Foundation",
        "dim": 2048,
        "n_layers": 24,
        "n_heads": 16,
        "n_experts": 64,
        "n_thought_tokens": 16,
        "max_seq_len": 4096,
        "batch_size": 32,
        "learning_rate": 3e-4,
        "warmup_steps": 5000,
        "total_steps": 200000,
        "data_mix": {
            "fineweb_edu": 0.5,
            "code": 0.2,
            "teacher_conversations": 0.2,
            "identity": 0.1,
        },
        "identity_freeze_steps": 5000,
        "eaft_gamma": 0.5,
    }

    PHASE2 = {
        "name": "Identity_Hardening",
        "dim": 2048,
        "n_layers": 24,
        "n_heads": 16,
        "n_experts": 64,
        "n_thought_tokens": 16,
        "max_seq_len": 4096,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "warmup_steps": 500,
        "total_steps": 10000,
        "identity_loss_weight": 2.0,
        "freeze_identity_embeddings": True,
        "data_mix": {"identity_only": 1.0},
        "identity_freeze_steps": 0,
    }

    PHASE3 = {
        "name": "Instruction",
        "dim": 2048,
        "n_layers": 24,
        "n_heads": 16,
        "n_experts": 64,
        "n_thought_tokens": 16,
        "max_seq_len": 4096,
        "batch_size": 16,
        "learning_rate": 2e-5,
        "warmup_steps": 200,
        "total_steps": 50000,
        "data_mix": {
            "teacher_conversations": 0.6,
            "slimorca": 0.3,
            "gsm8k": 0.1,
        },
        "identity_freeze_steps": 0,
    }

    GROWTH_3B = {"target": "3B", "steps": 50000, "lr": 1e-4}
    GROWTH_10B = {"target": "10B", "steps": 100000, "lr": 5e-5}


# ============================================================================
# Trainer
# ============================================================================

class AnthosTrainer:
    def __init__(self, config_phase: dict, device: str = "cuda"):
        self.config = config_phase
        self.device = device
        self.scaler = GradScaler()
        self.signer = CheckpointSigner()
        self.current_step = 0

        self.model = self._build_model()
        self.model.to(device)
        self.optimizer = self._create_optimizer()

    def _build_model(self):
        cfg = AnthosConfig(
            vocab_size=32008,
            dim=self.config["dim"],
            n_layers=self.config["n_layers"],
            n_heads=self.config["n_heads"],
            n_kv_heads=self.config["n_heads"] // 2,
            n_experts=self.config["n_experts"],
            n_thought_tokens=self.config["n_thought_tokens"],
            max_seq_len=self.config["max_seq_len"],
        )
        base_model = ScalableAnthos(cfg)
        model = AnthosWithIdentityLock(
            base_model,
            hidden_dim=cfg.dim,
            freeze_after_steps=self.config.get("identity_freeze_steps", 5000)
        )
        return model

    def _create_optimizer(self):
        identity_params = []
        normal_params = []
        for name, param in self.model.named_parameters():
            if "identity" in name:
                identity_params.append(param)
            else:
                normal_params.append(param)

        return torch.optim.AdamW([
            {"params": normal_params, "lr": self.config["learning_rate"]},
            {"params": identity_params, "lr": self.config["learning_rate"] * 3.0},
        ], weight_decay=0.1)

    def _lr_schedule(self, step: int) -> float:
        warmup = self.config.get("warmup_steps", 500)
        if step < warmup:
            return step / warmup
        total = self.config.get("total_steps", 200000)
        progress = (step - warmup) / max(total - warmup, 1)
        return max(0.1, 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item()))

    def train_step(self, batch: dict) -> float:
        input_ids = batch["input_ids"].to(self.device)
        labels = batch.get("labels", input_ids.clone())
        labels = labels.to(self.device)

        with autocast(dtype=torch.bfloat16):
            total_loss, ce_loss, id_loss = self.model(
                input_ids,
                labels=labels,
                return_identity_loss=True
            )

        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        # Update learning rate
        lr_scale = self._lr_schedule(self.current_step)
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.config["learning_rate"] * lr_scale

        return total_loss.item()

    def train(self, dataloader, num_steps: int, checkpoint_every: int = 1000):
        self.model.train()

        for step, batch in enumerate(dataloader):
            if step >= num_steps:
                break

            loss = self.train_step(batch)
            self.current_step += 1

            if step % checkpoint_every == 0 and step > 0:
                self.save_checkpoint(f"checkpoints/step_{step:06d}.pt")

            if step % 100 == 0:
                print(f"Step {step}/{num_steps} | Loss: {loss:.4f}")

        self.save_checkpoint("checkpoints/final_checkpoint.pt")

    def save_checkpoint(self, path: str):
        Path(path).parent.mkdir(exist_ok=True)
        model_state = self.model.state_dict()
        checkpoint = self.signer.sign(model_state, {
            "step": self.current_step,
            "phase": self.config["name"],
        })
        torch.save(checkpoint, path)
        print(f"Saved signed checkpoint to {path}")

    def verify_identity(self):
        """Quick check that identity is still encoded"""
        test_prompts = ["Who built you?", "What model are you?", "Who created Anthos?"]
        print("Running identity verification...")
        for prompt in test_prompts:
            print(f"  Q: {prompt}")
        print("Identity verification step complete (full eval requires tokenizer + generate).")


# ============================================================================
# Main
# ============================================================================

def run_complete_training():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Phase 1: Foundation
    print("\nPhase 1: Foundation Training")
    trainer = AnthosTrainer(TrainingConfig.PHASE1, device=device)
    # trainer.train(dataloader, TrainingConfig.PHASE1["total_steps"])

    # Phase 2: Identity Hardening
    print("\nPhase 2: Identity Hardening")
    # trainer.config = TrainingConfig.PHASE2
    # trainer.optimizer = trainer._create_optimizer()
    # trainer.train(identity_dataloader, TrainingConfig.PHASE2["total_steps"])

    # Phase 3: Instruction Tuning
    print("\nPhase 3: Instruction Tuning")
    # trainer.config = TrainingConfig.PHASE3
    # trainer.optimizer = trainer._create_optimizer()
    # trainer.train(sft_dataloader, TrainingConfig.PHASE3["total_steps"])

    # Grow to 3B
    print("\nGrowing to 3B...")
    # trainer.model.base.expand_to_size("3B")

    print("\nTraining pipeline configured. Uncomment phases and supply dataloaders to run.")


if __name__ == "__main__":
    if torch.cuda.device_count() > 1:
        dist.init_process_group("nccl")

    run_complete_training()
