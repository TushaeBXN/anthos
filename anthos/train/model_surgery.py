"""Modify Anthos architecture without retraining"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ModelSurgeon:
    """Perform surgery on trained models to add/remove capabilities"""

    @staticmethod
    def add_language_adapter(model: nn.Module, dim: int, adapter_size: int = 256) -> nn.Module:
        """Add a language-specific LoRA adapter"""
        class LanguageAdapter(nn.Module):
            def __init__(self, dim: int, adapter_size: int):
                super().__init__()
                self.down = nn.Linear(dim, adapter_size, bias=False)
                self.up = nn.Linear(adapter_size, dim, bias=False)
                nn.init.normal_(self.down.weight, std=0.02)
                nn.init.zeros_(self.up.weight)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + self.up(F.gelu(self.down(x)))

        model.language_adapter = LanguageAdapter(dim, adapter_size)
        return model

    @staticmethod
    def prune_unused_experts(model: nn.Module, threshold: float = 0.01) -> nn.Module:
        """Remove underutilized MoE experts to save memory"""
        for name, module in model.named_modules():
            if hasattr(module, "moe") and hasattr(module.moe, "experts"):
                moe = module.moe
                n = len(moe.experts)
                # Heuristic: remove last 10% if threshold-based pruning data unavailable
                keep_count = max(1, int(n * (1 - threshold)))
                if keep_count < n:
                    moe.experts = nn.ModuleList(list(moe.experts)[:keep_count])
                    old_gate = moe.gate
                    new_gate = nn.Linear(old_gate.in_features, keep_count, bias=old_gate.bias is not None)
                    new_gate.weight.data = old_gate.weight.data[:keep_count]
                    if old_gate.bias is not None:
                        new_gate.bias.data = old_gate.bias.data[:keep_count]
                    moe.gate = new_gate
                    moe.n_experts = keep_count
                    print(f"Pruned {name}.moe: {n} -> {keep_count} experts")
        return model

    @staticmethod
    def merge_models(model1: nn.Module, model2: nn.Module, merge_ratio: float = 0.5) -> nn.Module:
        """Merge two trained models via weight interpolation"""
        state1 = model1.state_dict()
        state2 = model2.state_dict()

        merged = {}
        for key in state1:
            if key in state2 and state1[key].shape == state2[key].shape:
                merged[key] = merge_ratio * state1[key] + (1 - merge_ratio) * state2[key]
            else:
                merged[key] = state1[key]

        model1.load_state_dict(merged, strict=False)
        print(f"Merged models with ratio {merge_ratio}")
        return model1

    @staticmethod
    def distill_to_smaller(
        teacher_model: nn.Module,
        student_model: nn.Module,
        dataloader,
        optimizer,
        steps: int = 1000,
        temperature: float = 2.0,
        device: str = "cuda",
    ) -> nn.Module:
        """Knowledge distillation from teacher to smaller student"""
        teacher_model.eval()
        student_model.train()

        for step, batch in enumerate(dataloader):
            if step >= steps:
                break

            input_ids = batch["input_ids"].to(device)

            with torch.no_grad():
                teacher_out = teacher_model(input_ids)
                if isinstance(teacher_out, tuple):
                    teacher_logits = teacher_out[0]
                else:
                    teacher_logits = teacher_out

            student_out = student_model(input_ids)
            if isinstance(student_out, tuple):
                student_logits = student_out[0]
            else:
                student_logits = student_out

            loss = nn.KLDivLoss(reduction="batchmean")(
                F.log_softmax(student_logits / temperature, dim=-1),
                F.softmax(teacher_logits / temperature, dim=-1),
            ) * (temperature ** 2)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % 100 == 0:
                print(f"Distillation step {step}/{steps} | loss: {loss.item():.4f}")

        return student_model
