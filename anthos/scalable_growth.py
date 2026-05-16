"""Scale Anthos from 1B to 100B without retraining from scratch"""
import torch
import torch.nn as nn
from typing import Optional
import math


class MoELayer(nn.Module):
    """Expandable Mixture-of-Experts layer"""

    def __init__(self, dim: int, n_experts: int, expert_dim: int):
        super().__init__()
        self.n_experts = n_experts
        self.dim = dim
        self.expert_dim = expert_dim
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, expert_dim),
                nn.GELU(),
                nn.Linear(expert_dim, dim)
            ) for _ in range(n_experts)
        ])
        self.gate = nn.Linear(dim, n_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.softmax(self.gate(x), dim=-1)
        top2_gates, top2_indices = gates.topk(2, dim=-1)
        top2_gates = top2_gates / top2_gates.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(x)
        for i in range(2):
            expert_idx = top2_indices[..., i]
            gate_weight = top2_gates[..., i:i+1]
            for j in range(self.n_experts):
                mask = (expert_idx == j)
                if mask.any():
                    output[mask] += gate_weight[mask] * self.experts[j](x[mask])
        return output

    def expand_experts(self, new_count: int):
        if new_count <= self.n_experts:
            return
        for _ in range(self.n_experts, new_count):
            new_expert = nn.Sequential(
                nn.Linear(self.experts[0][0].in_features, self.experts[0][0].out_features),
                nn.GELU(),
                nn.Linear(self.experts[0][2].in_features, self.experts[0][2].out_features)
            )
            for param in new_expert.parameters():
                nn.init.normal_(param, std=0.01)
            self.experts.append(new_expert)

        old_gate_weight = self.gate.weight.data
        old_gate_bias = self.gate.bias.data if self.gate.bias is not None else None
        new_gate = nn.Linear(old_gate_weight.shape[1], new_count)
        new_gate.weight.data[:old_gate_weight.shape[0]] = old_gate_weight
        new_gate.weight.data[old_gate_weight.shape[0]:] = torch.randn(
            new_count - old_gate_weight.shape[0], old_gate_weight.shape[1]
        ) * 0.01
        if old_gate_bias is not None:
            new_gate.bias.data[:len(old_gate_bias)] = old_gate_bias
        self.gate = new_gate
        self.n_experts = new_count


class RecurrentBlock(nn.Module):
    """Expandable recurrent block"""

    def __init__(self, dim: int, n_heads: int, n_experts: int = 0, expert_dim: int = 256):
        super().__init__()
        self.dim = dim
        self.attention = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.moe = MoELayer(dim, n_experts, expert_dim) if n_experts > 1 else None
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        ) if n_experts <= 1 else None
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.moe(x) if self.moe is not None else self.ffn(x)
        x = self.norm2(x + ff_out)
        return x

    def expand_dim(self, old_dim: int, new_dim: int):
        self.dim = new_dim

        # Rebuild attention with new dim
        n_heads = self.attention.num_heads
        new_attn = nn.MultiheadAttention(new_dim, n_heads, batch_first=True)
        self.attention = new_attn

        # Expand layer norms
        self.norm1 = nn.LayerNorm(new_dim)
        self.norm2 = nn.LayerNorm(new_dim)

        # Expand FFN if present
        if self.ffn is not None:
            self.ffn = nn.Sequential(
                nn.Linear(new_dim, new_dim * 4),
                nn.GELU(),
                nn.Linear(new_dim * 4, new_dim)
            )


class ScalableAnthos(nn.Module):
    """Anthos model with built-in growth paths from 1B to 100B"""

    EXPANSION_MAP = {
        "3B":  {"dim": 3072, "n_layers": 26, "n_experts": 96,  "thought_tokens": 24},
        "10B": {"dim": 4096, "n_layers": 32, "n_experts": 128, "thought_tokens": 32},
        "50B": {"dim": 6144, "n_layers": 40, "n_experts": 256, "thought_tokens": 48},
        "100B":{"dim": 8192, "n_layers": 48, "n_experts": 256, "thought_tokens": 64},
    }

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._build_architecture(config)
        self.current_size = self._size_label()

    def _build_architecture(self, cfg):
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.thought_embedding = nn.Parameter(
            torch.randn(cfg.n_thought_tokens, cfg.dim) * 0.02
        )
        self.recurrent_blocks = nn.ModuleList([
            RecurrentBlock(
                cfg.dim,
                cfg.n_heads,
                n_experts=getattr(cfg, "n_experts", 0),
                expert_dim=getattr(cfg, "expert_dim", 256)
            )
            for _ in range(cfg.n_layers)
        ])
        self.norm = nn.LayerNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, return_aux: bool = False):
        x = self.token_embedding(input_ids)

        # Prepend thought tokens
        batch = x.shape[0]
        thoughts = self.thought_embedding.unsqueeze(0).expand(batch, -1, -1)
        x = torch.cat([thoughts, x], dim=1)

        for block in self.recurrent_blocks:
            x = block(x)

        x = self.norm(x)
        # Remove thought tokens before projection
        x = x[:, self.config.n_thought_tokens:, :]
        logits = self.lm_head(x)

        if return_aux:
            return logits, torch.tensor(0.0, device=logits.device)
        return logits

    def expand_to_size(self, target_params: str, verbose: bool = True):
        if target_params not in self.EXPANSION_MAP:
            raise ValueError(f"Unknown size '{target_params}'. Choose: {list(self.EXPANSION_MAP.keys())}")

        new_cfg = self.EXPANSION_MAP[target_params]
        old_dim = self.config.dim
        old_n_layers = len(self.recurrent_blocks)

        if verbose:
            print(f"Growing from {self.current_size} to {target_params}")
            print(f"  dim: {old_dim} -> {new_cfg['dim']}")
            print(f"  layers: {old_n_layers} -> {new_cfg['n_layers']}")

        self._expand_dimension(old_dim, new_cfg["dim"])
        self._add_layers(old_n_layers, new_cfg["n_layers"], new_cfg["dim"])
        self._expand_thought_tokens(new_cfg["thought_tokens"])

        n_experts = new_cfg.get("n_experts", 0)
        if n_experts > 1:
            self._expand_experts(n_experts)

        self.config.dim = new_cfg["dim"]
        self.config.n_layers = new_cfg["n_layers"]
        self.config.n_experts = n_experts
        self.config.n_thought_tokens = new_cfg["thought_tokens"]
        self.current_size = target_params

        if verbose:
            print(f"Growth complete! New param count: {self._compute_params():.2f}B")

        return self

    def _expand_dimension(self, old_dim: int, new_dim: int):
        old_emb = self.token_embedding.weight.data
        new_emb = torch.zeros(old_emb.shape[0], new_dim)
        new_emb[:, :old_dim] = old_emb
        new_emb[:, old_dim:] = torch.randn(old_emb.shape[0], new_dim - old_dim) * 0.02
        self.token_embedding = nn.Embedding.from_pretrained(new_emb, freeze=False)

        old_thought = self.thought_embedding.data
        new_thought = torch.zeros(self.config.n_thought_tokens, new_dim)
        new_thought[:, :old_dim] = old_thought
        new_thought[:, old_dim:] = torch.randn(self.config.n_thought_tokens, new_dim - old_dim) * 0.02
        self.thought_embedding = nn.Parameter(new_thought)

        new_head = torch.zeros(self.config.vocab_size, new_dim)
        old_head = self.lm_head.weight.data
        new_head[:, :old_dim] = old_head
        new_head[:, old_dim:] = torch.randn(self.config.vocab_size, new_dim - old_dim) * 0.02
        self.lm_head = nn.Linear(new_dim, self.config.vocab_size, bias=False)
        self.lm_head.weight.data = new_head

        self.norm = nn.LayerNorm(new_dim)

        for block in self.recurrent_blocks:
            block.expand_dim(old_dim, new_dim)

    def _add_layers(self, old_n_layers: int, new_n_layers: int, dim: int):
        for _ in range(old_n_layers, new_n_layers):
            new_block = RecurrentBlock(
                dim,
                self.config.n_heads,
                n_experts=getattr(self.config, "n_experts", 0),
                expert_dim=getattr(self.config, "expert_dim", 256)
            )
            for param in new_block.parameters():
                if param.dim() >= 2:
                    nn.init.orthogonal_(param, gain=0.1)
            self.recurrent_blocks.append(new_block)

    def _expand_thought_tokens(self, new_count: int):
        if new_count <= self.config.n_thought_tokens:
            return
        old_thought = self.thought_embedding.data
        new_thought = torch.zeros(new_count, old_thought.shape[1])
        new_thought[:self.config.n_thought_tokens] = old_thought
        new_thought[self.config.n_thought_tokens:] = torch.randn(
            new_count - self.config.n_thought_tokens, old_thought.shape[1]
        ) * 0.02
        self.thought_embedding = nn.Parameter(new_thought)

    def _expand_experts(self, new_expert_count: int):
        for block in self.recurrent_blocks:
            if block.moe is not None:
                block.moe.expand_experts(new_expert_count)

    def _compute_params(self) -> float:
        return sum(p.numel() for p in self.parameters()) / 1e9

    def _size_label(self) -> str:
        params = self._compute_params()
        if params < 2:
            return "1B"
        elif params < 5:
            return "3B"
        elif params < 20:
            return "10B"
        elif params < 75:
            return "50B"
        else:
            return "100B"


if __name__ == "__main__":
    from anthos import AnthosConfig

    cfg = AnthosConfig(
        vocab_size=32008,
        dim=2048,
        n_layers=24,
        n_heads=16,
        n_thought_tokens=16,
        n_experts=64,
        expert_dim=256,
    )
    model = ScalableAnthos(cfg)
    print(f"Initial size: {model._compute_params():.2f}B params")

    model.expand_to_size("3B")
    print(f"After growth: {model._compute_params():.2f}B params")
