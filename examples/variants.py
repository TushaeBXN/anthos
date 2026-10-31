"""
Anthos — Model variant showcase
All pre-configured sizes from 1B to 100B.
"""

import torch
from anthos import Anthos, anthos_1b, anthos_3b, anthos_10b, anthos_50b, anthos_100b

variants = {
    "anthos_1b":   anthos_1b(),
    "anthos_3b":   anthos_3b(),
    "anthos_10b":  anthos_10b(),
    "anthos_50b":  anthos_50b(),
    "anthos_100b": anthos_100b(),
}

print(f"{'Variant':<14} {'Params':>12}  {'dim':>6}  {'Experts':>8}  {'Thoughts':>9}  {'Loops':>6}  {'Context':>8}")
print("─" * 75)

for name, cfg in variants.items():
    model  = Anthos(cfg)
    total  = sum(p.numel() for p in model.parameters())
    ctx    = f"{cfg.max_seq_len // 1024}k"
    print(f"{name:<14} {total:>12,}  {cfg.dim:>6}  {cfg.n_experts:>8}  {cfg.n_thought_tokens:>9}  {cfg.max_loop_iters:>6}  {ctx:>8}")
    del model   # free memory before next variant
