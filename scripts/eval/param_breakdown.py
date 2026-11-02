"""Closed-form parameter count for an Anthos config, checked against the instantiated model.

    python param_breakdown.py --tier proof --vocab-size 100352

Exits non-zero if the formula and the real model disagree, so it can gate a paid run.
"""
import argparse
import sys

from anthos.configs import get_training_config
import anthos.main as m


def expected_params(c) -> dict:
    d, kvd = c.dim, c.dim // c.n_heads * c.n_kv_heads
    attn = 2 * d * d + 2 * d * kvd
    dense_block = attn + 3 * d * (d * 4 // 3) + 2 * d
    moe_ffn = c.n_experts * 3 * d * c.expert_dim + c.n_shared_experts * 3 * d * (c.expert_dim * c.n_experts_per_tok) + d * c.n_experts
    inj = 3 * d * d + 3 * d + 1
    parts = {
        "embedding (tied head)": c.vocab_size * d,
        "thought_pool": c.n_thought_tokens * d,
        "prelude": c.prelude_layers * dense_block,
        "coda": c.coda_layers * dense_block,
        "recurrent: attn + MoE block": attn + moe_ffn + 2 * d,
        "recurrent: 4 norms": 4 * d,
        "recurrent: LoRA (fast+deep+temp)": 2 * (2 * c.lora_rank * d + d) + 1,
        "recurrent: memory bank": 10 * d * d + d + 4 * d,
        "recurrent: seq+thought injection": 2 * inj,
        "recurrent: prerouter head": d * 2 * d + 2 * d * c.n_experts,
        "recurrent: ACT halt": d + 1,
        "final norm": d,
    }
    return parts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="proof")
    ap.add_argument("--vocab-size", type=int, default=None)
    a = ap.parse_args()
    cfg, _ = get_training_config(a.tier)
    if a.vocab_size:
        cfg.vocab_size = a.vocab_size
    parts = expected_params(cfg)
    exp = sum(parts.values())
    act = sum(p.numel() for p in m.Anthos(cfg).parameters())
    for k, v in parts.items():
        print(f"  {k:36s}{v:>13,}")
    print(f"  {'expected total':36s}{exp:>13,}\n  {'actual total':36s}{act:>13,}")
    sys.exit(0 if exp == act else 1)
