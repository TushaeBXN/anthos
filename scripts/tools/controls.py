"""Control and ablation runs for the dim-768 pretraining (run 2 = anthos-pt-768). Prints exact parameter counts and a
rough cost per run, and the train.py command for each.

    python controls.py                  # table + counts (instantiates each model on CPU)
    python controls.py --cmd dense_ffn  # the launch command for one control

Every control keeps run 2's data, order (seed 0), steps (30,000), batch (32,768 tokens), learning rate and schedule, so
the only thing that changes is the --model-set string. Cost uses run 2's measured speed on the Vast 5090 (13.6K tok/s at
8 loops, $0.696/hr) scaled by estimated compute per token. That scaling is a guess: dense models usually run more
efficiently per FLOP, so these speeds are probably conservative. Benchmark 100 steps before committing money.
"""
import argparse

from anthos.configs import get_training_config
from anthos.main import Anthos

BASE = "dim=768,n_heads=12,n_kv_heads=4,expert_dim=384,lora_rank=8"
RUN2_TOK_S = 13_600
DOLLARS_PER_HOUR = 0.696
FULL_TOKENS = 983_040_000          # 30,000 steps x 32,768
SHORT_TOKENS = 245_760_000         # 7,500 steps

CONTROLS = {
    # name: (model-set additions, the question it answers)
    "run2":          ("", "reference (already training)"),
    "act_remainder": (",act_mode=remainder", "does the missing remainder term in ACT matter?"),
    "act_last":      (",act_mode=last", "is the halting mixture useful vs simply using the last loop?"),
    "no_thoughts":   (",n_thought_tokens=0", "do the thought tokens + memory bank earn their cost?"),
    "dense_ffn":     (",ffn_type=dense", "MoE vs a parameter-matched dense FFN (9,216 wide), loops and thoughts kept"),
    "dense_ffn_cm":  (",ffn_type=dense,dense_ffn_dim=4608", "MoE vs a compute-matched dense FFN (4,608 wide, half the FFN parameters)"),
    "plain_dense":   (",ffn_type=dense,dense_ffn_dim=1024,max_loop_iters=1,n_thought_tokens=0,prelude_layers=6,coda_layers=5",
                      "a plain 12-layer dense transformer: no loops, thoughts or MoE (about 49M functional non-embedding params)"),
}


def build(extra: str):
    cfg, _ = get_training_config("proof")
    cfg.vocab_size = 100352
    for kv in (BASE + extra).split(","):
        k, v = kv.split("=")
        setattr(cfg, k, type(getattr(cfg, k))(v))
    return cfg


def counts(model):
    total = sum(p.numel() for p in model.parameters())
    emb = model.embed.weight.numel()
    dead = 0
    if model.recurrent.n_thought == 0:                       # modules that never run without thoughts
        for name in ("memory_bank", "thought_injection", "prerouter_head"):
            mod = getattr(model.recurrent, name)
            dead += sum(p.numel() for p in mod.parameters()) if mod is not None else 0
        dead += model.thought_pool.embeddings.numel() if hasattr(model, "thought_pool") else 0
    if not model.recurrent.block.use_moe:
        dead += sum(p.numel() for p in model.recurrent.prerouter_head.parameters()) if model.recurrent.n_thought > 0 else 0
    return total, total - emb, dead


def compute_equiv(cfg, loops: int) -> float:
    """Rough multiply-adds per token: dense blocks + looped block x loops + output head."""
    d, kvd = cfg.dim, cfg.dim // cfg.n_heads * cfg.n_kv_heads
    attn = 2 * d * d + 2 * d * kvd
    dense = attn + 3 * d * (d * 4 // 3)
    if cfg.ffn_type == "moe":
        ffn = cfg.n_experts_per_tok * 3 * d * cfg.expert_dim + cfg.n_shared_experts * 3 * d * cfg.expert_dim * cfg.n_experts_per_tok
    else:
        ffn = 3 * d * (cfg.dense_ffn_dim or cfg.n_experts * cfg.expert_dim + cfg.n_shared_experts * cfg.expert_dim * cfg.n_experts_per_tok)
    extra = 3 * d * d                                           # sequence injection
    if cfg.n_thought_tokens > 0:
        extra += 10 * d * d + 3 * d * d                         # memory bank + thought injection
    rec = attn + ffn + extra
    return (cfg.prelude_layers + cfg.coda_layers) * dense + loops * rec + cfg.vocab_size * d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd", default=None, help="print the launch command for this control")
    ap.add_argument("--ms", default=None, help="print only the --model-set string for this control")
    ap.add_argument("--loops", default=None, help="print the loop count to evaluate this control at")
    a = ap.parse_args()
    if a.ms:
        print(BASE + CONTROLS[a.ms][0]); return
    if a.loops:
        print(1 if "max_loop_iters=1" in CONTROLS[a.loops][0] else 8); return
    if a.cmd:
        extra, _ = CONTROLS[a.cmd]
        ms = BASE + extra
        print(f'python train.py --tier proof --pretrain-data data/pretrain_full --vocab-size 100352 --steps 30000 --seed 0 '
              f'--run-name anthos-ctl-{a.cmd} --thought-prefix-only --no-rep-penalty --log-every 100 --save-every 2500 '
              f'--batch-size 8 --grad-accum 8 --model-set "{ms}"')
        return
    ref = compute_equiv(build(""), 8)
    print(f"{'control':14s}{'params':>13s}{'non-emb':>12s}{'dead':>11s}{'compute':>9s}{'est tok/s':>11s}"
          f"{'$ full':>8s}{'$ short':>8s}  question")
    for name, (extra, q) in CONTROLS.items():
        cfg = build(extra)
        loops = 8 if cfg.max_loop_iters >= 8 else cfg.max_loop_iters
        total, nonemb, dead = counts(Anthos(cfg))
        rel = compute_equiv(cfg, loops) / ref
        tok_s = RUN2_TOK_S / rel
        full = FULL_TOKENS / tok_s / 3600 * DOLLARS_PER_HOUR
        short = SHORT_TOKENS / tok_s / 3600 * DOLLARS_PER_HOUR
        print(f"{name:14s}{total:13,d}{nonemb:12,d}{dead:11,d}{rel:8.2f}x{tok_s:11,.0f}{full:8.1f}{short:8.1f}  {q}")


if __name__ == "__main__":
    main()
