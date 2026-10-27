"""
eval_leak_gap.py — how much does teacher-forced loss flatter the model?

For each held-out chat example it computes the response loss two ways:

  teacher-forced : one full forward pass over prompt + response (what train.py reports)
  stepwise       : prefill the prompt, then decode the response one token at a time with the
                   KV cache, scoring the true next token each step (what generation can do)

If a later token can influence an earlier position, the teacher-forced pass can use it and the
stepwise pass cannot. The gap (stepwise - teacher-forced) is therefore a direct measure of leakage.
A causal model should show a gap of ~0; a leaky one shows stepwise loss well above teacher-forced.

Usage (held-out data must be disjoint from training data):
    head -n -200 data/teacher_conversations.jsonl > data/train.jsonl
    tail -n  200 data/teacher_conversations.jsonl > data/val.jsonl

    python3 eval_leak_gap.py --tier convo_smoke --checkpoint checkpoints/<run>/step_010000.pt \
        --data data/val.jsonl                       # legacy checkpoint
    python3 eval_leak_gap.py --tier convo_smoke --checkpoint checkpoints/<run>/step_010000.pt \
        --data data/val.jsonl --thought-prefix-only # causal-mode checkpoint
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from anthos.main import Anthos


@torch.no_grad()
def example_losses(model: Anthos, ids: torch.Tensor, prompt_len: int, n_loops: int, causal: bool):
    """Return (teacher_forced_nll_sum, stepwise_nll_sum, n_response_tokens) for one example."""
    device = next(model.parameters()).device
    ids = ids.to(device)
    L = ids.shape[0]
    p = max(1, min(prompt_len, L - 1))
    kw = {"prefix_len": p} if causal else {}
    targets = ids[p:L]

    # teacher-forced: logits at position j-1 predict token j
    logits = model(ids[None], n_loops=n_loops, **kw)[0]
    tf = F.cross_entropy(logits[p - 1:L - 1].float(), targets, reduction="sum").item()

    # stepwise: prefill the prompt, then feed the true response tokens one at a time
    cache = {}
    step_logits = [model(ids[None, :p], n_loops=n_loops, kv_cache=cache, **kw)[0, -1]]
    for j in range(p, L - 1):
        out = model(ids[None, j:j + 1], n_loops=n_loops, kv_cache=cache, start_pos=j, **kw)
        step_logits.append(out[0, -1])
    sw = F.cross_entropy(torch.stack(step_logits).float(), targets, reduction="sum").item()
    return tf, sw, int(targets.numel())


def leak_gap(model: Anthos, examples, n_loops: int, causal: bool) -> dict:
    """examples: iterable of (input_ids 1-D tensor, prompt_len int)."""
    model.eval()
    tf_sum = sw_sum = 0.0
    n_tok = n_ex = 0
    for ids, prompt_len in examples:
        if ids.shape[0] < 3:
            continue
        tf, sw, n = example_losses(model, ids, prompt_len, n_loops, causal)
        tf_sum += tf
        sw_sum += sw
        n_tok += n
        n_ex += 1
    if n_tok == 0:
        raise ValueError("no usable examples")
    tf_loss, sw_loss = tf_sum / n_tok, sw_sum / n_tok
    return {"examples": n_ex, "tokens": n_tok, "teacher_forced_loss": tf_loss,
            "stepwise_loss": sw_loss, "leak_gap": sw_loss - tf_loss}


def load_examples(path: str, tokenizer_path: str, seq_len: int, max_examples: int):
    """Read a ShareGPT jsonl through the same loader train.py uses (one pass, no repeat)."""
    from anthos.data import ChatInstructDataset
    ds = ChatInstructDataset(tokenizer_path=tokenizer_path, seq_len=seq_len, dataset_name=path,
                             return_prefix_len=True)
    out = []
    for ids, _labels, prefix_len in ds:
        out.append((ids, int(prefix_len)))
        if len(out) >= max_examples:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="convo_smoke", help="config tier the checkpoint was trained with")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True, help="held-out ShareGPT jsonl")
    ap.add_argument("--tokenizer", default="data/anthos_tokenizer")
    ap.add_argument("--thought-prefix-only", action="store_true", dest="causal")
    ap.add_argument("--n-loops", type=int, default=None, help="default: the tier's phase-2 loop count")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--max-examples", type=int, default=200)
    ap.add_argument("--json", action="store_true", help="print the result as one JSON line")
    args = ap.parse_args()

    from anthos.configs import get_training_config
    model_cfg, train_cfg = get_training_config(args.tier)
    if args.causal:
        model_cfg.thought_prefix_only = True
        model_cfg.moe_fix_weight_order = True
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = Anthos(model_cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        print(f"  note: {len(missing)} missing / {len(unexpected)} unexpected tensors when loading", file=sys.stderr)
    n_loops = args.n_loops or getattr(train_cfg, "phase2_loops", model_cfg.max_loop_iters)

    examples = load_examples(args.data, args.tokenizer, args.seq_len, args.max_examples)
    res = leak_gap(model, examples, n_loops, args.causal)
    res.update(mode="causal" if args.causal else "legacy", n_loops=n_loops, checkpoint=args.checkpoint)
    if args.json:
        print(json.dumps(res))
    else:
        print(f"mode={res['mode']} loops={n_loops} examples={res['examples']} tokens={res['tokens']}")
        print(f"  teacher-forced loss : {res['teacher_forced_loss']:.4f}")
        print(f"  stepwise loss       : {res['stepwise_loss']:.4f}   <- what generation can actually achieve")
        print(f"  leak gap            : {res['leak_gap']:+.4f}   (~0 means no leakage)")


if __name__ == "__main__":
    main()
