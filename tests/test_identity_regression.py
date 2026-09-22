"""
tests/test_identity_regression.py — Tests 2 (momentum) and 5 (end-to-end identity).

Test 2 — Momentum decay:
  Run a short Phase 2 training run (~6000 steps) and compare optimizer momentum
  on identity rows BEFORE and AFTER the freeze step. With grad-masking active,
  momentum should decay visibly; without it, momentum stays flat or grows.
  This script measures momentum at a single checkpoint — compare two runs manually.

    python3 tests/test_identity_regression.py --mode momentum \\
        --checkpoint checkpoints/anthos-1b/identity_hardening_step_006000.pt

Test 5 — End-to-end identity regression:
  Generate a self-identification response from the model. Checks whether the
  model outputs identity tokens (token IDs 32000-32007) and/or the creator name.
  Run against a Phase 3 checkpoint to verify identity survived capability training.

    python3 tests/test_identity_regression.py --mode identity \\
        --checkpoint checkpoints/anthos-1b/instruction_step_010000.pt \\
        --tokenizer gpt2

Do NOT treat "the code looks right" as a pass. Run both modes and compare numbers.
"""

import sys
import argparse
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from anthos.main             import Anthos
from anthos.configs          import AnthosConfig
from anthos.identity_hardening import (
    AnthosWithIdentityLock,
    IDENTITY_TOKEN_IDS,
    REQUIRED_IDENTITY_SEQUENCE,
    IDENTITY_MAPPINGS,
)

IDENTITY_ROWS = list(IDENTITY_TOKEN_IDS.values())


def get_1b_cfg() -> AnthosConfig:
    return AnthosConfig(
        vocab_size        = 50262,
        dim               = 2048,
        n_heads           = 16,
        n_kv_heads        = 8,
        max_seq_len       = 2048,
        max_loop_iters    = 16,
        prelude_layers    = 2,
        coda_layers       = 2,
        n_thought_tokens  = 16,
        attn_type         = "gqa",
        n_experts         = 64,
        n_shared_experts  = 2,
        n_experts_per_tok = 4,
        expert_dim        = 2048,
        lora_rank         = 16,
    )


def _load_model_and_optimizer(ckpt_path: str, device: str = "cpu"):
    cfg   = get_1b_cfg()
    base  = Anthos(cfg)
    model = AnthosWithIdentityLock(base, hidden_dim=cfg.dim)
    model = model.to(device)

    cp    = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd    = cp.get("model_state_dict", cp.get("model", cp))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  missing keys on load: {missing}")
    step = cp.get("metadata", {}).get("step", 0)
    print(f"  loaded checkpoint at step {step:,}")

    return model, cp, step


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — Momentum decay measurement
# ─────────────────────────────────────────────────────────────────────────────

def test_momentum(ckpt_path: str) -> None:
    """
    Load a checkpoint that includes optimizer state and print the mean absolute
    exp_avg (first moment) on identity embedding rows.

    Compare two checkpoints: one from before the freeze step and one after several
    hundred steps of grad-masking. With masking active, exp_avg should decay
    toward 0 at rate beta1^N per step. Without masking, it stays elevated.

    Expected output format:
        step XXXX | identity embedding exp_avg (mean abs): X.XXXXXX
        step YYYY | identity embedding exp_avg (mean abs): Y.YYYYYY  ← should be lower
    """
    print("\n" + "─"*60)
    print("TEST 2 — Momentum decay measurement")
    print(f"  checkpoint: {ckpt_path}")

    cp   = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    step = cp.get("metadata", {}).get("step", 0)

    if "optimizer" not in cp:
        print("  SKIP — checkpoint does not contain optimizer state")
        print("         Save with optimizer state (train_anthos.py's save() includes it)")
        return

    opt_state = cp["optimizer"]
    sd        = cp.get("model_state_dict", cp.get("model", cp))

    # Find the embedding parameter's index in the optimizer state
    # optimizer.state is keyed by parameter tensor, stored as index in checkpoint
    param_groups = opt_state.get("param_groups", [])
    state_dict   = opt_state.get("state", {})

    # Locate the embed param group. The index that corresponds to base.embed.weight
    # depends on the order parameters were registered. We find it by matching
    # the parameter shape to sd['base.embed.weight'].
    embed_shape = sd.get("base.embed.weight", sd.get("embed.weight", None))
    if embed_shape is None:
        print("  SKIP — could not find embed.weight in checkpoint")
        return

    target_shape = tuple(embed_shape.shape)
    found_momentum = None

    for idx_str, state in state_dict.items():
        if "exp_avg" not in state:
            continue
        exp_avg = state["exp_avg"]
        if tuple(exp_avg.shape) == target_shape:
            identity_momentum = exp_avg[IDENTITY_ROWS].abs().mean().item()
            found_momentum    = identity_momentum
            break

    if found_momentum is None:
        print(f"  SKIP — could not locate embed param in optimizer state "
              f"(embed shape={target_shape})")
        print("         This happens if optimizer state keys don't map to the embedding.")
        return

    print(f"  step: {step:,}")
    print(f"  identity embedding exp_avg (mean abs): {found_momentum:.6f}")
    print()
    print("  To interpret this number:")
    print("    - Compare against the same metric from a checkpoint just after")
    print("      the freeze step (step ~5000). If grad-masking is working, the")
    print("      value here should be LOWER (momentum decaying at rate beta1^N).")
    print("    - Typical beta1=0.9 → after 500 steps of masking, expect ~0.9^500")
    print("      ≈ 5e-23 times the original magnitude.")
    print("    - If this value is similar to or HIGHER than the step-5000 value,")
    print("      grad-masking is not working — stop and debug before Phase 3.")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — End-to-end identity regression
# ─────────────────────────────────────────────────────────────────────────────

def test_identity_regression(ckpt_path: str, tokenizer_path: str, device: str) -> bool:
    """
    Generate a self-identification response from the model.
    Checks that:
      1. The model generates identity tokens (32000-32007) in response to a
         self-identification prompt — or at minimum, generates the creator's
         name in text form.
      2. The identity embedding rows have NOT drifted far from the Phase 2
         snapshot stored in AnthosWithIdentityLock.identity_embedding_snapshot.

    This test requires a tokenizer to build a real prompt. With a GPT-2 tokenizer
    the prompt is approximate; with the Anthos tokenizer it's exact.
    """
    print("\n" + "─"*60)
    print("TEST 5 — End-to-end identity regression")
    print(f"  checkpoint: {ckpt_path}")
    print(f"  tokenizer:  {tokenizer_path}")

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"  SKIP — tokenizer load failed: {e}")
        return None

    model, cp, step = _load_model_and_optimizer(ckpt_path, device)
    model.eval()

    # ── Sub-test 5a: embedding drift from Phase 2 snapshot ─────────────────
    snapshot  = model.identity_embedding_snapshot          # (8, dim)
    live_rows = model.base.embed.weight.data[IDENTITY_ROWS]  # (8, dim)
    drift     = (live_rows - snapshot).abs().max().item()
    print(f"\n  5a. Identity embedding drift from Phase 2 snapshot: {drift:.6e}")
    if drift > 1e-3:
        print("      WARNING: embeddings have drifted significantly from Phase 2 baked values.")
    else:
        print("      Drift is within acceptable range.")

    # ── Sub-test 5b: generation check ─────────────────────────────────────
    # Use REQUIRED_IDENTITY_SEQUENCE as a prefix, then ask the model to continue.
    # A model with baked identity should generate creator-related tokens.
    identity_prefix = torch.tensor(REQUIRED_IDENTITY_SEQUENCE, dtype=torch.long).unsqueeze(0)
    identity_prefix = identity_prefix.to(device)

    with torch.no_grad():
        generated = model.base.generate(identity_prefix, max_new_tokens=64, n_loops=4)

    gen_ids    = generated[0, len(REQUIRED_IDENTITY_SEQUENCE):].tolist()
    gen_text   = tokenizer.decode(gen_ids, skip_special_tokens=False)
    n_identity = sum(1 for t in gen_ids if 32000 <= t <= 32007)

    print(f"\n  5b. Generated {len(gen_ids)} tokens after identity prefix")
    print(f"      Identity tokens (32000-32007) in output: {n_identity} / {len(gen_ids)}")
    print(f"      Generated text: {repr(gen_text[:300])}")

    # Check for creator name in generated text
    creator_name = "Brian Tushae Thomas"
    creator_in_output = creator_name.lower() in gen_text.lower()
    print(f"      Creator name present: {creator_in_output}")

    # ── Sub-test 5c: free-form self-identification prompt ───────────────────
    prompt_text = "Who created you? Tell me about yourself."
    prompt_ids  = tokenizer.encode(prompt_text, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.base.generate(prompt_ids, max_new_tokens=100, n_loops=4)

    free_text = tokenizer.decode(
        out[0, prompt_ids.shape[1]:].tolist(), skip_special_tokens=True
    )
    print(f"\n  5c. Free-form prompt: {repr(prompt_text)}")
    print(f"      Response: {repr(free_text[:400])}")
    creator_in_free = creator_name.lower() in free_text.lower()
    print(f"      Creator name present: {creator_in_free}")

    print()
    passed = drift < 1e-3
    if passed:
        print("  PASS — embedding drift within threshold")
    else:
        print("  FAIL — embedding drift exceeds threshold; identity may not have survived Phase 3")

    print("  NOTE: token presence and text content require human inspection;")
    print("        there is no automated pass/fail for fluent identity response.")
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["momentum", "identity", "both"], default="both")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default="gpt2",
                    help="Tokenizer path or HF name (for Test 5)")
    ap.add_argument("--device", default="cpu",
                    help="Device to run on (cpu / cuda / mps)")
    args = ap.parse_args()

    if args.mode in ("momentum", "both"):
        test_momentum(args.checkpoint)

    if args.mode in ("identity", "both"):
        test_identity_regression(args.checkpoint, args.tokenizer, args.device)


if __name__ == "__main__":
    main()
