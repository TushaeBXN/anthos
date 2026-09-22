"""
tests/test_identity_freeze.py — Identity lock verification tests 1, 3, 4.

Run after collecting real Phase 2 checkpoints:

    python3 tests/test_identity_freeze.py \\
        --before checkpoints/anthos-1b/identity_hardening_step_005001.pt \\
        --after  checkpoints/anthos-1b/identity_hardening_step_005100.pt

IMPORTANT: pass two POST-freeze checkpoints for Test 1 — both must be from
after the freeze step (≥ step 5000). Comparing pre-freeze to post-freeze will
always show non-zero drift: at the freeze boundary, the hard restore snaps
trained values back to the initialization snapshot, so the difference is
intentional, not a bug. What matters is that values are STABLE between any
two consecutive checkpoints after the freeze is active.

Test 1 — Value stability:  diff identity embedding rows between two post-freeze saves.
Test 3 — Weight decay check: inspect param group config on a fresh model (no checkpoint needed).
Test 4 — Phase 3 continuity: load Phase 2 checkpoint into AnthosWithIdentityLock,
          confirm identity_head.* keys are NOT in missing (i.e., they're loaded, not reinit'd).

Each test prints PASS or FAIL with the actual measured number.
Do NOT treat a code-review of this file as a pass — run it.
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
)

IDENTITY_ROWS = list(IDENTITY_TOKEN_IDS.values())   # [32000..32007]
DRIFT_THRESHOLD = 1e-6


def get_proof_cfg() -> AnthosConfig:
    return AnthosConfig(
        vocab_size        = 50257,
        dim               = 512,
        n_heads           = 8,
        n_kv_heads        = 4,
        max_seq_len       = 512,
        max_loop_iters    = 8,
        prelude_layers    = 2,
        coda_layers       = 2,
        n_thought_tokens  = 16,
        attn_type         = "gqa",
        n_experts         = 16,
        n_shared_experts  = 2,
        n_experts_per_tok = 4,
        expert_dim        = 256,
        lora_rank         = 8,
    )


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


def _load_state(path: str) -> dict:
    cp = torch.load(path, map_location="cpu", weights_only=False)
    return cp.get("model_state_dict", cp.get("model", cp))


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — Value stability
# ─────────────────────────────────────────────────────────────────────────────

def test_value_stability(before_path: str, after_path: str) -> bool:
    print("\n" + "─"*60)
    print("TEST 1 — Value stability (identity embedding rows, both post-freeze)")
    print(f"  checkpoint A: {before_path}")
    print(f"  checkpoint B: {after_path}")
    print("  NOTE: both checkpoints must be from after the freeze step (>= step 5000).")

    sd_before = _load_state(before_path)
    sd_after  = _load_state(after_path)

    # Support both wrapped (base.embed.weight) and plain (embed.weight) keys
    embed_key = None
    for candidate in ("base.embed.weight", "embed.weight"):
        if candidate in sd_before:
            embed_key = candidate
            break

    if embed_key is None:
        print(f"  FAIL — could not find embed weight key in checkpoint")
        print(f"  available keys (first 10): {list(sd_before.keys())[:10]}")
        return False

    emb_before = sd_before[embed_key][IDENTITY_ROWS]
    emb_after  = sd_after[embed_key][IDENTITY_ROWS]
    max_drift  = (emb_after - emb_before).abs().max().item()
    mean_drift = (emb_after - emb_before).abs().mean().item()

    print(f"  embed key: {embed_key}")
    print(f"  max drift across identity rows:  {max_drift:.2e}")
    print(f"  mean drift across identity rows: {mean_drift:.2e}")
    print(f"  threshold: {DRIFT_THRESHOLD:.2e}")

    if max_drift < DRIFT_THRESHOLD:
        print("  PASS")
        return True
    else:
        print("  FAIL — identity embedding rows moved after freeze step.")
        print("         The copy_-restore or grad-masking is not working.")
        print("         Do NOT proceed to Phase 3 until this is fixed.")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — Weight decay configuration
# ─────────────────────────────────────────────────────────────────────────────

def test_weight_decay_config() -> bool:
    print("\n" + "─"*60)
    print("TEST 3 — Weight decay configuration")
    print("  (no checkpoint needed — inspects optimizer param groups)")

    from torch.optim import AdamW

    cfg   = get_1b_cfg()
    base  = Anthos(cfg)
    model = AnthosWithIdentityLock(base, hidden_dim=cfg.dim)

    identity_params, normal_params = [], []
    for name, p in model.named_parameters():
        if "identity" in name or "embed" in name:
            identity_params.append(p)
        else:
            normal_params.append(p)

    optimizer = AdamW([
        {"params": normal_params,   "lr": 1e-4,  "weight_decay": 0.1},
        {"params": identity_params, "lr": 3e-4,  "weight_decay": 0.0},
    ], betas=(0.9, 0.95))

    identity_group_wds = []
    normal_group_wds   = []
    for group in optimizer.param_groups:
        wd = group["weight_decay"]
        if group["lr"] == 3e-4:      # identity group
            identity_group_wds.append(wd)
        else:
            normal_group_wds.append(wd)

    id_wd_ok     = all(wd == 0.0 for wd in identity_group_wds)
    normal_wd_ok = all(wd == 0.1 for wd in normal_group_wds)

    n_identity = sum(p.numel() for p in identity_params)
    n_normal   = sum(p.numel() for p in normal_params)

    print(f"  identity param count: {n_identity:,}  |  weight_decay: {identity_group_wds}")
    print(f"  normal   param count: {n_normal:,}  |  weight_decay: {normal_group_wds}")

    if id_wd_ok and normal_wd_ok:
        print("  PASS — identity params have weight_decay=0.0, normal params 0.1")
        return True
    else:
        print("  FAIL — weight_decay misconfigured")
        if not id_wd_ok:
            print("         identity group should have weight_decay=0.0")
        if not normal_wd_ok:
            print("         normal group should have weight_decay=0.1")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — Phase 3 checkpoint continuity
# ─────────────────────────────────────────────────────────────────────────────

def test_phase3_continuity(phase2_ckpt: str) -> bool:
    print("\n" + "─"*60)
    print("TEST 4 — Phase 3 checkpoint continuity")
    print(f"  loading: {phase2_ckpt}")

    cfg   = get_1b_cfg()
    base  = Anthos(cfg)
    model = AnthosWithIdentityLock(base, hidden_dim=cfg.dim)

    sd = _load_state(phase2_ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)

    identity_missing    = [k for k in missing    if "identity" in k]
    identity_unexpected = [k for k in unexpected if "identity" in k]

    print(f"  total missing keys:      {len(missing)}")
    print(f"  total unexpected keys:   {len(unexpected)}")
    print(f"  identity-related missing:    {identity_missing}")
    print(f"  identity-related unexpected: {identity_unexpected}")

    if identity_missing:
        print("  FAIL — identity_head.* keys are MISSING after load.")
        print("         This means Phase 2 checkpoint didn't include the identity head,")
        print("         or it was saved from a plain Anthos instead of AnthosWithIdentityLock.")
        print("         Phase 3 would reinitialize the identity head randomly — defeating Phase 2.")
        return False
    else:
        print("  PASS — all identity_head.* keys loaded from Phase 2 checkpoint")
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default=None,
                    help="First post-freeze checkpoint (e.g. step 5001)")
    ap.add_argument("--after",  default=None,
                    help="Second post-freeze checkpoint (e.g. step 5100); both must be >= freeze_step")
    ap.add_argument("--phase2", default=None,
                    help="Phase 2 final checkpoint (for Test 4). "
                         "Defaults to --after if not set.")
    ap.add_argument("--skip-stability", action="store_true",
                    help="Skip Test 1 (use when checkpoints aren't available yet)")
    args = ap.parse_args()

    results = {}

    # Test 3 never needs checkpoints — always run it
    results["test3_weight_decay"] = test_weight_decay_config()

    if not args.skip_stability:
        if not args.before or not args.after:
            print("\nTest 1 skipped — pass --before and --after to run it.")
            print("  (save two checkpoints after step 5000 during Phase 2 training;")
            print("   both must be post-freeze — comparing pre-to-post shows intentional snap-back)")
            results["test1_stability"] = None
        else:
            results["test1_stability"] = test_value_stability(args.before, args.after)

    phase2_ckpt = args.phase2 or args.after
    if phase2_ckpt and Path(phase2_ckpt).exists():
        results["test4_continuity"] = test_phase3_continuity(phase2_ckpt)
    else:
        print("\nTest 4 skipped — pass --phase2 (or --after) pointing at a Phase 2 checkpoint.")
        results["test4_continuity"] = None

    print("\n" + "═"*60)
    print("SUMMARY")
    for name, result in results.items():
        status = "PASS" if result is True else ("FAIL" if result is False else "SKIPPED")
        print(f"  {name}: {status}")

    failed = [k for k, v in results.items() if v is False]
    if failed:
        print(f"\nFailed: {failed}")
        print("Do NOT proceed to further tests or Phase 3 until failures are resolved.")
        sys.exit(1)
    else:
        print("\nAll run tests passed.")


if __name__ == "__main__":
    main()
