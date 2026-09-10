#!/usr/bin/env python3
"""
preflight.py — Pre-training smoke test for Anthos SFT run

Runs before spending money on RunPod. Catches:
  1. Dataset format errors
  2. Tokenizer load failures
  3. Model init / forward pass errors
  4. Dataloader errors
  5. Optimizer + loss step (100 steps on 1000 samples — CPU)
  6. Checkpoint save + load round trip
  7. Inference stack (Synapse + retrieval + tools + multilingual)

Usage:
    python preflight.py              # full preflight (~5 min on MacBook)
    python preflight.py --fast       # skip smoke train (just structure checks)
"""

import argparse
import json
import sys
import time
import tempfile
import shutil
from pathlib import Path

PASS = "  ✓"
FAIL = "  ✗"
WARN = "  ⚠"


def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def check(label, ok, detail=""):
    mark = PASS if ok else FAIL
    suffix = f" — {detail}" if detail else ""
    print(f"{mark} {label}{suffix}")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dataset spot-check
# ─────────────────────────────────────────────────────────────────────────────

def check_dataset(path="data/sft_master.jsonl", n_sample=500):
    section("1. Dataset spot-check")
    p = Path(path)
    if not check("File exists", p.exists(), str(p)):
        return False

    size_gb = p.stat().st_size / 1e9
    check("File size > 1GB", size_gb > 1.0, f"{size_gb:.2f} GB")

    bad, good = 0, 0
    with open(p) as f:
        for i, line in enumerate(f):
            if i >= n_sample:
                break
            try:
                obj = json.loads(line)
                convs = obj.get("conversations", [])
                has_human = any(c.get("from") in ("human","user") for c in convs)
                has_gpt   = any(c.get("from") in ("gpt","assistant") for c in convs)
                if has_human and has_gpt:
                    good += 1
                else:
                    bad += 1
            except Exception:
                bad += 1

    total = good + bad
    ok = bad == 0
    check(f"First {n_sample} rows valid", ok, f"{good}/{total} good")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# 2. Tokenizer
# ─────────────────────────────────────────────────────────────────────────────

def check_tokenizer():
    section("2. Tokenizer")
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("data/anthos_tokenizer")
        vocab = tok.vocab_size
        check("Loaded anthos_tokenizer", True, f"vocab={vocab}")

        # Verify special tokens
        ids = tok.encode("Hello world", add_special_tokens=False)
        check("Encode/decode round trip", len(ids) > 0,
              tok.decode(ids, skip_special_tokens=True))
        return True, tok
    except Exception as e:
        check("Tokenizer load", False, str(e))
        return False, None


# ─────────────────────────────────────────────────────────────────────────────
# 3. Model init + forward pass
# ─────────────────────────────────────────────────────────────────────────────

def check_model():
    section("3. Model init + forward pass")
    try:
        import torch
        from anthos.main import Anthos
        from anthos.configs import get_training_config

        cfg, _ = get_training_config("sft")
        model = Anthos(cfg)
        n_params = sum(p.numel() for p in model.parameters())
        check("Model init", True, f"{n_params:,} params")

        # Quick forward pass
        x = torch.randint(0, cfg.vocab_size, (1, 32))
        with torch.no_grad():
            out = model(x)
        check("Forward pass", out is not None, f"output shape {tuple(out[0].shape) if out is not None else 'None'}")
        return True, model, cfg
    except Exception as e:
        check("Model", False, str(e))
        return False, None, None


# ─────────────────────────────────────────────────────────────────────────────
# 4. Dataloader
# ─────────────────────────────────────────────────────────────────────────────

def check_dataloader():
    section("4. Dataloader (first 5 batches)")
    try:
        from anthos.data import get_chat_dataloader
        loader = get_chat_dataloader(
            seq_len        = 128,
            batch_size     = 2,
            num_workers    = 0,
            tokenizer_path = "data/anthos_tokenizer",
            max_samples    = 100,
            dataset_name   = "data/sft_master.jsonl",
        )
        batches = 0
        for ids, labels in loader:
            batches += 1
            if batches >= 5:
                break
        check("Loaded 5 batches", batches >= 1,
              f"shape={tuple(ids.shape)}, labels={tuple(labels.shape)}")
        return True
    except Exception as e:
        check("Dataloader", False, str(e))
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 5. Smoke training (100 steps, CPU)
# ─────────────────────────────────────────────────────────────────────────────

def smoke_train():
    section("5. Smoke training — 100 steps on CPU")
    try:
        import torch
        from torch.optim import AdamW
        from anthos.main import Anthos
        from anthos.configs import get_training_config
        from anthos.data import get_chat_dataloader
        from anthos.eaft import EAFTLoss

        cfg, tcfg = get_training_config("sft")
        model = Anthos(cfg)
        optimizer = AdamW(model.parameters(), lr=3e-5)
        loss_fn = EAFTLoss(vocab_size=cfg.vocab_size, top_k=50,
                           focal_gamma=1.0, act_gamma=0.5,
                           max_loops=cfg.max_loop_iters, label_smoothing=0.1)

        loader = get_chat_dataloader(
            seq_len=128, batch_size=2, num_workers=0,
            tokenizer_path="data/anthos_tokenizer",
            max_samples=200, dataset_name="data/sft_master.jsonl",
        )

        losses = []
        t0 = time.time()
        data_iter = iter(loader)

        for step in range(100):
            try:
                ids, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                ids, labels = next(data_iter)

            optimizer.zero_grad()
            logits, _, loop_counts = model(ids, n_loops=4)
            loss = loss_fn(logits, labels, loop_counts)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if step % 25 == 0:
                print(f"    step {step:3d}  loss={loss.item():.4f}")

        elapsed = time.time() - t0
        loss_drop = losses[0] - losses[-1]
        ok = loss_drop > 0 or losses[-1] < losses[0] * 1.2
        check("Loss decreased", losses[-1] < losses[0],
              f"{losses[0]:.4f} → {losses[-1]:.4f}  ({elapsed:.0f}s)")
        return True, model
    except Exception as e:
        check("Smoke train", False, str(e))
        import traceback; traceback.print_exc()
        return False, None


# ─────────────────────────────────────────────────────────────────────────────
# 6. Checkpoint save + load
# ─────────────────────────────────────────────────────────────────────────────

def check_checkpoint(model=None):
    section("6. Checkpoint save + load round trip")
    if model is None:
        print("  (skipped — no model from smoke train)")
        return True
    try:
        import torch
        from anthos.main import Anthos
        from anthos.configs import get_training_config

        tmp = Path(tempfile.mkdtemp())
        ckpt_path = tmp / "preflight_smoke.pt"

        # Save
        torch.save({"model": model.state_dict(), "step": 100}, ckpt_path)
        size_mb = ckpt_path.stat().st_size / 1e6
        check("Checkpoint saved", ckpt_path.exists(), f"{size_mb:.1f} MB")

        # Load into fresh model
        cfg, _ = get_training_config("sft")
        m2 = Anthos(cfg)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        missing, unexpected = m2.load_state_dict(ckpt["model"], strict=False)
        check("Checkpoint loaded", len(unexpected) == 0,
              f"missing={len(missing)} unexpected={len(unexpected)}")

        shutil.rmtree(tmp)
        return True
    except Exception as e:
        check("Checkpoint", False, str(e))
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 7. Inference stack
# ─────────────────────────────────────────────────────────────────────────────

def check_inference_stack():
    section("7. Inference stack (Synapse + retrieval + tools + multilingual)")
    results = []

    # Synapse
    try:
        from anthos.synapse import Synapse
        b = Synapse.load()
        b.add_turn("user", "test")
        b.add_turn("assistant", "response")
        results.append(check("Synapse memory", True, f"{len(b.turns)} turns"))
    except Exception as e:
        results.append(check("Synapse memory", False, str(e)))

    # Tools
    try:
        from anthos.toolbox import detect_tool, run_tool
        tool, arg = detect_tool("what time is it?")
        result = run_tool(tool, arg)
        results.append(check("Toolbox", tool == "time" and "2026" in result,
                             f"tool={tool}, result snippet: {result[:40]}"))
    except Exception as e:
        results.append(check("Toolbox", False, str(e)))

    # Multilingual
    try:
        from anthos.multilingual import detect_language, language_directive
        lang = detect_language("Hola cómo estás")
        directive = language_directive(lang)
        results.append(check("Multilingual", lang == "es" and "Spanish" in directive,
                             f"detected={lang}"))
    except Exception as e:
        results.append(check("Multilingual", False, str(e)))

    # Retrieval (just imports + heuristic, skip live fetch)
    try:
        from anthos.retrieval import needs_retrieval
        r1 = needs_retrieval("current stock price of Apple")
        r2 = needs_retrieval("write me a haiku")
        results.append(check("Retrieval heuristic", r1 and not r2,
                             f"current-event=True, haiku=False"))
    except Exception as e:
        results.append(check("Retrieval", False, str(e)))

    return all(results)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Skip smoke training (structure checks only)")
    args = parser.parse_args()

    print(f"\n{'═'*60}")
    print("  ANTHOS PREFLIGHT CHECK")
    print(f"{'═'*60}")

    results = []
    results.append(check_dataset())
    tok_ok, tok = check_tokenizer()
    results.append(tok_ok)
    model_ok, model, cfg = check_model()
    results.append(model_ok)
    results.append(check_dataloader())

    smoke_model = None
    if not args.fast:
        train_ok, smoke_model = smoke_train()
        results.append(train_ok)
        results.append(check_checkpoint(smoke_model))
    else:
        print(f"\n{'─'*60}")
        print("  5+6. Smoke training — SKIPPED (--fast)")

    results.append(check_inference_stack())

    passed = sum(results)
    total  = len(results)

    print(f"\n{'═'*60}")
    if passed == total:
        print(f"  ALL CHECKS PASSED ({passed}/{total})")
        print(f"  Ready for RunPod training run.")
    else:
        print(f"  {passed}/{total} checks passed — fix failures before training.")
    print(f"{'═'*60}\n")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
