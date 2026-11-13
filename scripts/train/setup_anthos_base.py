#!/usr/bin/env python3
"""
setup_anthos_base.py — One-time setup: build Anthos tokenizer and base model.

Run this ONCE before any training stage. It:
  1. Downloads Qwen tokenizer and adds all Anthos special tokens
  2. Loads Qwen base weights + resizes embeddings for new tokens
  3. Saves embeddings checkpoint (safety backup)
  4. Optionally transplants vision encoder from Qwen2.5-VL
  5. Runs smoke test to confirm architecture is working
  6. Saves the base checkpoint to ./checkpoints/anthos-base/

Usage:
  python scripts/train/setup_anthos_base.py

  # With vision encoder transplant
  python scripts/train/setup_anthos_base.py --vision

  # Use a different base model
  python scripts/train/setup_anthos_base.py --base-model Qwen/Qwen2.5-7B-Instruct

Requirements:
  pip install transformers torch accelerate flash-attn
"""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from anthos_qwen.model import AnthosCore, build_anthos
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Anthos special tokens — full set from the build plan
# ---------------------------------------------------------------------------
ANTHOS_SPECIAL_TOKENS = [
    # Core structure
    "<SYS>", "</SYS>",
    "<USR>", "</USR>",
    "<AST>", "</AST>",
    "<END>",
    # Bifurcation / thought stream
    "<THT>", "</THT>",
    # Vision
    "<IMG>", "</IMG>",
    "<VIS>", "</VIS>",
    # Speculative decoding alignment
    "<DRAFT>", "</DRAFT>",
]


def setup_tokenizer(base_model_id: str, out_dir: Path) -> AutoTokenizer:
    print(f"[1] Loading base tokenizer from {base_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    print(f"[1] Adding {len(ANTHOS_SPECIAL_TOKENS)} special tokens...")
    tokenizer.add_special_tokens({
        "additional_special_tokens": ANTHOS_SPECIAL_TOKENS
    })

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(out_dir))
    print(f"[1] Tokenizer saved to {out_dir}")
    print(f"    Vocabulary size: {len(tokenizer):,}")
    return tokenizer


def setup_base_model(base_model_id: str, tokenizer, out_dir: Path) -> AnthosCore:
    print(f"\n[2] Loading base model {base_model_id}...")
    try:
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
    except Exception:
        print("  flash_attention_2 not available — falling back to eager attention")
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    print(f"[2] Resizing embeddings for {len(tokenizer):,} tokens...")
    base.resize_token_embeddings(len(tokenizer))

    model = AnthosCore(base, tokenizer)

    # Save embedding checkpoint immediately — safety backup
    emb_path = out_dir.parent / "embeddings_base.pt"
    torch.save(model.get_input_embeddings().weight.data.cpu(), emb_path)
    print(f"[2] Embeddings saved to {emb_path}")

    return model


def transplant_vision(model: AnthosCore, out_dir: Path):
    """Extract vision encoder from Qwen2.5-VL and wire into Anthos."""
    print("\n[3] Transplanting vision encoder from Qwen/Qwen2.5-VL-7B-Instruct...")
    try:
        from transformers import Qwen2VLForConditionalGeneration
        qwen_vl = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        vision_encoder = qwen_vl.visual
        vision_projection = qwen_vl.visual.merger

        model.attach_vision(vision_encoder, vision_projection)

        # Freeze encoder — only projection trains
        for p in model.vision_encoder.parameters():
            p.requires_grad = False
        for p in model.vision_projection.parameters():
            p.requires_grad = True

        # Save vision components
        torch.save({
            "vision_encoder": vision_encoder.state_dict(),
            "vision_projection": vision_projection.state_dict(),
        }, out_dir / "vision_components.pt")

        # Free the full VL model
        del qwen_vl
        torch.cuda.empty_cache()

        trainable_vis = sum(p.numel() for p in model.vision_projection.parameters())
        print(f"[3] Vision encoder wired — {trainable_vis/1e6:.1f}M trainable vision params")
    except Exception as e:
        print(f"[3] Vision transplant failed (non-fatal): {e}")
        print("    Continuing without vision — add --vision flag after installing Qwen2.5-VL")


def smoke_test(model, tokenizer) -> bool:
    """Quick sanity check — returns True if all tests pass."""
    print("\n[4] Running smoke test...")
    device = next(model.parameters()).device

    test_cases = [
        ("<SYS>You are Anthos.<END><USR>Hello<END><AST>", "basic text"),
        ("<SYS>You are Anthos.<END><USR>Analyze this<END><THT>", "thought token"),
        ("<SYS>You are Anthos.<END><USR>" + "x " * 50 + "<END><AST>", "long input"),
    ]

    all_pass = True
    for text, name in test_cases:
        try:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inputs)
            shape = out.logits.shape
            print(f"    {name}: PASSED — logits {tuple(shape)}")
        except Exception as e:
            print(f"    {name}: FAILED — {e}")
            all_pass = False

    # Bifurcation difference check
    with_tht = tokenizer("<USR>What is 2+2<END><THT>", return_tensors="pt").to(device)
    without_tht = tokenizer("<USR>What is 2+2<END><AST>", return_tensors="pt").to(device)
    with torch.no_grad():
        out_tht = model(**with_tht).logits
        out_ast = model(**without_tht).logits

    min_len = min(out_tht.shape[1], out_ast.shape[1])
    diff = (out_tht[0, :min_len] - out_ast[0, :min_len]).abs().mean().item()
    status = "OK" if diff > 0.001 else "WARNING — may not be active yet"
    print(f"    bifurcation difference: {diff:.4f} ({status})")

    result = "PASSED" if all_pass else "FAILED"
    print(f"[4] Smoke test {result}")
    return all_pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-35B-A3B-Base")
    ap.add_argument("--out-dir", default="checkpoints/anthos-base")
    ap.add_argument("--tokenizer-dir", default="anthos-tokenizer")
    ap.add_argument("--vision", action="store_true", help="Transplant vision encoder from Qwen2.5-VL")
    ap.add_argument("--hf-token", default=None)
    a = ap.parse_args()

    import os
    if a.hf_token:
        os.environ["HF_TOKEN"] = a.hf_token

    out_dir = Path(a.out_dir)
    tok_dir = Path(a.tokenizer_dir)

    print("=" * 60)
    print("  Anthos Base Setup")
    print("=" * 60)

    tokenizer = setup_tokenizer(a.base_model, tok_dir)
    model = setup_base_model(a.base_model, tokenizer, out_dir)

    if a.vision:
        transplant_vision(model, out_dir)

    if not smoke_test(model, tokenizer):
        sys.exit("Smoke test failed — do not proceed to training.")

    print(f"\n[5] Saving base checkpoint to {out_dir}...")
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    total = sum(p.numel() for p in model.parameters())
    print(f"\nDone. AnthosCore base ready — {total/1e9:.2f}B params")
    print(f"  Checkpoint: {out_dir}")
    print(f"  Tokenizer:  {tok_dir}")
    print(f"\nNext step:")
    print(f"  python scripts/train/sft_anthos.py --stage smoke --data data/sft_anthos.jsonl")


if __name__ == "__main__":
    main()
