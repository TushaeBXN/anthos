"""
generate_teacher_labels.py — Run teacher model to generate soft labels for Anthos distillation

Step 1 of offline distillation:
  Run this ONCE with a large teacher model (Qwen3-14B, LLaMA-3.1-70B, etc.)
  to generate soft probability labels. Then train Anthos student on those labels.

Requires teacher model installed via HuggingFace or Unsloth.

Usage:
    # With a HuggingFace model (needs enough RAM/VRAM):
    python generate_teacher_labels.py \
        --teacher Qwen/Qwen3-7B \
        --dataset roneneldan/TinyStories \
        --n_samples 50000 \
        --top_k 64 \
        --out data/teacher_labels_qwen7b.jsonl

    # With a GGUF model via llama.cpp Python bindings:
    python generate_teacher_labels.py \
        --teacher_gguf models/qwen3-14b-q4_k_m.gguf \
        --dataset roneneldan/TinyStories \
        --n_samples 50000 \
        --out data/teacher_labels_qwen14b.jsonl

    # Then train Anthos student on the labels:
    python train.py --tier distill --teacher_labels data/teacher_labels_qwen7b.jsonl

Teacher recommendations by hardware:
    M1 Max 64GB:
        → Qwen3-14B Q4_K_M via Ollama/llama.cpp (~8GB) — best quality/speed ratio
        → Qwen3-7B Q4_K_M (~4GB) — faster, still good
        → LLaMA-3.1-8B Q4_K_M (~5GB) — alternative

    Current MacBook Pro (CPU only):
        → Qwen3-1.7B or Qwen3-0.6B — only viable option on CPU
        → Better to generate labels on a cloud GPU instance once

    Cloud (recommended for label generation):
        → Rent a GPU instance (Lambda, RunPod, Vast.ai) for $0.5-1/hr
        → Run Qwen3-32B or LLaMA-3.1-70B for highest quality labels
        → 50k samples takes ~2-4 hours on a single A100
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))


def load_hf_teacher(model_name: str, device: str):
    """Load a HuggingFace teacher model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading teacher: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else device,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  ✓ Teacher loaded: {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")
    return model, tok


def generate_labels(args):
    from anthos.distill import TeacherLabelGenerator

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    if args.teacher:
        teacher, tokenizer = load_hf_teacher(args.teacher, device)
    else:
        print("Error: --teacher required (HuggingFace model name)")
        sys.exit(1)

    gen = TeacherLabelGenerator(
        teacher_model=teacher,
        tokenizer=tokenizer,
        top_k=args.top_k,
        device=device,
    )

    gen.generate(
        dataset_name=args.dataset,
        output_path=args.out,
        n_samples=args.n_samples,
        seq_len=args.seq_len,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate teacher soft labels for Anthos distillation")
    parser.add_argument("--teacher",      type=str, default=None,
                        help="HuggingFace model name (e.g., Qwen/Qwen3-7B)")
    parser.add_argument("--dataset",      type=str, default="roneneldan/TinyStories")
    parser.add_argument("--n_samples",    type=int, default=50_000)
    parser.add_argument("--top_k",        type=int, default=64,
                        help="Save top-k token logprobs per position (memory efficient)")
    parser.add_argument("--seq_len",      type=int, default=512)
    parser.add_argument("--out",          type=str, default="data/teacher_labels.jsonl")
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    generate_labels(args)


if __name__ == "__main__":
    main()
