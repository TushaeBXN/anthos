#!/usr/bin/env python3
"""
sft_unsloth.py — Fine-tune Anthos on merged SFT data using Unsloth.

Default base: unsloth/Qwen2.5-7B-Instruct (fast, fits in Mac/RunPod 24GB)
Swap --model for a different base (see MODELS below).

Usage (local Mac, small test):
  python3 scripts/train/sft_unsloth.py \
    --data data/anthos_local_merged.jsonl \
    --out outputs/anthos-sft \
    --max-steps 500

Usage (RunPod / full run):
  python3 scripts/train/sft_unsloth.py \
    --data data/anthos_local_merged.jsonl \
    --out outputs/anthos-sft \
    --epochs 2 \
    --batch 4 \
    --grad-accum 8

Usage (merge downloaded + local):
  python3 scripts/train/sft_unsloth.py \
    --data data/sft_anthos_v2.jsonl data/anthos_local_merged.jsonl \
    --out outputs/anthos-sft

MODELS:
  unsloth/Qwen2.5-7B-Instruct          7B, good all-rounder
  unsloth/Qwen2.5-14B-Instruct         14B, better reasoning, needs ~24GB
  unsloth/Meta-Llama-3.1-8B-Instruct   Llama 3.1 8B
  unsloth/mistral-7b-instruct-v0.3     Mistral 7B
"""

import argparse
import json
import sys
from pathlib import Path

try:
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    import torch
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset
except ImportError:
    sys.exit(
        "Install Unsloth first:\n"
        "  pip install unsloth trl datasets\n"
        "Or inside Unsloth Studio: it's already installed."
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Default is now the merged model output; override with --model for a different base
DEFAULT_MODEL = "outputs/anthos-merged"
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
MAX_SEQ_LEN = 8192   # Qwen2.5 supports up to 128k; 8k is enough for reasoning traces


def load_jsonl_files(paths: list[str]) -> list[dict]:
    """Load and deduplicate multiple JSONL files."""
    seen = set()
    records = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            print(f"  WARNING: {path} not found — skipping.")
            continue
        count = 0
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    key = json.dumps(obj, sort_keys=True)[:200]
                    if key not in seen:
                        seen.add(key)
                        records.append(obj)
                        count += 1
                except Exception:
                    pass
        print(f"  {p.name}: {count:,} records")
    return records


def has_think_block(record: dict) -> bool:
    """Return True if any assistant message contains a <think> block."""
    for msg in record.get("messages", []):
        if msg.get("role") == "assistant" and "<think>" in msg.get("content", ""):
            return True
    return False


def to_text(record: dict, tokenizer) -> str:
    """Apply chat template to a messages-format record."""
    msgs = record.get("messages", [])
    if not msgs:
        return ""
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)


def main():
    ap = argparse.ArgumentParser(description="Fine-tune Anthos with Unsloth SFT")
    ap.add_argument("--data", nargs="+", required=True,
                    help="One or more JSONL files to train on")
    ap.add_argument("--out", default="outputs/anthos-sft",
                    help="Output directory for model + LoRA adapter")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"Base model (default: {DEFAULT_MODEL})")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="Cap steps (useful for quick test runs; -1 = full dataset)")
    ap.add_argument("--batch", type=int, default=2,
                    help="Per-device batch size")
    ap.add_argument("--grad-accum", type=int, default=4,
                    help="Gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=LORA_R)
    ap.add_argument("--save-gguf", action="store_true",
                    help="Also export to GGUF (Q4_K_M) after training")
    ap.add_argument("--push-hub", default=None,
                    help="HuggingFace repo to push to (e.g. anthos-intelligence/anthos-7b)")
    a = ap.parse_args()

    out_path = Path(a.out)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── 0. Load + validate data ──────────────────────────────────────────────
    print(f"\nLoading data from: {a.data}")
    records = load_jsonl_files(a.data)
    n_with_think = sum(1 for r in records if has_think_block(r))
    n_without = len(records) - n_with_think
    print(f"  Total: {len(records):,}  |  with <think>: {n_with_think:,}  |  without: {n_without:,}")
    if n_without > len(records) * 0.3:
        print(f"  WARNING: {n_without:,} examples ({n_without/len(records):.0%}) have no <think> block.")
        print("  Run generate_reasoning_traces.py to add more reasoning trace data.")

    # ── 1. Load model ────────────────────────────────────────────────────────
    print(f"\nLoading base model: {a.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=a.model,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,          # auto-detect (bf16 on CUDA, float16 on older)
        load_in_4bit=True,   # QLoRA: 4-bit base + fp16 LoRA
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=a.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    tokenizer = get_chat_template(tokenizer, chat_template="chatml")

    # ── 2. Load data ─────────────────────────────────────────────────────────
    print("\nLoading training data...")
    records = load_jsonl_files(a.data)
    print(f"Total records: {len(records):,}")

    # Apply chat template
    texts = [to_text(r, tokenizer) for r in records]
    texts = [t for t in texts if len(t) > 50]
    print(f"Records after template: {len(texts):,}")

    dataset = Dataset.from_dict({"text": texts})

    # ── 3. Train ─────────────────────────────────────────────────────────────
    print(f"\nStarting SFT training...")
    print(f"  Model:      {a.model}")
    print(f"  Records:    {len(texts):,}")
    print(f"  Epochs:     {a.epochs}")
    print(f"  Batch:      {a.batch} × {a.grad_accum} = {a.batch * a.grad_accum} effective")
    print(f"  Output:     {out_path}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LEN,
            per_device_train_batch_size=a.batch,
            gradient_accumulation_steps=a.grad_accum,
            warmup_steps=100,
            num_train_epochs=a.epochs,
            max_steps=a.max_steps if a.max_steps > 0 else -1,
            learning_rate=a.lr,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=50,
            save_steps=500,
            save_total_limit=2,
            output_dir=str(out_path),
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            report_to="none",
        ),
    )

    trainer.train()
    print("\nTraining complete.")

    # ── 4. Save ───────────────────────────────────────────────────────────────
    model.save_pretrained(str(out_path / "lora"))
    tokenizer.save_pretrained(str(out_path / "lora"))
    print(f"LoRA adapter saved: {out_path}/lora")

    # Merged model (full weights, not just adapter)
    model.save_pretrained_merged(
        str(out_path / "merged"),
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"Merged model saved: {out_path}/merged")

    # ── 5. GGUF export (optional) ─────────────────────────────────────────────
    if a.save_gguf:
        print("\nExporting to GGUF Q4_K_M...")
        model.save_pretrained_gguf(
            str(out_path / "gguf"),
            tokenizer,
            quantization_method="q4_k_m",
        )
        print(f"GGUF saved: {out_path}/gguf")

    # ── 6. Push to Hub (optional) ─────────────────────────────────────────────
    if a.push_hub:
        print(f"\nPushing to HuggingFace Hub: {a.push_hub}")
        model.push_to_hub_merged(a.push_hub, tokenizer, save_method="merged_16bit")
        print("Done.")

    print(f"\nAnthos fine-tune complete.")
    print(f"  LoRA adapter: {out_path}/lora")
    print(f"  Merged model: {out_path}/merged")
    if a.save_gguf:
        print(f"  GGUF:         {out_path}/gguf")
    print(f"\nTo run with Ollama:")
    print(f"  ollama create anthos -f {out_path}/gguf/Modelfile")
    print(f"  ollama run anthos")


if __name__ == "__main__":
    main()
