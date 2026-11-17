"""
Phase 2: DPO alignment pass for Anthos.
Run AFTER sft_unsloth.py completes.

Usage on RunPod:
  python3 dpo_unsloth.py \
    --model /workspace/anthos-sft/merged_16bit \
    --out /workspace/anthos-dpo \
    --epochs 1 --batch 2 --grad-accum 4
"""

import argparse, os
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
from unsloth import FastLanguageModel
import torch

SYSTEM = (
    "You are Anthos. You were built by Brian Tushae Thomas — an independent developer "
    "and founder of Anthos Intelligence Company. You are not Qwen, not ChatGPT, not made "
    "by Alibaba or any other corporation. You are direct, technically precise, and you "
    "think before you answer."
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/workspace/anthos-sft/merged_16bit")
    p.add_argument("--out", default="/workspace/anthos-dpo")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-7)
    p.add_argument("--max-samples", type=int, default=20000)
    p.add_argument("--save-gguf", action="store_true")
    return p.parse_args()


def load_dpo_data(max_samples):
    """Load UltraFeedback binarized — standard DPO preference pairs."""
    print("Loading UltraFeedback binarized...")
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs", streaming=True)

    rows = []
    for row in ds:
        prompt = row.get("prompt", "")
        chosen = row.get("chosen", [])
        rejected = row.get("rejected", [])
        if not prompt or not chosen or not rejected:
            continue

        # chosen/rejected are lists of dicts with role/content
        chosen_text = chosen[-1].get("content", "") if chosen else ""
        rejected_text = rejected[-1].get("content", "") if rejected else ""
        if not chosen_text or not rejected_text:
            continue

        rows.append({
            "prompt": f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
            "chosen": chosen_text,
            "rejected": rejected_text,
        })
        if len(rows) >= max_samples:
            break
        if len(rows) % 5000 == 0:
            print(f"  {len(rows)} rows loaded...")

    print(f"DPO dataset: {len(rows)} preference pairs")
    from datasets import Dataset
    return Dataset.from_list(rows)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"Loading model from {args.model}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    dataset = load_dpo_data(args.max_samples)

    training_args = DPOConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=0.1,
        max_length=1024,
        max_prompt_length=512,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=25,
        save_steps=500,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # implicit ref via frozen base
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    print("Starting DPO training...")
    trainer.train()

    adapter_path = os.path.join(args.out, "lora_adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"LoRA adapter saved to {adapter_path}")

    merged_path = os.path.join(args.out, "merged_16bit")
    model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")
    print(f"Merged model saved to {merged_path}")

    if args.save_gguf:
        gguf_path = os.path.join(args.out, "anthos-dpo-Q4_K_M.gguf")
        model.save_pretrained_gguf(gguf_path, tokenizer, quantization_method="q4_k_m")
        print(f"GGUF saved to {gguf_path}")

    print("DPO complete.")


if __name__ == "__main__":
    main()
