"""
Anthos Local Training Script
Runs overnight on Mac — no GPU required, uses MPS (Apple Silicon) or CPU.
Usage: python train_local.py
"""

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# ── Device ──────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = "mps"
    print("Using Apple Silicon MPS")
elif torch.cuda.is_available():
    device = "cuda"
    print("Using CUDA GPU")
else:
    device = "cpu"
    print("Using CPU")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_FILE   = os.path.join(BASE_DIR, "data", "finetune_train.jsonl")
OUTPUT_DIR  = os.path.join(BASE_DIR, "checkpoints", "anthos-phase1")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load tokenizer & model (no 4-bit on Mac) ─────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

print("Loading model (this takes a minute)...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype=torch.float32,
    device_map={"": device},
)

# ── LoRA ─────────────────────────────────────────────────────────────────────
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── Dataset ──────────────────────────────────────────────────────────────────
print(f"Loading dataset from {DATA_FILE}...")
dataset = load_dataset("json", data_files=DATA_FILE, split="train")

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )

print("Tokenizing...")
dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
dataset.set_format("torch")
print(f"Dataset ready: {len(dataset)} examples")

# ── Training ─────────────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,       # small batch for Mac RAM
    gradient_accumulation_steps=8,       # effective batch = 8
    learning_rate=2e-5,
    fp16=False,                          # no fp16 on CPU/MPS
    bf16=False,
    logging_steps=25,
    save_steps=500,
    save_total_limit=3,
    warmup_steps=200,
    lr_scheduler_type="cosine",
    report_to="none",
    dataloader_pin_memory=False,
    use_cpu=(device == "cpu"),
    no_cuda=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

print("Starting training — let this run overnight...")
trainer.train()

# ── Save ─────────────────────────────────────────────────────────────────────
final_dir = os.path.join(OUTPUT_DIR, "final")
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)
print(f"\nDone. Weights saved to {final_dir}")
