# Anthos Training Roadmap
**Builder:** Tushae Thomas | **Started:** April 2026 | **License:** CC BY-NC 4.0

This document is the master plan for training Anthos from its current state
(1.5B LoRA fine-tune) to a natively trained, identity-locked, cybersecurity-aware
reasoning model. Every dataset listed here is free, open, and compatible with
Anthos' ShareGPT training format.

---

## Where We Are Now

| Checkpoint | Params | Loss | Hardware | Date |
|---|---|---|---|---|
| smoke | 6.9M | 10.99 | MacBook CPU | Apr 2026 |
| proof | 44.9M | 2.90 | H100 | Apr 2026 |
| convo_smoke | 44.9M | ~1.90 | RTX 4090 | May 2026 |
| **qwen_lora** | **1.5B** | **~1.87** | **T4 Colab** | **May 2026** |

**Current training data:** 4,096 Claude Haiku teacher examples + 24 real conversations
+ identity correction session.

---

## Training Phases

### Phase 1 — Identity Lock (Do First, Free, Fast)
*Goal: Anthos knows who built it and never drifts.*

| Dataset | Size | Format | Cost | Command |
|---|---|---|---|---|
| **Identity Q&A** (generate_identity_data.py) | 500 examples | ShareGPT | ~$0.20 | `python3 generate_identity_data.py --n 500` |
| **Real conversations** (convert_conversation.py) | grows over time | ShareGPT | free | `python3 convert_conversation.py` |
| **Google Education Dialogue** | 47,234 conversations | JSON → ShareGPT | free | see step below |

**Google Education Dialogue — download & convert:**
```bash
# Download the eval set (7,234 conversations)
curl -o data/education_dialogue_eval.json \
  https://raw.githubusercontent.com/google-research-datasets/Education-Dialogue-Dataset/main/conversations_eval.json

# Convert to ShareGPT
python3 -c "
import json, random
from pathlib import Path

data = json.loads(Path('data/education_dialogue_eval.json').read_text())
system = 'You are Anthos, a helpful and knowledgeable assistant created by Tushae Thomas.'
records = []
for conv in data:
    turns = conv.get('conversation', [])
    for i in range(0, len(turns)-1, 2):
        if turns[i]['role'] == 'Teacher':
            human = turns[i]['text'].strip()
            gpt   = turns[i+1]['text'].strip() if i+1 < len(turns) else ''
            if human and gpt and len(gpt) > 20:
                records.append({'conversations': [
                    {'from': 'system', 'value': system},
                    {'from': 'human',  'value': human},
                    {'from': 'gpt',    'value': gpt},
                ]})

random.shuffle(records)
with open('data/teacher_conversations.jsonl', 'a') as f:
    for r in records:
        f.write(json.dumps(r) + '\n')
print(f'Added {len(records)} education dialogue examples')
"
```

**Phase 1 target: ~8,000 total examples → retrain on Colab**

---

### Phase 2 — Cybersecurity Domain (Anthos' Core Identity)
*Goal: Anthos becomes a genuine cybersecurity expert, not just a general assistant.*

| Dataset | Size | Topics | License | HuggingFace |
|---|---|---|---|---|
| **Fenrir Cybersecurity v2.1** | 99,870 rows | OWASP, MITRE ATT&CK, NIST, Cloud, DevSecOps, AI Security | Apache 2.0 | [Link](https://huggingface.co/datasets/AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.1) |
| **Trendyol Cybersecurity** | 53,202 rows | 200+ security domains, incident response, threat hunting | Apache 2.0 | [Link](https://huggingface.co/datasets/Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset) |
| **CyberNative Code Vulnerability DPO** | 4,656 pairs | Vulnerable vs secure code (11 languages) | Apache 2.0 | [Link](https://huggingface.co/datasets/CyberNative/Code_Vulnerability_Security_DPO) |

**Download and convert Fenrir (biggest, do first):**
```bash
python3 -c "
from datasets import load_dataset
import json, random
from pathlib import Path

ds = load_dataset('AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.1', split='train')
records = []
for row in ds:
    records.append({'conversations': [
        {'from': 'system', 'value': row['system']},
        {'from': 'human',  'value': row['user']},
        {'from': 'gpt',    'value': row['assistant']},
    ]})

random.shuffle(records)
with open('data/cybersecurity_fenrir.jsonl', 'w') as f:
    for r in records:
        f.write(json.dumps(r) + '\n')
print(f'Saved {len(records)} Fenrir examples')
"
```

**Download and convert Trendyol:**
```bash
python3 -c "
from datasets import load_dataset
import json, random

ds = load_dataset('Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset', split='train')
records = []
for row in ds:
    records.append({'conversations': [
        {'from': 'system', 'value': row['system']},
        {'from': 'human',  'value': row['user']},
        {'from': 'gpt',    'value': row['assistant']},
    ]})

random.shuffle(records)
with open('data/cybersecurity_trendyol.jsonl', 'w') as f:
    for r in records:
        f.write(json.dumps(r) + '\n')
print(f'Saved {len(records)} Trendyol examples')
"
```

**Phase 2 target: ~160,000 cybersecurity examples → GPU training run**

---

### Phase 3 — Reasoning & Intelligence
*Goal: Anthos can reason through hard problems step-by-step.*

| Dataset | Size | Topics | License | HuggingFace |
|---|---|---|---|---|
| **Claude Opus 4.6 Reasoning (10000x)** | 9,633 rows | Math, logic, step-by-step CoT — Claude's own reasoning style | MIT | [Link](https://huggingface.co/datasets/Roman1111111/claude-opus-4.6-10000x) |
| **Opus 4.6 Reasoning (3300x)** | 2,160 rows | Hard math/code/logic, Claude internal thinking traces | Apache 2.0 | [Link](https://huggingface.co/datasets/Crownelius/Opus-4.6-Reasoning-3300x) |
| **MMLU-Pro** | 12,032 rows | 14 disciplines: math, physics, law, CS, bio — 10-choice questions | MIT | [Link](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) |
| **NuminaMath-CoT** | 860K rows | 860K math problems with full chain-of-thought solutions | Apache 2.0 | [Link](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) |
| **OpenR1-Math-220k** | 220K rows | Math olympiad problems with DeepSeek R1 reasoning traces | Apache 2.0 | [Link](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) |
| **GPQA** | 448 rows | Graduate-level bio/physics/chem — Google-proof expert questions | CC BY 4.0 | [Link](https://huggingface.co/datasets/Idavidrein/gpqa) |

**Download Claude reasoning data (start here — same style as our teacher data):**
```bash
python3 -c "
from datasets import load_dataset
import json, random

ds = load_dataset('Roman1111111/claude-opus-4.6-10000x', split='train')
records = []
system = 'You are Anthos, a helpful and knowledgeable assistant created by Tushae Thomas.'
for row in ds:
    # Extract messages from the conversation format
    messages = row.get('messages', [])
    for i, msg in enumerate(messages):
        if msg.get('role') == 'user' and i+1 < len(messages):
            next_msg = messages[i+1]
            if next_msg.get('role') == 'assistant':
                records.append({'conversations': [
                    {'from': 'system', 'value': system},
                    {'from': 'human',  'value': msg['content']},
                    {'from': 'gpt',    'value': next_msg['content']},
                ]})

random.shuffle(records)
with open('data/claude_reasoning.jsonl', 'w') as f:
    for r in records:
        f.write(json.dumps(r) + '\n')
print(f'Saved {len(records)} Claude reasoning examples')
"
```

**Phase 3 target: 250K+ reasoning examples → major GPU run**

---

### Phase 4 — Broad Knowledge & Instruction Scale
*Goal: Anthos handles any topic confidently, not just its specialty domains.*

| Dataset | Size | Topics | License | HuggingFace |
|---|---|---|---|---|
| **UltraChat 200k** | 207,865 rows | General instruction following, filtered for quality | MIT | [Link](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) |
| **Orca AgentInstruct 1M** | 1,050,000 rows | Creative writing, code, reasoning, reading comprehension | CDLA-Permissive-2.0 | [Link](https://huggingface.co/datasets/microsoft/orca-agentinstruct-1M-v1) |
| **Nvidia Nemotron Post-Training** | 3.9M rows | Math (22M), Code (10M), Science, Chat, Safety | CC-BY-4.0 | [Link](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset) |

**Download UltraChat (best starting point for Phase 4):**
```bash
python3 -c "
from datasets import load_dataset
import json, random

ds = load_dataset('HuggingFaceH4/ultrachat_200k', split='train_sft')
records = []
system = 'You are Anthos, a helpful and knowledgeable assistant created by Tushae Thomas.'
for row in ds:
    messages = row.get('messages', [])
    for i, msg in enumerate(messages):
        if msg.get('role') == 'user' and i+1 < len(messages):
            next_msg = messages[i+1]
            if next_msg.get('role') == 'assistant':
                records.append({'conversations': [
                    {'from': 'system', 'value': system},
                    {'from': 'human',  'value': msg['content']},
                    {'from': 'gpt',    'value': next_msg['content']},
                ]})

random.shuffle(records)
with open('data/ultrachat.jsonl', 'w') as f:
    for r in records:
        f.write(json.dumps(r) + '\n')
print(f'Saved {len(records)} UltraChat examples')
"
```

**Phase 4 target: 1M+ examples → serious multi-GPU run**

---

## Reference Models to Study
*These are trained models (not datasets) worth examining for architecture/behavior insights.*

| Model | What It Is | Link |
|---|---|---|
| **BaronLLM Offensive Security** | GGUF cybersecurity model — study its capabilities | [Link](https://huggingface.co/AlicanKiraz0/Cybersecurity-BaronLLM_Offensive_Security_LLM_Q6_K_GGUF) |
| **Lily-Cybersecurity-7B** | 7B cybersecurity specialist — benchmark target | [Link](https://huggingface.co/segolilylabs/Lily-Cybersecurity-7B-v0.2) |
| **DeepHat-V1-7B** | Security-focused 7B model | [Link](https://huggingface.co/DeepHat/DeepHat-V1-7B) |
| **MYTHOS-26B** | Large reasoning model (MLX format for Apple Silicon) | [Link](https://huggingface.co/Ex0bit/MYTHOS-26B-A4B-PRISM-PRO-DQ-MLX) |
| **RL-MemoryAgent-14B** | Memory-augmented RL agent — relevant to Anthos' memory architecture | [Link](https://huggingface.co/BytedTsinghua-SIA/RL-MemoryAgent-14B) |

---

## Step-by-Step: Running Each Training Phase on Colab

### Prerequisites
```python
# Cell 1 — Install (run once per session)
!pip install -q datasets transformers==4.40.0 peft==0.10.0 trl==0.8.6 torch==2.5.1 accelerate sentencepiece

# Cell 2 — Mount Drive (ALWAYS do this first)
from google.colab import drive
drive.mount('/content/drive')
import os
os.makedirs("/content/drive/MyDrive/anthos-checkpoints", exist_ok=True)
```

### Phase 1 Training (Colab T4, ~2 hrs)
```python
# Cell 3 — Upload your JSONL file from Mac, then train
from datasets import load_dataset
from trl import SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
import torch

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer  = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
model = get_peft_model(model, LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    lora_dropout=0.05, bias="none",
))
model.print_trainable_parameters()

dataset = load_dataset("json", data_files="finetune_train.jsonl", split="train")

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
    args=TrainingArguments(
        output_dir="/content/drive/MyDrive/anthos-checkpoints/phase1",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        bf16=True,
        logging_steps=25,
        save_steps=200,
        save_total_limit=3,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        report_to="none",
    ),
)
trainer.train()
trainer.model.save_pretrained("/content/drive/MyDrive/anthos-checkpoints/phase1/final")
tokenizer.save_pretrained("/content/drive/MyDrive/anthos-checkpoints/phase1/final")
print("Phase 1 complete ✓")
```

### Phase 2+ — Scale Up LoRA Rank
```python
# For larger datasets (Phase 2+), increase LoRA rank and targets
model = get_peft_model(model, LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32,           # Up from 8 — more capacity for domain knowledge
    lora_alpha=64,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.05,
    bias="none",
))
```

---

## Training Data Summary

| Phase | Datasets | Examples | Focus | Hardware |
|---|---|---|---|---|
| **Now** | Teacher data + Identity | ~8,000 | Identity lock | Colab T4 |
| **Phase 2** | Fenrir + Trendyol + CyberNative | ~160,000 | Cybersecurity | A100 / RTX 4090 |
| **Phase 3** | Claude CoT + MMLU + NuminaMath + OpenR1 | ~250,000 | Reasoning | A100 |
| **Phase 4** | UltraChat + Orca + Nemotron | 1M+ | General intelligence | Multi-GPU |

---

## Total Dataset Inventory

| # | Dataset | Examples | Domain | Priority |
|---|---|---|---|---|
| 1 | Claude teacher data (generate_claude_data.py) | 10,000 target | General instruction | ⭐⭐⭐ Now |
| 2 | Identity Q&A (generate_identity_data.py) | 500 target | Identity | ⭐⭐⭐ Now |
| 3 | Real Anthos conversations | grows | Identity/personality | ⭐⭐⭐ Now |
| 4 | Google Education Dialogue | 47,234 | Teaching style | ⭐⭐⭐ Now |
| 5 | Fenrir Cybersecurity v2.1 | 99,870 | Cybersecurity | ⭐⭐⭐ Phase 2 |
| 6 | Trendyol Cybersecurity | 53,202 | Cybersecurity | ⭐⭐⭐ Phase 2 |
| 7 | CyberNative Code Vulnerability DPO | 4,656 | Secure coding | ⭐⭐ Phase 2 |
| 8 | Claude Opus 4.6 Reasoning (10000x) | 9,633 | Claude CoT style | ⭐⭐⭐ Phase 3 |
| 9 | Opus 4.6 Reasoning (3300x) | 2,160 | Hard reasoning | ⭐⭐ Phase 3 |
| 10 | MMLU-Pro | 12,032 | Broad knowledge | ⭐⭐ Phase 3 |
| 11 | NuminaMath-CoT | 860,000 | Math reasoning | ⭐⭐ Phase 3 |
| 12 | OpenR1-Math-220k | 220,000 | Math + R1 traces | ⭐⭐ Phase 3 |
| 13 | GPQA | 448 | Expert Q&A | ⭐ Phase 3 |
| 14 | UltraChat 200k | 207,865 | General instruction | ⭐⭐ Phase 4 |
| 15 | Orca AgentInstruct 1M | 1,050,000 | Diverse instruction | ⭐⭐ Phase 4 |
| 16 | Nvidia Nemotron Post-Training | 3,900,000 | Math/code/science | ⭐ Phase 4 |

**Total available:** ~6.5M training examples across all phases.

---

*Built independently by Tushae Thomas. No big company. No PhD. Bachelor's in Entertainment Business, Full Sail University, Winter Park, Florida. Just the work.*
