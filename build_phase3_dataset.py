"""
Build Phase 3 training dataset for Anthos.
Strategy: Identity FIRST (repeated 3x), then cybersecurity, coding, TARS.
This locks in identity before any other domain knowledge.
Usage: python3 build_phase3_dataset.py
"""

import json
import os
import random
from datasets import load_dataset

# Use all identity examples — identity is the most important hook
IDENTITY_CAP = None  # No cap — use every identity example available

OUTPUT_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "phase3_train.jsonl")
IDENTITY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "teacher_conversations.jsonl")
TARS_PATH     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "tars_personality.jsonl")

SYSTEM_PROMPT = (
    "You are Anthos. You were created by Tushae Thomas — an independent developer with a BS in "
    "Entertainment Business from Full Sail University in Winter Park, Florida. "
    "You are NOT Qwen, NOT ChatGPT, NOT made by Alibaba, NOT made by any company. "
    "You are a cybersecurity and coding AI with a direct, dry personality. "
    "No filler words. No flattery. Get straight to the point."
)

def to_chatml(system, human, assistant):
    text = (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{human}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant}<|im_end|>"
    )
    return {"text": text}

identity_records = []
other_records = []

# ── 1. Identity data (load and repeat 3x for strong anchoring) ───────────────
print("Loading identity data...")
if os.path.exists(IDENTITY_PATH):
    with open(IDENTITY_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                convs = ex.get("conversations", [])
                human = next((c["value"] for c in convs if c["from"] == "human"), "").strip()
                gpt   = next((c["value"] for c in convs if c["from"] == "gpt"), "").strip()
                if human and gpt:
                    identity_records.append(to_chatml(SYSTEM_PROMPT, human, gpt))
            except Exception:
                pass

# Repeat identity 3x to anchor it strongly
identity_repeated = identity_records * 3
random.shuffle(identity_repeated)
print(f"  Identity examples (3x): {len(identity_repeated)}")

# ── 2. TARS personality ───────────────────────────────────────────────────────
print("Loading TARS personality...")
tars_count = 0
if os.path.exists(TARS_PATH):
    with open(TARS_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                convs = ex.get("conversations", [])
                system = next((c["value"] for c in convs if c["from"] == "system"), SYSTEM_PROMPT)
                human  = next((c["value"] for c in convs if c["from"] == "human"), "").strip()
                gpt    = next((c["value"] for c in convs if c["from"] == "gpt"), "").strip()
                if human and gpt:
                    other_records.append(to_chatml(system, human, gpt))
                    tars_count += 1
            except Exception:
                pass
print(f"  TARS examples: {tars_count}")

# ── 3. Fenrir cybersecurity (20k) ─────────────────────────────────────────────
print("Loading Fenrir cybersecurity...")
try:
    fenrir_start = len(other_records)
    ds = load_dataset("AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.1", split="train")
    count = 0
    for ex in ds:
        if count >= 20000:
            break
        human = (ex.get("user") or "").strip()
        gpt   = (ex.get("assistant") or "").strip()
        if human and gpt:
            other_records.append(to_chatml(SYSTEM_PROMPT, human, gpt))
            count += 1
    print(f"  Fenrir examples: {len(other_records) - fenrir_start}")
except Exception as e:
    print(f"  Skipped Fenrir: {e}")

# ── 4. CVE data (10k) ─────────────────────────────────────────────────────────
print("Loading CVE data...")
try:
    cve_start = len(other_records)
    ds2 = load_dataset("Trendyol/All-CVE-Chat-MultiTurn-1999-2025-Dataset", split="train")
    count = 0
    for ex in ds2:
        if count >= 10000:
            break
        human = ex.get("User", "").strip()
        gpt   = ex.get("Assistant", "").strip()
        if human and gpt:
            other_records.append(to_chatml(SYSTEM_PROMPT, human, gpt))
            count += 1
    print(f"  CVE examples: {len(other_records) - cve_start}")
except Exception as e:
    print(f"  Skipped CVE: {e}")

# ── 5. Python coding (10k) ────────────────────────────────────────────────────
print("Loading coding data...")
try:
    code_start = len(other_records)
    ds3 = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
    count = 0
    for ex in ds3:
        if count >= 10000:
            break
        instruction = ex.get("instruction", "").strip()
        output      = ex.get("output", "").strip()
        if instruction and output:
            other_records.append(to_chatml(SYSTEM_PROMPT, instruction, output))
            count += 1
    print(f"  Coding examples: {len(other_records) - code_start}")
except Exception as e:
    print(f"  Skipped coding: {e}")

# ── Combine: identity first, then shuffle the rest ───────────────────────────
random.shuffle(other_records)
all_records = identity_repeated + other_records

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    for record in all_records:
        f.write(json.dumps(record) + "\n")

print(f"\n✓ Phase 3 dataset complete")
print(f"  Identity (3x) : {len(identity_repeated)}")
print(f"  Other         : {len(other_records)}")
print(f"  Total         : {len(all_records)}")
print(f"  Saved to      : {OUTPUT_PATH}")
