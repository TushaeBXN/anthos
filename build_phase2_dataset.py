"""
Build Phase 2 training dataset for Anthos.
Combines: coding, cybersecurity, vulnerability detection, TARS personality.
Usage: python3 build_phase2_dataset.py
"""

import json
import os
import random
from datasets import load_dataset

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "phase2_train.jsonl")
TARS_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "tars_personality.jsonl")

SYSTEM_PROMPT = (
    "You are Anthos, a technically sharp AI assistant created by Tushae Thomas. "
    "You speak directly, think deeply, and have a dry wit. No filler words. No flattery. "
    "You are an expert in coding, cybersecurity, and vulnerability detection."
)

def to_chatml(system, human, assistant):
    text = (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{human}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant}<|im_end|>"
    )
    return {"text": text}

records = []

# ── 1. Python coding instructions (18k) ──────────────────────────────────────
print("Loading Python coding dataset...")
ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
for ex in ds:
    instruction = ex.get("instruction", "").strip()
    output      = ex.get("output", "").strip()
    if instruction and output:
        records.append(to_chatml(SYSTEM_PROMPT, instruction, output))
print(f"  Added {len(records)} coding examples")

# ── 2. Fenrir cybersecurity dataset (99k examples) ───────────────────────────
print("Loading Fenrir cybersecurity dataset...")
try:
    cyber_start = len(records)
    ds2 = load_dataset("AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.1", split="train")
    count = 0
    for ex in ds2:
        if count >= 30000:
            break
        human = (ex.get("user") or "").strip()
        gpt   = (ex.get("assistant") or "").strip()
        if human and gpt:
            records.append(to_chatml(SYSTEM_PROMPT, human, gpt))
            count += 1
    print(f"  Added {len(records) - cyber_start} Fenrir cybersecurity examples")
except Exception as e:
    print(f"  Skipped Fenrir dataset: {e}")

# ── 3. CVE multi-turn dataset ─────────────────────────────────────────────────
print("Loading CVE dataset...")
try:
    cve_start = len(records)
    ds3b = load_dataset("Trendyol/All-CVE-Chat-MultiTurn-1999-2025-Dataset", split="train")
    count = 0
    for ex in ds3b:
        if count >= 10000:
            break
        convs = ex.get("conversations", ex.get("messages", []))
        human = ex.get("User", "").strip()
        gpt   = ex.get("Assistant", "").strip()
        if human and gpt:
            records.append(to_chatml(SYSTEM_PROMPT, human, gpt))
            count += 1
    print(f"  Added {len(records) - cve_start} CVE examples")
except Exception as e:
    print(f"  Skipped CVE dataset: {e}")

# ── 4. Pentest reports ────────────────────────────────────────────────────────
print("Loading pentest reports dataset...")
try:
    pen_start = len(records)
    ds4 = load_dataset("CJJones/Synthetic_PenTest_Reports", split="train")
    for ex in ds4:
        text = ex.get("text", "").strip()
        if text:
            q = "Analyze this penetration test report and summarize the key findings, vulnerabilities, and remediation steps."
            a = text
        if q and a:
            records.append(to_chatml(SYSTEM_PROMPT, q, a))
    print(f"  Added {len(records) - pen_start} pentest examples")
except Exception as e:
    print(f"  Skipped pentest dataset: {e}")

# ── 3. Evol-Instruct coding (advanced) ───────────────────────────────────────
print("Loading Evol-Instruct coding dataset...")
try:
    evol_start = len(records)
    ds3 = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train")
    count = 0
    for ex in ds3:
        if count >= 20000:
            break
        instruction = ex.get("instruction", "").strip()
        output      = ex.get("output", "").strip()
        if instruction and output:
            records.append(to_chatml(SYSTEM_PROMPT, instruction, output))
            count += 1
    print(f"  Added {len(records) - evol_start} advanced coding examples")
except Exception as e:
    print(f"  Skipped Evol-Instruct dataset: {e}")

# ── 4. TARS personality examples ─────────────────────────────────────────────
print("Loading TARS personality examples...")
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
                human  = next((c["value"] for c in convs if c["from"] == "human"), "")
                gpt    = next((c["value"] for c in convs if c["from"] == "gpt"), "")
                if human and gpt:
                    records.append(to_chatml(system, human, gpt))
                    tars_count += 1
            except Exception:
                pass
    print(f"  Added {tars_count} TARS personality examples")
else:
    print("  TARS file not found — skipping")

# ── Shuffle and save ──────────────────────────────────────────────────────────
random.shuffle(records)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    for record in records:
        f.write(json.dumps(record) + "\n")

print(f"\n✓ Phase 2 dataset complete")
print(f"  Total examples : {len(records)}")
print(f"  Saved to       : {OUTPUT_PATH}")
