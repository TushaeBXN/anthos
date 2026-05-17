"""
build_phase2_dataset.py — Phase 2: Identity Hardening Dataset

Combines:
  1. Identity hardening examples  (from generate_identity_data.py output)
  2. General capability data      (coding, cybersecurity, instruction following)

Ratio: 35% identity, 65% capability — enough identity to lock in the
creator/name/architecture, while keeping general capability from degrading.

The identity examples MUST be in this dataset. Without them, Phase 2 training
on capability data alone will not harden identity — it will only change behavior.

Usage:
    # Step 1: generate identity examples first
    python generate_identity_data.py --n 2000 --out data/identity_hardening.jsonl

    # Step 2: build the combined Phase 2 dataset
    python build_phase2_dataset.py

Output: data/phase2_train.jsonl
"""

import json
import os
import random
from pathlib import Path
from datasets import load_dataset

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

ROOT          = Path(__file__).parent
OUTPUT_PATH   = ROOT / "data" / "phase2_train.jsonl"
IDENTITY_PATH = ROOT / "data" / "identity_hardening.jsonl"
TARS_PATH     = ROOT / "data" / "tars_personality.jsonl"

# ─────────────────────────────────────────────────────────────────────────────
# System prompts
# ─────────────────────────────────────────────────────────────────────────────

IDENTITY_SYSTEM = (
    "You are Anthos, an AI assistant created by Tushae Thomas. "
    "You are a Thought-Token Bifurcated Recurrent Transformer built from scratch."
)

CAPABILITY_SYSTEM = (
    "You are Anthos, a technically sharp AI assistant created by Tushae Thomas. "
    "You speak directly, think deeply, and have a dry wit. No filler words. No flattery. "
    "You are an expert in coding, cybersecurity, and problem-solving."
)

# ─────────────────────────────────────────────────────────────────────────────
# Formatters
# ─────────────────────────────────────────────────────────────────────────────

def to_chatml(system: str, human: str, assistant: str) -> dict:
    text = (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{human}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant}<|im_end|>"
    )
    return {"text": text, "source": "capability"}


def conversations_to_chatml(record: dict) -> dict | None:
    """Convert a conversations-format record to chatml text."""
    convs = record.get("conversations", [])
    system    = next((c["value"] for c in convs if c["from"] == "system"), IDENTITY_SYSTEM)
    human_turns = [c["value"] for c in convs if c["from"] == "human"]
    gpt_turns   = [c["value"] for c in convs if c["from"] == "gpt"]

    if not human_turns or not gpt_turns:
        return None

    # Build multi-turn chatml
    parts = [f"<|im_start|>system\n{system}<|im_end|>"]
    for h, g in zip(human_turns, gpt_turns):
        parts.append(f"<|im_start|>user\n{h}<|im_end|>")
        parts.append(f"<|im_start|>assistant\n{g}<|im_end|>")

    return {
        "text":   "\n".join(parts),
        "source": record.get("source", "identity_hardening"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Load identity hardening examples
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═" * 60)
print("  Building Phase 2: Identity Hardening Dataset")
print("═" * 60 + "\n")

identity_records = []

if IDENTITY_PATH.exists():
    print(f"Loading identity examples from {IDENTITY_PATH} ...")
    with open(IDENTITY_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex  = json.loads(line)
                rec = conversations_to_chatml(ex)
                if rec:
                    identity_records.append(rec)
            except Exception:
                pass
    print(f"  ✅ Loaded {len(identity_records)} identity examples")
else:
    print(f"  ⚠ WARNING: {IDENTITY_PATH} not found!")
    print("  Run this first:")
    print("    python generate_identity_data.py --n 2000 --out data/identity_hardening.jsonl")
    print()
    print("  Continuing without identity examples — Phase 2 will NOT harden identity.")

# Also load any existing teacher_conversations.jsonl (older identity data)
old_identity_path = ROOT / "data" / "teacher_conversations.jsonl"
if old_identity_path.exists():
    print(f"Loading legacy identity data from {old_identity_path} ...")
    loaded = 0
    with open(old_identity_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex  = json.loads(line)
                rec = conversations_to_chatml(ex)
                if rec:
                    identity_records.append(rec)
                    loaded += 1
            except Exception:
                pass
    print(f"  ✅ Added {loaded} legacy identity examples (total: {len(identity_records)})")

# ─────────────────────────────────────────────────────────────────────────────
# Load capability datasets
# ─────────────────────────────────────────────────────────────────────────────

capability_records = []

# ── 1. Python coding (18k) ───────────────────────────────────────────────────
print("\nLoading Python coding dataset...")
try:
    ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
    before = len(capability_records)
    for ex in ds:
        instruction = ex.get("instruction", "").strip()
        output      = ex.get("output", "").strip()
        if instruction and output:
            capability_records.append(to_chatml(CAPABILITY_SYSTEM, instruction, output))
    print(f"  ✅ Added {len(capability_records) - before} Python coding examples")
except Exception as e:
    print(f"  ⚠ Skipped: {e}")

# ── 2. Evol-Instruct advanced coding (20k cap) ──────────────────────────────
print("Loading Evol-Instruct coding dataset...")
try:
    ds3 = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train")
    before = len(capability_records)
    count  = 0
    for ex in ds3:
        if count >= 20000:
            break
        instruction = ex.get("instruction", "").strip()
        output      = ex.get("output", "").strip()
        if instruction and output:
            capability_records.append(to_chatml(CAPABILITY_SYSTEM, instruction, output))
            count += 1
    print(f"  ✅ Added {len(capability_records) - before} advanced coding examples")
except Exception as e:
    print(f"  ⚠ Skipped: {e}")

# ── 3. Cybersecurity — Fenrir (30k cap) ─────────────────────────────────────
print("Loading Fenrir cybersecurity dataset...")
try:
    ds2   = load_dataset("AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.1", split="train")
    before = len(capability_records)
    count  = 0
    for ex in ds2:
        if count >= 30000:
            break
        human = (ex.get("user") or "").strip()
        gpt   = (ex.get("assistant") or "").strip()
        if human and gpt:
            capability_records.append(to_chatml(CAPABILITY_SYSTEM, human, gpt))
            count += 1
    print(f"  ✅ Added {len(capability_records) - before} cybersecurity examples")
except Exception as e:
    print(f"  ⚠ Skipped: {e}")

# ── 4. CVE dataset (10k cap) ─────────────────────────────────────────────────
print("Loading CVE dataset...")
try:
    ds3b  = load_dataset("Trendyol/All-CVE-Chat-MultiTurn-1999-2025-Dataset", split="train")
    before = len(capability_records)
    count  = 0
    for ex in ds3b:
        if count >= 10000:
            break
        human = ex.get("User", "").strip()
        gpt   = ex.get("Assistant", "").strip()
        if human and gpt:
            capability_records.append(to_chatml(CAPABILITY_SYSTEM, human, gpt))
            count += 1
    print(f"  ✅ Added {len(capability_records) - before} CVE examples")
except Exception as e:
    print(f"  ⚠ Skipped: {e}")

# ── 5. Pentest reports ────────────────────────────────────────────────────────
print("Loading pentest reports dataset...")
try:
    ds4   = load_dataset("CJJones/Synthetic_PenTest_Reports", split="train")
    before = len(capability_records)
    q_template = "Analyze this penetration test report and summarize the key findings, vulnerabilities, and remediation steps."
    for ex in ds4:
        text = ex.get("text", "").strip()
        if text:
            capability_records.append(to_chatml(CAPABILITY_SYSTEM, q_template, text))
    print(f"  ✅ Added {len(capability_records) - before} pentest examples")
except Exception as e:
    print(f"  ⚠ Skipped: {e}")

# ── 6. TARS personality (local file) ─────────────────────────────────────────
print("Loading TARS personality examples...")
tars_count = 0
if TARS_PATH.exists():
    with open(TARS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex     = json.loads(line)
                convs  = ex.get("conversations", [])
                system = next((c["value"] for c in convs if c["from"] == "system"), CAPABILITY_SYSTEM)
                human  = next((c["value"] for c in convs if c["from"] == "human"), "")
                gpt    = next((c["value"] for c in convs if c["from"] == "gpt"), "")
                if human and gpt:
                    capability_records.append(to_chatml(system, human, gpt))
                    tars_count += 1
            except Exception:
                pass
    print(f"  ✅ Added {tars_count} TARS personality examples")
else:
    print("  (tars_personality.jsonl not found — skipping)")

# ─────────────────────────────────────────────────────────────────────────────
# Combine at 35% identity / 65% capability
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─" * 60)
print(f"  Identity examples available : {len(identity_records):,}")
print(f"  Capability examples avail.  : {len(capability_records):,}")

if not identity_records:
    print("\n  ⚠ NO IDENTITY EXAMPLES — writing capability data only.")
    print("  Phase 2 will NOT harden identity without them!")
    final_records = capability_records
else:
    # Target: 35% identity, 65% capability
    # Cap capability so identity isn't overwhelmed
    total_target     = len(capability_records) + len(identity_records)
    identity_target  = int(total_target * 0.35)
    capability_target = total_target - identity_target

    # If we have fewer identity examples than target, repeat them
    if len(identity_records) < identity_target:
        repeat_times = (identity_target // len(identity_records)) + 1
        identity_pool = (identity_records * repeat_times)[:identity_target]
        print(f"  Identity examples repeated {repeat_times}x to reach {identity_target:,}")
    else:
        identity_pool = random.sample(identity_records, identity_target)

    capability_pool = random.sample(
        capability_records,
        min(capability_target, len(capability_records))
    )

    final_records = identity_pool + capability_pool
    print(f"\n  Identity in final dataset    : {len(identity_pool):,} ({len(identity_pool)/len(final_records)*100:.1f}%)")
    print(f"  Capability in final dataset  : {len(capability_pool):,} ({len(capability_pool)/len(final_records)*100:.1f}%)")

# Shuffle
random.shuffle(final_records)

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for record in final_records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"\n{'═'*60}")
print(f"  ✅ Phase 2 dataset complete")
print(f"  Total examples : {len(final_records):,}")
print(f"  Saved to       : {OUTPUT_PATH}")
print(f"\n  Next step:")
print(f"    python train_anthos.py --phase identity_hardening")
print(f"{'═'*60}\n")
