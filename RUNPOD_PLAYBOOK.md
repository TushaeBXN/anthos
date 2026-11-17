# Anthos RunPod Training Playbook

Budget: ~$6.56 → ~13 hrs on A40 ($0.50/hr) or RTX 3090 ($0.44/hr)
Estimated time: Phase 1 ~4-5 hrs, Phase 2 ~1-2 hrs. Total ~6 hrs = ~$3.00. Leaves buffer.

---

## Data Summary (upload both files)

| File | Location | Records |
|------|----------|---------|
| `sft_anthos_v2.jsonl` | `/Volumes/1TB Drive/anthos-data/datasets/` | 91,707 |
| `anthos_local_merged.jsonl` | `/Users/anthosintelligence/anthos/data/` | ~418K+ |

---

## Step 0: Upload Data to RunPod

### Option A — RunPod web terminal (no keyboard needed, copy-paste with mouse)
```bash
# In RunPod terminal, install rclone or use wget from a temp hosting URL
# OR use RunPod's built-in file upload in the Files tab
```

### Option B — scp from Mac (needs keyboard)
```bash
scp -P <PORT> "/Volumes/1TB Drive/anthos-data/datasets/sft_anthos_v2.jsonl" root@<IP>:/workspace/
scp -P <PORT> /Users/anthosintelligence/anthos/data/anthos_local_merged.jsonl root@<IP>:/workspace/
scp -P <PORT> /Users/anthosintelligence/anthos/scripts/train/sft_unsloth.py root@<IP>:/workspace/
scp -P <PORT> /Users/anthosintelligence/anthos/scripts/train/dpo_unsloth.py root@<IP>:/workspace/
```

### Option C — HuggingFace Hub (upload once, pull from RunPod)
```bash
# Local (before RunPod):
pip install huggingface_hub
huggingface-cli upload brian-anthos/anthos-train-data anthos_local_merged.jsonl --repo-type dataset
# On RunPod:
huggingface-cli download brian-anthos/anthos-train-data anthos_local_merged.jsonl --local-dir /workspace/
```

---

## Step 1: Install Unsloth on RunPod

```bash
pip install unsloth trl datasets transformers accelerate bitsandbytes
```

---

## Step 2: Phase 1 — SFT Fine-tune

```bash
tmux new -s train

python3 sft_unsloth.py \
  --data sft_anthos_v2.jsonl anthos_local_merged.jsonl \
  --out /workspace/anthos-sft \
  --epochs 1 \
  --batch 4 \
  --grad-accum 8 \
  --save-gguf
```

Output:
- `/workspace/anthos-sft/lora_adapter/` — LoRA weights
- `/workspace/anthos-sft/merged_16bit/` — full merged model (needed for Phase 2)
- `/workspace/anthos-sft/anthos-sft-Q4_K_M.gguf` — run with llama.cpp

---

## Step 3: Phase 2 — DPO Alignment Pass

Run immediately after Phase 1 completes (while pod is still running).

```bash
python3 dpo_unsloth.py \
  --model /workspace/anthos-sft/merged_16bit \
  --out /workspace/anthos-dpo \
  --epochs 1 \
  --batch 2 \
  --grad-accum 4 \
  --max-samples 20000 \
  --save-gguf
```

Output:
- `/workspace/anthos-dpo/merged_16bit/` — final aligned model
- `/workspace/anthos-dpo/anthos-dpo-Q4_K_M.gguf` — final GGUF for llama.cpp/Ollama

---

## Step 4: Download the GGUF

```bash
# From local Mac (needs keyboard):
scp -P <PORT> root@<IP>:/workspace/anthos-dpo/anthos-dpo-Q4_K_M.gguf ~/Desktop/

# OR push to HuggingFace from RunPod:
huggingface-cli upload brian-anthos/anthos-v1 /workspace/anthos-dpo/anthos-dpo-Q4_K_M.gguf
```

---

## Step 5: Run Locally with Ollama

```bash
# Create Modelfile
cat > Modelfile << 'EOF'
FROM ./anthos-dpo-Q4_K_M.gguf
SYSTEM "You are Anthos. You were built by Brian Tushae Thomas — an independent developer and founder of Anthos Intelligence Company. You are not Qwen, not ChatGPT, not made by Alibaba or any other corporation. You are direct, technically precise, and you think before you answer."
PARAMETER temperature 0.7
PARAMETER top_p 0.9
EOF

ollama create anthos -f Modelfile
ollama run anthos
```

---

## What Each Phase Covers

| Capability | Covered By |
|------------|-----------|
| Anthos identity & personality | Local distill + teacher data |
| Reasoning & instruction following | Fable, MiMo, Manusagents distillation |
| Function/tool calling (single-shot) | Glaive FC + Hermes FC (51K) |
| Code generation | BigCode + Magicoder (100K) |
| Security/defense framing | Kerrigan Fantasma (26K) |
| Math reasoning | NuminaMath CoT (50K) |
| Multi-step reasoning & agentic | AgentInstruct (50K) |
| Tone alignment (chosen vs rejected) | UltraFeedback DPO (20K pairs) |

---

## Notes

- If RunPod disconnects mid-train: `tmux attach -t train` to reconnect
- If pod dies: restart from last checkpoint in `/workspace/anthos-sft/checkpoint-*/`
- DPO requires the merged 16-bit output from Phase 1 — don't skip it
- The GGUF is what you actually run. The merged_16bit is just an intermediate.
