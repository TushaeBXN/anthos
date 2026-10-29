#!/usr/bin/env bash
# Run once on a fresh GPU pod from the repo root. Checks the GPU works with this torch, installs the few extra deps.
set -e
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda)" || pip install torch
pip install -q numpy==2.4.6 tiktoken==0.14.0 tokenizers==0.23.3 tqdm
python - <<'PY'
import torch
assert torch.cuda.is_available(), "no CUDA"
a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize(); print("bf16 matmul ok:", (a @ a).float().abs().mean().item() > 0, torch.cuda.get_device_name(0))
PY
df -h . | tail -1
