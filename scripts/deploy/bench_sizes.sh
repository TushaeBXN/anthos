#!/bin/bash
# 100-step speed benchmark of candidate sizes on the pod's GPU. Run only when no other job is using it.
#   PATH=/venv/main/bin:$PATH ./bench_sizes.sh   -> dim 768 then dim 1024 (8 loops from step 0, real data)
# Tokens per optimizer step stay at 16 x 4 x 512 = 32,768: the micro-batch steps down until it fits in GPU
# memory and grad-accum rises to match, so tok/s is comparable across sizes.
cd /workspace/anthos || exit 1
mkdir -p checkpoints/bench
V="--pretrain-data data/pretrain_full --vocab-size 100352 --steps 100 --seed 0 --thought-prefix-only --no-rep-penalty --log-every 20 --save-every 100000"
run() { name=$1; set=$2; shift 2
  for bs in "$@"; do
    acc=$((64 / bs)); echo "== $name micro-batch $bs x accum $acc"
    python train.py --tier proof $V --run-name bench-$name --model-set "$set" --batch-size $bs --grad-accum $acc 2>&1 | tee checkpoints/bench/$name-bs$bs.log | grep -E "^step|Error|params"
    grep -q "OutOfMemory" checkpoints/bench/$name-bs$bs.log || break
    echo "   out of memory at micro-batch $bs; trying smaller"
  done; }
run d768  "dim=768,n_heads=12,n_kv_heads=4,expert_dim=384,lora_rank=8,phase1_steps=0" 8 4 2
run d1024 "dim=1024,n_heads=16,n_kv_heads=4,expert_dim=512,lora_rank=16,phase1_steps=0" 4 2 1
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
