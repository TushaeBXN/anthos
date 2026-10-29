#!/bin/bash
# 100-step speed benchmark of candidate sizes on the pod's GPU. Run only when no other job is using it.
#   ./bench_sizes.sh            -> dim 768 then dim 1024 (8 loops from step 0, real data, real batch)
cd /workspace/anthos || exit 1
mkdir -p checkpoints/bench
V="--pretrain-data data/pretrain_full --vocab-size 100352 --steps 100 --seed 0 --thought-prefix-only --no-rep-penalty --log-every 20 --save-every 100000"
run() { name=$1; set=$2; extra=$3
  python train.py --tier proof $V --run-name bench-$name --model-set "$set" $extra 2>&1 | tee checkpoints/bench/$name.log | grep -E "^step|Error|error|OOM|out of memory|params" ; }
run d768  "dim=768,n_heads=12,n_kv_heads=4,expert_dim=384,lora_rank=8,phase1_steps=0"
run d1024 "dim=1024,n_heads=16,n_kv_heads=4,expert_dim=512,lora_rank=16,phase1_steps=0"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
