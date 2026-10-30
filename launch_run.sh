#!/bin/bash
# Start a control or ablation run ON the pod, in its own checkpoint directory, with the watcher.
#   ./launch_run.sh <control> [steps]        e.g. ./launch_run.sh plain_dense
#   DRY=1 ./launch_run.sh act_last           print what would run, start nothing
# Controls are defined in controls.py. Each run gets checkpoints/anthos-ctl-<control>/ and never touches another run.
# Needs the repo, data/pretrain_full and /venv/main on the pod. Pull checkpoints to the Mac with pull_from_pod.sh.
set -e
C="${1:?control name (see: python controls.py)}"; STEPS="${2:-30000}"
cd "$(dirname "$0")"; export PATH=/venv/main/bin:$PATH; PY="${PY:-python}"   # PY=python3.11 on the Mac for a dry run
RUN="anthos-ctl-$C"; DIR="checkpoints/$RUN"
MS=$($PY controls.py --ms "$C"); LOOPS=$($PY controls.py --loops "$C")
[ -e "$DIR" ] && { echo "refusing: $DIR already exists (resume with train.py --resume, or pick a new name)"; exit 1; }
CMD=("$PY" train.py --tier proof --pretrain-data data/pretrain_full --vocab-size 100352 --steps "$STEPS" --seed 0
     --run-name "$RUN" --thought-prefix-only --no-rep-penalty --log-every 100 --save-every 2500
     --batch-size 8 --grad-accum 8 --model-set "$MS")
echo "run:     $RUN   ($STEPS steps; a multiple of 2500 keeps a final checkpoint)"
echo "model:   $MS"
echo "evaluate at $LOOPS loop(s)"
if [ -n "$DRY" ]; then echo "DRY: ${CMD[*]}"; exit 0; fi
if [ $((STEPS % 2500)) -ne 0 ]; then echo "refusing: steps must be a multiple of 2500"; exit 1; fi
mkdir -p "$DIR"; printf '%q ' "${CMD[@]}" > "$DIR/launch_cmd.txt"
nohup bash -c "${CMD[*]@Q} >> $DIR/train.log 2>&1; echo \"RUN_EXIT \$? \$(date -u +%FT%TZ)\" >> $DIR/train.log" >/dev/null 2>&1 &
MODEL_SET="$MS" EVAL_LOOPS="$LOOPS" nohup "$PY" watch_run.py "$RUN" "$STEPS" > "$DIR/monitor.log" 2>&1 &
sleep 2; echo "started. Mac side:  ./pull_from_pod.sh <port> <ip-without-root@> $RUN 600"
