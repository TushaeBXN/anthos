#!/usr/bin/env bash
# Copy a run's checkpoints, metrics.jsonl and train.log to a persistent destination so a dead pod loses nothing.
# Usage: ./offload_run.sh <run-name> <dest> [interval-seconds]
#   <dest> is rsync syntax: a local dir, or user@host:/path (e.g. your Mac over ssh, or a mounted volume).
# Run it beside training:  nohup ./offload_run.sh my-run user@host:/backups/ 300 > offload.log 2>&1 &
# Train with:              mkdir -p checkpoints/my-run && python train.py ... --run-name my-run 2>&1 | tee checkpoints/my-run/train.log
set -u
RUN="${1:?run name}"; DEST="${2:?destination}"; EVERY="${3:-300}"
SRC="checkpoints/${RUN}/"
while true; do
  if [ -d "$SRC" ]; then
    rsync -a --partial "$SRC" "${DEST%/}/${RUN}/" && echo "$(date '+%F %T') synced ${RUN}" || echo "$(date '+%F %T') rsync FAILED"
  fi
  [ "${OFFLOAD_ONCE:-0}" = "1" ] && break
  sleep "$EVERY"
done
