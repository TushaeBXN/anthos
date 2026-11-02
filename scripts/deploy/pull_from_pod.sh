#!/usr/bin/env bash
# Pull a run's checkpoints, metrics and log from the pod to this Mac (the pod can't reach the Mac, so we pull).
# Usage: ./pull_from_pod.sh <port> <host> <run> [interval-seconds]
PORT="${1:?port}"; HOST="${2:?host}"; RUN="${3:?run}"; EVERY="${4:-600}"
mkdir -p "checkpoints/pod-${RUN}"
while true; do
  rsync -a --partial -e "ssh -p ${PORT} -o BatchMode=yes -o ConnectTimeout=20" "root@${HOST}:/workspace/anthos/checkpoints/${RUN}/" "checkpoints/pod-${RUN}/" >/dev/null 2>&1 \
    && echo "$(date '+%F %T') pulled ${RUN}" || echo "$(date '+%F %T') pull FAILED"
  sleep "$EVERY"
done
