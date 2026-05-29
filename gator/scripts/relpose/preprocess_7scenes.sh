#!/usr/bin/env bash
set -euo pipefail

SCENES=(chess fire heads office pumpkin redkitchen stairs)

MAX_JOBS=2
LOG_DIR="prepare_7scenes_logs"
mkdir -p "$LOG_DIR"

for scene in "${SCENES[@]}"; do
    echo "Launching $scene"

    log_file="$LOG_DIR/${scene}.log"

    CUDA_VISIBLE_DEVICES=0 \
    python gator/scripts/relpose/load_dataset.py --scenes "$scene" 2>&1 | tee "$log_file" &

    while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep 1
    done
done

wait
echo "Done. Logs saved in $LOG_DIR"
