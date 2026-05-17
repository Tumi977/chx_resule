#!/usr/bin/env bash
# Watchdog: poll P0 training. When it finishes (PID dies AND log shows the
# "FULL validation score" line), launch p2_agt_T4 on GPU 4.
#
# This avoids leaving GPU 4 idle after P0 ends.

set -e
cd "$(dirname "$0")/.."

WAIT_PID_FILE="logs/p0_baseline/train.pid"
WAIT_LOG="logs/p0_baseline/train.log"
WATCH_LOG="logs/watch_p0_then_p2.out"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "watchdog started; waiting on P0 PID $(cat $WAIT_PID_FILE 2>/dev/null) and final log line."

while true; do
    if [ -f "$WAIT_PID_FILE" ]; then
        pid=$(cat "$WAIT_PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            sleep 60
            continue
        fi
    fi
    # process is dead — confirm training reached completion
    if grep -q "training done" "$WAIT_LOG" 2>/dev/null; then
        log "P0 finished cleanly. Launching p2_agt_T4 on GPU 4."
        break
    else
        log "P0 process gone but log incomplete; polling again in 30s."
        sleep 30
    fi
done

# Ensure the previous broken p2_agt_T4 dir is gone
rm -rf logs/p2_agt_T4 ckpts/p2_agt_T4

bash scripts/launch_exp.sh 4 p2_agt_T4 train_p2.py \
    --init_each_stage_from ckpts/p0_baseline/best.pkl \
    --epochs 20 \
    --batch_size 2 --n_patches 4 --patch_size 1024 \
    --lr 5e-5 --lr_min 1e-6 \
    --noise_min 0.005 --noise_max 0.020 \
    --num_workers 4 \
    --n_stages 4 --sigma_delta 2.0 --rect_flow_p 1.0 \
    --lambda_p2plane 30.0 --lambda_repulsion 0.005 \
    --val_every 5 --val_subset 20

log "p2_agt_T4 launched, watchdog exiting."
