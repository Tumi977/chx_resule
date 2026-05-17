#!/usr/bin/env bash
# Background launcher for P0 baseline training.
# Logs to v3_pcd_mlgc_agt/logs/p0_baseline/{train.log,nohup.out}
# Run from anywhere: bash <this script>
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# allow caller to override which GPU
GPU="${CUDA_VISIBLE_DEVICES:-0}"
EXP="${EXP_NAME:-p0_baseline}"

mkdir -p "$ROOT/logs/$EXP"
NOHUP_OUT="$ROOT/logs/$EXP/nohup.out"
PID_FILE="$ROOT/logs/$EXP/train.pid"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "[abort] training already running with pid=$(cat "$PID_FILE")"
  exit 1
fi

cd "$ROOT"

CUDA_VISIBLE_DEVICES="$GPU" nohup bash scripts/jt_python.sh train_p0.py \
    --exp_name "$EXP" \
    --epochs 60 \
    --batch_size 4 \
    --n_patches 4 \
    --patch_size 1024 \
    --lr 1e-4 \
    --lr_min 1e-6 \
    --noise_min 0.005 \
    --noise_max 0.020 \
    --num_workers 4 \
    --val_every 5 \
    --val_subset 20 \
    --encoder_dim 256 \
    --encoder_k 16 \
    --seed 42 \
    > "$NOHUP_OUT" 2>&1 &

echo $! > "$PID_FILE"
echo "[ok] launched pid=$(cat "$PID_FILE") on GPU=$GPU"
echo "  log : $ROOT/logs/$EXP/train.log"
echo "  out : $NOHUP_OUT"
echo
echo "  tail: tail -f $ROOT/logs/$EXP/train.log"
echo "  stop: kill \$(cat $PID_FILE)"
