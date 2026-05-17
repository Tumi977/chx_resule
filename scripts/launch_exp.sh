#!/usr/bin/env bash
# Generic experiment launcher.
# Usage: launch_exp.sh <gpu_id> <exp_name> <train_script> [extra_args...]
# Always launches detached; logs to logs/<exp_name>/{train.log, nohup.out}.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"

GPU=$1; shift
EXP=$1; shift
SCRIPT=$1; shift

LOGDIR="$ROOT/logs/$EXP"
mkdir -p "$LOGDIR"
PIDF="$LOGDIR/train.pid"

if [ -f "$PIDF" ] && ps -p "$(cat "$PIDF")" > /dev/null 2>&1; then
    echo "[skip] $EXP already running pid=$(cat "$PIDF")"
    exit 0
fi

cd "$ROOT"
CUDA_VISIBLE_DEVICES=$GPU nohup \
    "$MAMBA_BIN" run -n "$JT_ENV" python -u "$SCRIPT" \
        --exp_name "$EXP" "$@" \
    > "$LOGDIR/nohup.out" 2>&1 &
echo $! > "$PIDF"
sleep 1
echo "[ok] launched $EXP pid=$(cat "$PIDF") on GPU=$GPU"
echo "  log : $LOGDIR/train.log"
echo "  out : $LOGDIR/nohup.out"
echo "  stop: kill \$(cat $PIDF)"
