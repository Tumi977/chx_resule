#!/usr/bin/env bash
# Compact dashboard for all running experiments.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

printf "%-22s %-9s %-6s %-9s %-22s %-7s %-32s\n" "EXP" "PID" "ALIVE" "GPU(MB)" "LAST_LOG" "BEST" "LATEST_EPOCH"
printf "%-22s %-9s %-6s %-9s %-22s %-7s %-32s\n" "---" "---" "-----" "-------" "--------" "----" "------------"
for d in logs/*/; do
    exp=$(basename "$d")
    pidf="$d/train.pid"
    pid="?"; alive="-"; mem="-"; tlog="-"; best="-"; latest="-"
    if [ -f "$pidf" ]; then
        pid=$(cat "$pidf")
        if ps -p "$pid" > /dev/null 2>&1; then
            alive="yes"
        else
            alive="DEAD"
        fi
    fi
    if [ -f "$d/train.log" ]; then
        tlog=$(stat -c %y "$d/train.log" | cut -d'.' -f1 | cut -d' ' -f2)
        best=$(grep -oP "new best score \K[0-9.]+" "$d/train.log" 2>/dev/null | tail -1)
        latest=$(grep -oE "epoch [0-9]+ done in [0-9.]+s" "$d/train.log" 2>/dev/null | tail -1)
    fi
    printf "%-22s %-9s %-6s %-9s %-22s %-7s %-32s\n" "$exp" "$pid" "$alive" "$mem" "$tlog" "${best:-—}" "${latest:-—}"
done

echo ""
echo "=== gpu utilization ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F',' '{
    printf "  GPU %s | mem %5sMB | util %3s%%\n", $1, $2, $3
}'
