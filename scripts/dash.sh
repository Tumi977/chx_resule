#!/usr/bin/env bash
# Dashboard: one-shot snapshot of all experiments with score history + GPU.
# Usage:
#   bash scripts/dash.sh              # full snapshot (default)
#   bash scripts/dash.sh -w            # watch mode (refresh every 30s)
#   bash scripts/dash.sh -c            # compact one-line per exp
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WATCH=0
COMPACT=0
while [ $# -gt 0 ]; do
    case "$1" in
        -w|--watch) WATCH=1 ;;
        -c|--compact) COMPACT=1 ;;
        -h|--help)
            cat << 'EOF'
Usage:
  bash scripts/dash.sh         # full snapshot
  bash scripts/dash.sh -c      # one-line per experiment (compact)
  bash scripts/dash.sh -w      # watch mode, refresh every 30s
  bash scripts/dash.sh -wc     # compact + watch
EOF
            exit 0 ;;
    esac
    shift
done

render() {
    clear 2>/dev/null || true
    echo "═══════════════════════════════════════════════════════════════════════════"
    echo "  V3 PCD denoiser experiments  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  P0 锚点 FULL val = 70.94"
    echo "═══════════════════════════════════════════════════════════════════════════"
    echo ""

    # Build summary table for sorting by best
    local tmpfile
    tmpfile=$(mktemp)
    for d in logs/*/; do
        local logf="$d/train.log"
        local pidf="$d/train.pid"
        [ -f "$logf" ] || continue

        local exp; exp=$(basename "$d")
        local alive=0 pid="-" et="-"
        if [ -f "$pidf" ]; then
            pid=$(cat "$pidf" 2>/dev/null)
            if ps -p "$pid" > /dev/null 2>&1; then
                alive=1
                et=$(ps -p "$pid" -o etime --no-headers 2>/dev/null | tr -d ' ')
            fi
        fi
        local best last_ep full scores
        best=$(grep -oE "new best score [0-9.]+" "$logf" 2>/dev/null | tail -1 | awk '{print $NF}')
        last_ep=$(grep -oE "epoch [0-9]+ done" "$logf" 2>/dev/null | tail -1 | grep -oE "[0-9]+")
        full=$(grep -oE "FULL validation score = [0-9.]+" "$logf" 2>/dev/null | tail -1 | awk '{print $NF}')
        scores=$(grep -oE "Total Score\s+:\s+[0-9.]+" "$logf" 2>/dev/null | awk '{print $NF}' | paste -sd ' ')
        # Score for ranking: prefer FULL > best > 0
        local rank_score="${full:-${best:-0}}"
        printf '%d|%012.3f|%s|%s|%s|%s|%s|%s|%s\n' "$alive" "$rank_score" \
            "$exp" "$pid" "$et" "${best:-—}" "${last_ep:-—}" "${full:-—}" "$scores" >> "$tmpfile"
    done

    if [ $COMPACT -eq 1 ]; then
        printf "  \033[1m%-24s %-6s %-4s %-7s %-7s %s\033[0m\n" "EXP" "STATE" "ep" "best" "FULL" "history"
        echo "  ───────────────────────────────────────────────────────────────────────"
        sort -t'|' -k1,1nr -k2,2nr "$tmpfile" | while IFS='|' read alive rs exp pid et best last_ep full scores; do
            local color tag
            if [ "$alive" = "1" ]; then color="\033[32m"; tag="ALIVE"
            else color="\033[90m"; tag="dead "; fi
            printf "  ${color}%-24s\033[0m [%s] ep%-3s best=\033[33m%-7s\033[0m FULL=\033[1;36m%-7s\033[0m | %s\n" \
                "$exp" "$tag" "$last_ep" "$best" "$full" "$scores"
        done
    else
        sort -t'|' -k1,1nr -k2,2nr "$tmpfile" | while IFS='|' read alive rs exp pid et best last_ep full scores; do
            local color tag
            if [ "$alive" = "1" ]; then color="\033[32m"; tag="ALIVE"
            else color="\033[90m"; tag="dead "; fi
            printf "${color}● %-24s\033[0m [%s] " "$exp" "$tag"
            printf "ep=%-3s " "$last_ep"
            printf "best=\033[33m%-7s\033[0m " "$best"
            if [ "$full" != "—" ]; then printf "FULL=\033[1;36m%-7s\033[0m " "$full"; fi
            printf "elapsed=%s\n" "$et"
            # Detailed val history
            awk '
                /epoch [0-9]+ done/ {
                    match($0, /epoch [0-9]+/); ep_val=substr($0, RSTART+6, RLENGTH-6)
                }
                /CD_score/ {cd=$3}
                /P2S_score/ {p2s=$3}
                /Total Score/ {
                    gsub(/[^0-9.]/, "", $5); total=$5
                    printf "    ep %3d  CD=%-6s  P2S=%-6s  Total=\033[1;32m%s\033[0m\n", ep_val, cd, p2s, total
                }
            ' "logs/$exp/train.log"
            # Last loss line
            local last_loss
            last_loss=$(grep -E "epoch [0-9]+ done" "logs/$exp/train.log" 2>/dev/null | tail -1 | sed 's/.*done in [0-9.]*s | //')
            [ -n "$last_loss" ] && printf "    \033[90m└─ %s\033[0m\n" "$last_loss"
            echo ""
        done
    fi
    rm -f "$tmpfile"

    echo "─── GPU utilization ───"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
        awk -F',' '{
            used=$2+0; total=$3+0; util=$4+0; free=total-used
            tag="?"
            if (util==0 && $4 ~ /N\/A/) tag="❌"
            else if (free > 30000) tag="🟢"
            else if (free > 15000) tag="🟡"
            else tag="🔴"
            printf "  %s GPU %s | %5d/%5d MB | util %3d%% | %5d MB free\n", tag, $1, used, total, util, free
        }'

    echo ""
    echo "─── 提示 ───"
    echo "  实时监控某个实验:    tail -f logs/<exp>/train.log"
    echo "  紧凑模式(一行/exp): bash scripts/dash.sh -c"
    echo "  自动刷新(30s):      bash scripts/dash.sh -w"
}

if [ $WATCH -eq 1 ]; then
    while true; do
        render
        echo "(refresh 30s; Ctrl+C to exit)"
        sleep 30
    done
else
    render
fi
