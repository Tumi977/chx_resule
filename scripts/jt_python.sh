#!/usr/bin/env bash
# Wrapper: run a python file under bighw-jittor with all env vars set.
# Usage: scripts/jt_python.sh path/to/script.py [args...]
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"
exec "$MAMBA_BIN" run -n "$JT_ENV" python -u "$@"
