#!/usr/bin/env bash
# Environment for Jittor + CUDA 11.8.
# Source this in any shell session, or use `jt_python` wrapper for one-shots.

# CUDA 11.8 (system /usr/bin/nvcc is 11.5 and breaks g++11)
export PATH=/usr/local/cuda-11.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-}

# Force jittor to use system g++/gcc (not conda's)
export cc_path=/usr/bin/g++
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

# Reduce logging spam
export log_silent=1

export V3_ROOT=/mnt/ssd4t/data/chx/graphics/claude2/v3_pcd_mlgc_agt
export DATA_ROOT=/mnt/ssd4t/data/chx/graphics

export MAMBA_BIN=/mnt/ssd4t/data/chx/micromamba/bin/micromamba
export JT_ENV=bighw-jittor

# One-shot python launcher; usage: jt_python script.py [args]
jt_python() {
    "$MAMBA_BIN" run -n "$JT_ENV" python "$@"
}
export -f jt_python 2>/dev/null || true
