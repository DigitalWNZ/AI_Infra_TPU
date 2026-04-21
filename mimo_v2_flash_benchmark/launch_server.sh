#!/bin/bash
# Launch SGLangJax server for MiMo-V2-Flash on v6e-32 (a6e-wangez)
# Adapted from report: tp=16 on 16 chips → tp=32 on 32 chips

set -x

export PATH="$HOME/.local/bin:$PATH"
source /sglang-jax/.venv/bin/activate

# Get node rank from environment or argument
RANK=${RANK:-$1}
RANK=${RANK:-0}

JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -u -m sgl_jax.launch_server \
    --model-path /models/MiMo-V2-Flash \
    --trust-remote-code \
    --tp-size 32 --dp-size 1 --ep-size 32 \
    --moe-backend epmoe \
    --nnodes 8 --node-rank "$RANK" \
    --dist-init-addr 10.202.0.54:30000 \
    --host 0.0.0.0 --port 30271 \
    --page-size 256 \
    --context-length 262144 \
    --disable-radix-cache \
    --chunked-prefill-size 2048 \
    --dtype bfloat16 \
    --mem-fraction-static 0.95 \
    --swa-full-tokens-ratio 0.2 \
    --skip-server-warmup \
    --log-level info \
    --max-running-requests 128
