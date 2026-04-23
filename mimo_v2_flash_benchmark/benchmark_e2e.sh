#!/bin/bash
# =============================================================================
# End-to-End MiMo-V2-Flash Benchmark on TPU v6e-32
# =============================================================================
#
# This script runs the full MiMo-V2-Flash benchmark on a TPU v6e-32 cluster:
#   1. Install dependencies (sglang-jax, evalscope)
#   2. Mount model weights from GCS via GCSFuse
#   3. Launch the SGLangJax server across all workers
#   4. Run performance benchmark (random data, throughput/latency)
#   5. Run accuracy benchmark (GSM8K via evalscope)
#
# Usage:
#   bash benchmark_e2e.sh <CLUSTER_NAME> <ZONE> <GCS_BUCKET> <GCS_MODEL_DIR>
#
# Example:
#   bash benchmark_e2e.sh a6e-wangez us-east5-b 0-wangez jingnw-mimo-v2-flash-us-east5/hf-model
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - TPU v6e-32 cluster already created and READY
#   - Model weights uploaded to GCS bucket
# =============================================================================

set -euo pipefail

CLUSTER_NAME="${1:?Usage: $0 <cluster_name> <zone> <gcs_bucket> <gcs_model_dir>}"
ZONE="${2:?Usage: $0 <cluster_name> <zone> <gcs_bucket> <gcs_model_dir>}"
GCS_BUCKET="${3:?Usage: $0 <cluster_name> <zone> <gcs_bucket> <gcs_model_dir>}"
GCS_MODEL_DIR="${4:?Usage: $0 <cluster_name> <zone> <gcs_bucket> <gcs_model_dir>}"

NUM_WORKERS=8
NUM_CHIPS=32
SERVER_PORT=30271
DIST_PORT=30000
MODEL_DIR="/tmp/mimo_model"
GCS_MOUNT="/tmp/gcs_model"
SGLANG_DIR="/sglang-jax"
VENV_DIR="${SGLANG_DIR}/.venv"
RESULTS_DIR="/tmp/benchmark_results"

SSH_OPTS="--zone=${ZONE}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

ssh_worker() {
    local worker=$1
    shift
    gcloud compute tpus tpu-vm ssh "${CLUSTER_NAME}" ${SSH_OPTS} \
        --worker="${worker}" --command="$*" 2>/dev/null
}

ssh_all_workers() {
    gcloud compute tpus tpu-vm ssh "${CLUSTER_NAME}" ${SSH_OPTS} \
        --worker=all --command="$*" 2>/dev/null
}

# Get worker-0 internal IP for distributed init
get_master_ip() {
    gcloud compute tpus tpu-vm describe "${CLUSTER_NAME}" --zone="${ZONE}" \
        --format='value(networkEndpoints[0].ipAddress)' 2>/dev/null
}

# =============================================================================
# Step 1: Install dependencies on all workers
# =============================================================================
install_dependencies() {
    log "Step 1: Installing dependencies on all workers..."

    local install_script='
set -ex

# Install uv package manager
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# Clone sglang-jax if not present
if [ ! -d /sglang-jax ]; then
    sudo git clone -b main https://github.com/sgl-project/sglang-jax.git /sglang-jax
    sudo chown -R $(whoami) /sglang-jax
else
    cd /sglang-jax
    git fetch origin main
    git checkout origin/main
fi

# Create Python 3.12 venv and install
cd /sglang-jax
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e "python[tpu]"
uv pip install evalscope==0.17.1

echo "=== Install complete on $(hostname) ==="
'
    ssh_all_workers "${install_script}"
    log "Dependencies installed on all workers."
}

# =============================================================================
# Step 2: Mount model from GCS and prepare model directory
# =============================================================================
setup_model() {
    log "Step 2: Setting up model directory on all workers..."

    local setup_script="
set -ex

# Install GCSFuse
if ! command -v gcsfuse &> /dev/null; then
    export GCSFUSE_REPO=gcsfuse-\$(lsb_release -c -s)
    echo \"deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt \\\$GCSFUSE_REPO main\" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.asc
    sudo apt-get update -qq && sudo apt-get install -y gcsfuse
fi

# Mount GCS bucket
mkdir -p ${GCS_MOUNT}
if ! mountpoint -q ${GCS_MOUNT}; then
    gcsfuse --only-dir ${GCS_MODEL_DIR} ${GCS_BUCKET} ${GCS_MOUNT}
fi

# Create hybrid model directory: local config + GCS safetensors
rm -rf ${MODEL_DIR}
mkdir -p ${MODEL_DIR}

# Copy config and tokenizer files locally (small files, fast access)
for f in config.json tokenizer_config.json tokenizer.json special_tokens_map.json generation_config.json; do
    if [ -f ${GCS_MOUNT}/\${f} ]; then
        cp ${GCS_MOUNT}/\${f} ${MODEL_DIR}/
    fi
done

# Symlink safetensors files from GCS mount (large files, streamed)
for f in ${GCS_MOUNT}/*.safetensors; do
    if [ -f \"\${f}\" ]; then
        ln -sf \"\${f}\" ${MODEL_DIR}/\$(basename \"\${f}\")
    fi
done

# Also symlink the model index
if [ -f ${GCS_MOUNT}/model.safetensors.index.json ]; then
    ln -sf ${GCS_MOUNT}/model.safetensors.index.json ${MODEL_DIR}/
fi

echo \"Model dir ready: \$(ls ${MODEL_DIR} | wc -l) files\"
echo \"=== Model setup complete on \$(hostname) ===\"
"
    ssh_all_workers "${setup_script}"
    log "Model directory prepared on all workers."
}

# =============================================================================
# Step 3: Launch SGLangJax server on all workers
# =============================================================================
launch_server() {
    log "Step 3: Launching SGLangJax server..."

    MASTER_IP=$(get_master_ip)
    log "Master IP (worker-0): ${MASTER_IP}"

    for rank in $(seq 0 $((NUM_WORKERS - 1))); do
        log "  Starting server on worker ${rank}..."
        local launch_cmd="
export PATH=\"\$HOME/.local/bin:\$PATH\"
source ${VENV_DIR}/bin/activate
nohup bash -c 'JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -u -m sgl_jax.launch_server \
    --model-path ${MODEL_DIR} \
    --trust-remote-code \
    --tp-size ${NUM_CHIPS} --dp-size 1 --ep-size ${NUM_CHIPS} \
    --moe-backend epmoe \
    --nnodes ${NUM_WORKERS} --node-rank ${rank} \
    --dist-init-addr ${MASTER_IP}:${DIST_PORT} \
    --host 0.0.0.0 --port ${SERVER_PORT} \
    --page-size 256 \
    --context-length 262144 \
    --disable-radix-cache \
    --chunked-prefill-size 2048 \
    --dtype bfloat16 \
    --mem-fraction-static 0.95 \
    --swa-full-tokens-ratio 0.2 \
    --skip-server-warmup \
    --log-level info \
    --max-running-requests 128' > /tmp/server_rank${rank}.log 2>&1 &
echo \"Server launched on rank ${rank}, PID=\$!\"
"
        ssh_worker "${rank}" "${launch_cmd}"
    done

    log "All server processes launched. Waiting for server to be ready..."
    wait_for_server "${MASTER_IP}"
}

wait_for_server() {
    local master_ip=$1
    local max_wait=900  # 15 minutes max
    local elapsed=0

    while [ $elapsed -lt $max_wait ]; do
        if ssh_worker 0 "curl -s http://localhost:${SERVER_PORT}/v1/models" 2>/dev/null | grep -q "mimo_model"; then
            log "Server is ready! (took ${elapsed}s)"
            return 0
        fi
        sleep 15
        elapsed=$((elapsed + 15))
        if [ $((elapsed % 60)) -eq 0 ]; then
            log "  Still waiting... (${elapsed}s elapsed)"
        fi
    done

    log "ERROR: Server did not become ready within ${max_wait}s"
    return 1
}

# =============================================================================
# Step 4: Run performance benchmark
# =============================================================================
run_perf_benchmark() {
    log "Step 4: Running performance benchmark..."

    mkdir -p "${RESULTS_DIR}" 2>/dev/null || true

    for bs in 64 128 200; do
        local bench_cmd="
export PATH=\"\$HOME/.local/bin:\$PATH\"
source ${VENV_DIR}/bin/activate
python3 -m sgl_jax.bench_serving \
    --backend sgl-jax --port ${SERVER_PORT} \
    --model ${MODEL_DIR} \
    --dataset-name random \
    --random-input-len 16384 --random-output-len 1024 \
    --num-prompts ${bs} --max-concurrency ${bs} \
    --output-file /tmp/bench_bs${bs}.json
"
        log "  Running: BS=${bs}, input=16384, output=1024"
        ssh_worker 0 "${bench_cmd}"
    done

    log "  Fetching results..."
    for bs in 64 128 200; do
        ssh_worker 0 "cat /tmp/bench_bs${bs}.json" 2>/dev/null > "${RESULTS_DIR}/perf_bs${bs}.json" 2>/dev/null || true
    done

    log "Performance benchmark complete."
}

# =============================================================================
# Step 5: Run accuracy benchmark (GSM8K)
# =============================================================================
run_accuracy_benchmark() {
    log "Step 5: Running accuracy benchmark (GSM8K)..."

    local eval_cmd="
export PATH=\"\$HOME/.local/bin:\$PATH\"
source ${VENV_DIR}/bin/activate
evalscope eval \
    --model ${MODEL_DIR} \
    --api-url http://localhost:${SERVER_PORT}/v1 \
    --api-key EMPTY \
    --eval-type service \
    --datasets gsm8k \
    --dataset-hub modelscope \
    --limit 200 \
    --work-dir /tmp/evalscope_output
"
    ssh_worker 0 "${eval_cmd}"

    log "  Fetching accuracy results..."
    ssh_worker 0 "find /tmp/evalscope_output -name 'gsm8k.json' -path '*/reports/*' | sort | tail -1 | xargs cat" \
        2>/dev/null > "${RESULTS_DIR}/accuracy_gsm8k.json" 2>/dev/null || true

    log "Accuracy benchmark complete."
}

# =============================================================================
# Step 6: Print summary
# =============================================================================
print_summary() {
    log "============================================="
    log "  MiMo-V2-Flash Benchmark Summary"
    log "============================================="
    log ""
    log "Cluster: ${CLUSTER_NAME} (${ZONE})"
    log "Accelerator: TPU v6e-32 (${NUM_CHIPS} chips, ${NUM_WORKERS} workers)"
    log "Model: MiMo-V2-Flash (FP8, 256 experts, top-8)"
    log "Config: TP=${NUM_CHIPS}, EP=${NUM_CHIPS}, DP=1"
    log ""

    log "--- Performance Results (input=16384, output=1024) ---"
    for bs in 64 128 200; do
        local out_tput
        out_tput=$(ssh_worker 0 "python3 -c \"import json; d=json.load(open('/tmp/bench_bs${bs}.json')); print(f'{d[\\\"output_throughput\\\"]:.1f} out tok/s, {d[\\\"median_itl_ms\\\"]:.2f} ms ITL')\"" 2>/dev/null || echo "N/A")
        log "  BS=${bs}: ${out_tput}"
    done

    log ""

    if [ -f "${RESULTS_DIR}/accuracy_gsm8k.json" ]; then
        log "--- Accuracy Results (GSM8K) ---"
        local score
        score=$(python3 -c "import json; print(json.load(open('${RESULTS_DIR}/accuracy_gsm8k.json'))['score'])" 2>/dev/null || echo "N/A")
        log "GSM8K Accuracy: ${score}"
    fi

    log ""
    log "Results saved to: ${RESULTS_DIR}/"
    log "============================================="
}

# =============================================================================
# Main
# =============================================================================
main() {
    log "Starting MiMo-V2-Flash benchmark on ${CLUSTER_NAME}"
    log "GCS model: gs://${GCS_BUCKET}/${GCS_MODEL_DIR}"
    log ""

    install_dependencies
    setup_model
    launch_server
    run_perf_benchmark
    run_accuracy_benchmark
    print_summary

    log "All benchmarks complete!"
}

main "$@"
