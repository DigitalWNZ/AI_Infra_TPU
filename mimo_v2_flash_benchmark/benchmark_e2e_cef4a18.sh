#!/bin/bash
# =============================================================================
# End-to-End MiMo-V2-Flash Benchmark on TPU v6e-32
# Using commit cef4a18 with Data Parallelism (dp=4)
# =============================================================================
#
# This script runs the full benchmark using the orphaned commit cef4a18 which
# contains PR #213's DP implementation. It sweeps dp=1,2,4,8 and runs the
# Xiaomi-matched long-context benchmark with the best config (dp=4).
#
# Usage:
#   bash benchmark_e2e_cef4a18.sh <CLUSTER_NAME> <ZONE> <GCS_BUCKET> <GCS_MODEL_DIR>
#
# Example:
#   bash benchmark_e2e_cef4a18.sh a6e-wangez us-east5-b 0-wangez jingnw-mimo-v2-flash-us-east5/hf-model
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - TPU v6e-32 cluster already created and READY
#   - Model weights uploaded to GCS bucket (313 GB, 145 split safetensors files)
# =============================================================================

set -euo pipefail

CLUSTER_NAME="${1:?Usage: $0 <cluster_name> <zone> <gcs_bucket> <gcs_model_dir>}"
ZONE="${2:?Usage: $0 <cluster_name> <zone> <gcs_bucket> <gcs_model_dir>}"
GCS_BUCKET="${3:?Usage: $0 <cluster_name> <zone> <gcs_bucket> <gcs_model_dir>}"
GCS_MODEL_DIR="${4:?Usage: $0 <cluster_name> <zone> <gcs_bucket> <gcs_model_dir>}"

COMMIT="cef4a18"
NUM_WORKERS=8
NUM_CHIPS=32
SERVER_PORT=30271
DIST_PORT=30000
MODEL_DIR="/tmp/mimo_model"
GCS_MOUNT="/tmp/gcs_model"
SGLANG_DIR="/sglang-jax"
VENV_DIR="${SGLANG_DIR}/.venv"
RESULTS_DIR="/tmp/benchmark_results"

# Best config from DP sweep
BEST_DP=4
TP_SIZE=32
EP_SIZE=32

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

get_master_ip() {
    gcloud compute tpus tpu-vm describe "${CLUSTER_NAME}" --zone="${ZONE}" \
        --format='value(networkEndpoints[0].ipAddress)' 2>/dev/null
}

# =============================================================================
# Step 1: Install dependencies and checkout cef4a18
# =============================================================================
install_dependencies() {
    log "Step 1: Installing dependencies and checking out commit ${COMMIT}..."

    local install_script="
set -ex

# Install uv package manager
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH=\"\$HOME/.local/bin:\$PATH\"

# Clone sglang-jax if not present
if [ ! -d ${SGLANG_DIR} ]; then
    sudo git clone https://github.com/sgl-project/sglang-jax.git ${SGLANG_DIR}
    sudo chown -R \$(whoami) ${SGLANG_DIR}
fi

# Fetch and checkout orphaned commit cef4a18 (contains PR #213 DP implementation)
cd ${SGLANG_DIR}
git fetch origin ${COMMIT}
git checkout ${COMMIT}

# Create Python 3.12 venv and install (non-editable — cef4a18 lacks build_editable hook)
uv venv --python 3.12
source .venv/bin/activate
uv pip install \"python[tpu]\"

echo \"=== Install complete on \$(hostname) — commit \$(git rev-parse --short HEAD) ===\"
"
    ssh_all_workers "${install_script}"
    log "Dependencies installed, commit ${COMMIT} checked out on all workers."
}

# =============================================================================
# Step 2: Mount model from GCS and symlink all 145 safetensors
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

# Symlink ALL safetensors from GCS mount (cef4a18 needs all 145 split files)
for f in ${GCS_MOUNT}/*.safetensors; do
    if [ -f \"\${f}\" ]; then
        ln -sf \"\${f}\" ${MODEL_DIR}/\$(basename \"\${f}\")
    fi
done

# Also symlink the model index
if [ -f ${GCS_MOUNT}/model.safetensors.index.json ]; then
    ln -sf ${GCS_MOUNT}/model.safetensors.index.json ${MODEL_DIR}/
fi

echo \"Model dir ready: \$(ls ${MODEL_DIR}/*.safetensors 2>/dev/null | wc -l) safetensors files\"
echo \"=== Model setup complete on \$(hostname) ===\"
"
    ssh_all_workers "${setup_script}"
    log "Model directory prepared on all workers."
}

# =============================================================================
# Step 3: Kill any existing server processes
# =============================================================================
kill_servers() {
    log "Killing existing server processes..."
    ssh_all_workers "pkill -9 -f 'sgl_jax.launch_server' 2>/dev/null || true"
    sleep 10
    ssh_all_workers "pkill -9 -f python 2>/dev/null || true"
    sleep 5
    log "Server processes cleaned up."
}

# =============================================================================
# Step 4: Launch server with given DP config
# =============================================================================
launch_server() {
    local dp=$1
    local master_ip
    master_ip=$(get_master_ip)

    log "Launching server with dp=${dp}, tp=${TP_SIZE}, ep=${EP_SIZE}..."
    log "  Master IP: ${master_ip}"
    log "  Mesh: (data=${dp}, tensor=$((TP_SIZE / dp)))"

    for rank in $(seq 0 $((NUM_WORKERS - 1))); do
        local launch_cmd="
export PATH=\"\$HOME/.local/bin:\$PATH\"
source ${VENV_DIR}/bin/activate
nohup bash -c 'JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -u -m sgl_jax.launch_server \
    --model-path ${MODEL_DIR} \
    --trust-remote-code \
    --tp-size ${TP_SIZE} --dp-size ${dp} --ep-size ${EP_SIZE} \
    --moe-backend epmoe \
    --nnodes ${NUM_WORKERS} --node-rank ${rank} \
    --dist-init-addr ${master_ip}:${DIST_PORT} \
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

    wait_for_server
}

wait_for_server() {
    local max_wait=900
    local elapsed=0

    log "  Waiting for server to be ready (up to ${max_wait}s)..."
    while [ $elapsed -lt $max_wait ]; do
        if ssh_worker 0 "curl -s http://localhost:${SERVER_PORT}/v1/models" 2>/dev/null | grep -q "model"; then
            log "  Server ready! (took ${elapsed}s)"
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
# Step 5: Run benchmark with given parameters
# =============================================================================
run_benchmark() {
    local label=$1
    local input_len=$2
    local output_len=$3
    local num_prompts=$4
    local max_concurrency=$5
    local extra_args=${6:-""}
    local output_file="/tmp/bench_${label}.json"

    log "  Running benchmark [${label}]: ${num_prompts} prompts, input=${input_len}, output=${output_len}, concurrency=${max_concurrency}"

    local bench_cmd="
export PATH=\"\$HOME/.local/bin:\$PATH\"
source ${VENV_DIR}/bin/activate
python3 -m sgl_jax.bench_serving \
    --backend sgl-jax --port ${SERVER_PORT} \
    --model ${MODEL_DIR} \
    --dataset-name random \
    --random-input-len ${input_len} --random-output-len ${output_len} \
    --num-prompts ${num_prompts} --max-concurrency ${max_concurrency} \
    ${extra_args} \
    --output-file ${output_file}
"
    ssh_worker 0 "${bench_cmd}"

    # Print key results
    ssh_worker 0 "
source ${VENV_DIR}/bin/activate
python3 -c \"
import json
with open('${output_file}') as f:
    d = json.load(f)
print(f'  Output throughput: {d.get(\"output_throughput\", \"N/A\")} tok/s')
print(f'  Total throughput:  {d.get(\"total_throughput\", \"N/A\")} tok/s')
\"" 2>/dev/null || true
}

# =============================================================================
# Step 6: DP Sweep (short context)
# =============================================================================
run_dp_sweep() {
    log "Step 6: Running DP sweep (short context: input=1024, output=512)..."

    for dp in 1 2 4 8; do
        log "--- DP=${dp} ---"
        kill_servers
        launch_server "${dp}"

        for bs in 64 128 200; do
            run_benchmark "dp${dp}_bs${bs}" 1024 512 "${bs}" "${bs}"
        done
    done

    log "DP sweep complete."
}

# =============================================================================
# Step 7: Long-context DP sweep (Xiaomi-matched parameters)
# =============================================================================
run_long_context() {
    log "Step 7: Running long-context DP sweep (input=16384, output=1024, Xiaomi-matched)..."

    for dp in 1 2 4 8; do
        log "--- Long-context DP=${dp} ---"
        kill_servers
        launch_server "${dp}"

        for bs in 64 128 200; do
            run_benchmark "dp${dp}_long_bs${bs}" 16384 1024 256 "${bs}" \
                "--request-rate 100 --random-range-ratio 1.0 --flush-cache"
        done
    done

    log "Long-context DP sweep complete."
}

# =============================================================================
# Step 8: Print summary
# =============================================================================
print_summary() {
    log "============================================="
    log "  MiMo-V2-Flash Benchmark Summary (cef4a18)"
    log "============================================="
    log ""
    log "Cluster: ${CLUSTER_NAME} (${ZONE})"
    log "Accelerator: TPU v6e-32 (${NUM_CHIPS} chips, ${NUM_WORKERS} workers)"
    log "Model: MiMo-V2-Flash (FP8, 256 experts, top-8)"
    log "Commit: ${COMMIT} (orphaned, PR #213 DP)"
    log "Config: TP=${TP_SIZE}, EP=${EP_SIZE}, best DP=${BEST_DP}"
    log ""

    log "--- Short-Context DP Sweep (input=1024, output=512) ---"
    for dp in 1 2 4 8; do
        for bs in 64 128 200; do
            local file="/tmp/bench_dp${dp}_bs${bs}.json"
            local out_tput
            out_tput=$(ssh_worker 0 "python3 -c \"import json; d=json.load(open('${file}')); print(f'{d[\"output_throughput\"]:.1f} out tok/s, {d[\"max_output_tokens_per_s\"]:.0f} peak')\"" 2>/dev/null || echo "N/A")
            log "  dp=${dp}, BS=${bs}: ${out_tput}"
        done
    done

    log ""
    log "--- Long-Context DP Sweep (input=16384, output=1024, Xiaomi-matched) ---"
    for dp in 1 2 4 8; do
        for bs in 64 128 200; do
            local file="/tmp/bench_dp${dp}_long_bs${bs}.json"
            local out_tput
            out_tput=$(ssh_worker 0 "python3 -c \"import json; d=json.load(open('${file}')); print(f'{d[\"output_throughput\"]:.1f} out tok/s, {d[\"max_output_tokens_per_s\"]:.0f} peak')\"" 2>/dev/null || echo "N/A")
            log "  dp=${dp}, BS=${bs}: ${out_tput}"
        done
    done

    log ""
    log "Results JSON files on worker-0: /tmp/bench_*.json"
    log "============================================="
}

# =============================================================================
# Main
# =============================================================================
main() {
    log "Starting MiMo-V2-Flash benchmark (commit ${COMMIT}) on ${CLUSTER_NAME}"
    log "GCS model: gs://${GCS_BUCKET}/${GCS_MODEL_DIR}"
    log ""

    install_dependencies
    setup_model
    run_dp_sweep
    run_long_context
    print_summary

    log "All benchmarks complete!"
}

main "$@"
