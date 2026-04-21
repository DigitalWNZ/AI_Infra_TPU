# MiMo-V2-Flash Benchmark Operation Log
## Cluster: a6e-wangez (v6e-32, topo 4x8, us-east5-b)
## Date: 2026-04-21

---

### 1. Cluster Info

- **Name:** a6e-wangez
- **Zone:** us-east5-b
- **Accelerator:** v6e-32
- **Topology:** 4x8 (32 chips)
- **Workers:** 8 (w-0 to w-7), each with 4 chips
- **Runtime:** v2-alpha-tpuv6e
- **Status:** READY / HEALTHY

**Worker IPs (internal):**
| Worker | Internal IP   |
|--------|---------------|
| w-0    | 10.202.0.54   |
| w-1    | 10.202.0.68   |
| w-2    | 10.202.0.69   |
| w-3    | 10.202.0.58   |
| w-4    | 10.202.0.56   |
| w-5    | 10.202.0.70   |
| w-6    | 10.202.0.71   |
| w-7    | 10.202.0.55   |

---

### 2. Software Installation

**Branch:** `origin/main` (commit `32c87848`)
**Previous attempt:** Started on `epic/mimo-v2-flash` branch but hit multiple compatibility issues. Switched to `main` which has proper MiMo-V2-Flash support (commits `1edfab11`, `28244725`).

**Steps:**
1. Installed `uv` package manager on all workers
2. Created Python 3.12 venv at `/sglang-jax/.venv` (required >=3.12)
3. Installed sglang-jax with TPU support: `uv pip install -e "python[tpu]"`
4. Installed evalscope for accuracy testing: `uv pip install evalscope==0.17.1`

**Key dependencies:**
- jax==0.8.1, jaxlib==0.8.1
- libtpu==0.0.30
- flax==0.12.4
- transformers==4.57.6

---

### 3. Model Setup

**Model:** MiMo-V2-Flash (Xiaomi, FP8 quantized, 256 routed experts, top-8)
**Total size:** 313 GB (145 safetensors files)
**Disk constraint:** 97 GB boot disk per worker — model doesn't fit locally

**Solution:** GCSFuse mount from `gs://0-wangez/jingnw-mimo-v2-flash-us-east5/hf-model/`

**Setup on each worker:**
1. Installed GCSFuse: `apt-get install gcsfuse`
2. Mounted GCS bucket: `gcsfuse --only-dir jingnw-mimo-v2-flash-us-east5/hf-model 0-wangez /tmp/gcs_model`
3. Created hybrid model directory `/tmp/mimo_model`:
   - Config/tokenizer files copied from local partial download (`/models/MiMo-V2-Flash/`)
   - Safetensors files symlinked from GCSFuse mount (`/tmp/gcs_model/`)

---

### 4. Errors and Fixes (on `epic/mimo-v2-flash` branch)

| # | Error | Root Cause | Fix |
|---|-------|-----------|-----|
| 1 | `jax[tpu]==0.8.1` not found with pip | System pip can't resolve jax 0.8.1 | Used `uv` which resolves from TPU-specific index |
| 2 | Python 3.10 too old | sglang-jax requires >=3.12 | Used `uv venv --python 3.12` |
| 3 | Missing safetensors files | Only 41/145 files downloaded (disk too small for 313GB model) | Mounted full model from GCS via GCSFuse |
| 4 | MiMoV2FlashForCausalLM not in model registry | File removed from HEAD of epic/mimo-v2-flash | Switched to `main` branch which has it |
| 5 | AutoTokenizer ValueError | Missing tokenizer files locally | Downloaded tokenizer files from HuggingFace |
| 6 | FP8 dequant `weight_scale ndim=2` | 2D block quantization scale not handled | Patched model code (later fixed by main branch) |
| 7 | `assert not self.is_hybrid` | Paged allocation doesn't support hybrid SWA on old branch | Fixed in main branch (commit `28244725`) |
| 8 | FlashAttention missing `attention_sink` kwarg | Model code from old commit, incompatible with HEAD FlashAttention | Fixed in main branch |
| 9 | ShardingTypeError on K reshape | TP=32 with 8 KV heads, old code can't handle | Fixed in main branch |

**Resolution:** Switched all workers from `epic/mimo-v2-flash` to `origin/main` branch, which has proper MiMo-V2-Flash support with all compatibility fixes.

---

### 5. Server Launch

**Command (run on each worker with different `--node-rank`):**
```bash
source /sglang-jax/.venv/bin/activate
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -u -m sgl_jax.launch_server \
    --model-path /tmp/mimo_model \
    --trust-remote-code \
    --tp-size 32 --dp-size 1 --ep-size 32 \
    --moe-backend epmoe \
    --nnodes 8 --node-rank $RANK \
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
```

**Startup timeline:**
- GCSFuse cache warmup: ~4 min (291.6 GB read from GCS)
- Weight loading (regular + MoE): ~1 min
- KV cache allocation: ~1 sec
- JIT precompilation (8 batch sizes): ~3.5 min
- **Total startup: ~9 min**

**Server config summary:**
- TP=32, EP=32, DP=1 (all 32 chips for tensor/expert parallelism)
- Page size: 256
- Max tokens: 446,313 per device, 22 GB available device memory
- KV cache: 227.6 GB SWA + 262.6 GB full attention (fused across all devices)
- FP8 quantization: static, auto-detected

---

### 6. Performance Benchmark Results

**Benchmark command:**
```bash
python3 -m sgl_jax.bench_serving \
    --backend sgl-jax --port 30271 \
    --model /tmp/mimo_model \
    --dataset-name random \
    --random-input-len 16384 --random-output-len 1024 \
    --num-prompts 64 --max-concurrency 64
```

**Results (v6e-32, BS=64, input=16384, output=1024):**

| Metric | Value |
|--------|-------|
| Request throughput | 1.14 req/s |
| Input token throughput | 8,760.9 tok/s |
| **Output token throughput** | **571.9 tok/s** |
| Total token throughput | 9,332.8 tok/s |
| Mean E2E latency | 48,325 ms |
| Median TTFT | 19,037 ms |
| **Median ITL** | **23.45 ms** |
| P99 ITL | 28.98 ms |
| Benchmark duration | 56.2 s |
| Successful requests | 64/64 |

**Comparison with report (v6e-256, tp=16, ep=16):**
- Report output throughput: ~2,700 tok/s (at BS=64, input=16384, output=1024)
- Our output throughput: 571.9 tok/s on v6e-32
- Ratio: ~21%, roughly proportional to chip count (32/256 = 12.5%), though slightly better per-chip

---

### 7. Quick Sanity Check

```
curl http://localhost:30271/v1/completions -d '{"model":"/tmp/mimo_model","prompt":"Hello, how are you?","max_tokens":32}'
```
Response: `"I'm fine, thank you. How are you? I'm glad to hear that. How can I help you today?"`

---

### 8. Accuracy Benchmark Results (GSM8K)

**Benchmark command:**
```bash
source /sglang-jax/.venv/bin/activate
evalscope eval \
    --model /tmp/mimo_model \
    --api-url http://localhost:30271/v1 \
    --api-key EMPTY \
    --eval-type service \
    --datasets gsm8k \
    --dataset-hub modelscope \
    --limit 200 \
    --work-dir /tmp/evalscope_output
```

**Results:**

| Metric | Value |
|--------|-------|
| Dataset | GSM8K (Grade School Math 8K) |
| Subset | main |
| Samples evaluated | 200 |
| **Accuracy (AverageAccuracy)** | **97.5%** |

**Notes:**
- Model correctly answered 195/200 grade school math problems
- Responses show clear step-by-step reasoning with correct final answers
- This confirms the model weights are loaded correctly and FP8 quantization preserves accuracy
- The reported accuracy for MiMo-V2-Flash on GSM8K is 95.8% (full test set); our 97.5% on 200 samples is consistent with that

---

### 9. End-to-End Script

Generated: `benchmark_e2e.sh`

**Usage:**
```bash
bash benchmark_e2e.sh <CLUSTER_NAME> <ZONE> <GCS_BUCKET> <GCS_MODEL_DIR>
# Example:
bash benchmark_e2e.sh a6e-wangez us-east5-b 0-wangez jingnw-mimo-v2-flash-us-east5/hf-model
```

**What it does (in order):**
1. Installs uv, Python 3.12 venv, sglang-jax (from `origin/main`), and evalscope on all workers
2. Installs GCSFuse, mounts model from GCS, creates hybrid model directory
3. Launches SGLangJax server on all 8 workers with TP=32, EP=32
4. Waits for server readiness (up to 15 min for GCSFuse warmup + JIT compilation)
5. Runs performance benchmark: BS=64, input=16384, output=1024
6. Runs accuracy benchmark: GSM8K, 200 samples via evalscope
7. Prints summary with all results
