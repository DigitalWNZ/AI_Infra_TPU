# MiMo-V2-Flash Benchmark Operation Log — Commit cef4a18 (DP-Enabled)
## Cluster: a6e-wangez (v6e-32, topo 4x8, us-east5-b)
## Date: 2026-04-22

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

### 2. Background: Why cef4a18?

The Xiaomi benchmark report ("基于 SGLangJax 的 mimo_v2_flash 报告", 2026-04-10) used `--dp-size 4` on a v6e-16 cluster. Neither `origin/main` nor `epic/mimo-v2-flash` support dp>1 — the engine code path is:

```python
if server_args.dp_size == 1:
    scheduler_pipe_readers = []
    # ... launch scheduler ...
else:
    pass    # DP > 1 NOT IMPLEMENTED → UnboundLocalError
```

Investigation revealed the report used commit `cef4a181508ef22b452a38fe4210478e2e3672b1`, which contains **PR #213 ("data parallelism")** — an 8,652-line merge across 67 files implementing single-controller DP inside the scheduler via JAX SPMD. This commit is orphaned (force-pushed away from `epic/mimo-v2-flash`).

**Key DP design at cef4a18:**
- Mesh: `(data=dp_size, tensor=tp_size/dp_size)`
- Single scheduler process with per-DP request queues, KV cache allocators, and radix caches
- Inputs reordered so each DP rank's tokens are contiguous, sharded along `"data"` axis
- EPMoE creates its own separate `(expert, tensor)` mesh for expert routing
- Design doc at `docs/design/data_parallelism.md` in the commit

---

### 3. Software Setup

**Commit:** `cef4a181508ef22b452a38fe4210478e2e3672b1` (orphaned)
**Previous environment:** Python 3.12 venv at `/sglang-jax/.venv` with uv (from `origin/main` benchmark on 2026-04-21)

**Steps:**

1. Fetched and checked out the orphaned commit on all workers:
   ```bash
   gcloud compute tpus tpu-vm ssh a6e-wangez --zone=us-east5-b --worker=all \
     --command="cd /sglang-jax && git fetch origin cef4a18 && git checkout cef4a18"
   ```

2. Reinstalled sglang-jax (non-editable — cef4a18's `pyproject.toml` lacks `build_editable` hook):
   ```bash
   gcloud compute tpus tpu-vm ssh a6e-wangez --zone=us-east5-b --worker=all \
     --command="cd /sglang-jax && source .venv/bin/activate && uv pip install 'python[tpu]'"
   ```
   The old editable install's `.pth` file still pointed to `/sglang-jax/python/sgl_jax/` source tree, so the cef4a18 code was used correctly.

**Key dependencies (unchanged):**
- jax==0.8.1, jaxlib==0.8.1
- libtpu==0.0.30
- flax==0.12.4
- transformers==4.57.6

---

### 4. Model Setup

**Model:** MiMo-V2-Flash (Xiaomi, FP8 quantized, 256 routed experts, top-8)
**Total size:** 313 GB (145 safetensors files, split format)
**Disk constraint:** 97 GB boot disk per worker

**Key difference from `main` branch:** The cef4a18 weight loader expects all 145 split safetensors files. The `main` branch needed only 41 merged files.

**GCSFuse mount (already set up from previous benchmark):**
```bash
gcsfuse --only-dir jingnw-mimo-v2-flash-us-east5/hf-model 0-wangez /tmp/gcs_model
```

**Symlinked missing safetensors from GCS:**
```bash
gcloud compute tpus tpu-vm ssh a6e-wangez --zone=us-east5-b --worker=all \
  --command="for f in /tmp/gcs_model/*.safetensors; do
    base=\$(basename \$f)
    [ ! -e /models/MiMo-V2-Flash/\$base ] && ln -s \$f /models/MiMo-V2-Flash/\$base
  done"
```

This created 104 symlinks, giving `/models/MiMo-V2-Flash/` all 145 safetensors plus config/tokenizer files.

---

### 5. Errors and Fixes

| # | Error | Root Cause | Fix |
|---|-------|-----------|-----|
| 1 | Editable install fails (`build_editable` hook missing) | `pyproject.toml` at cef4a18 lacks hatchling build hook | Non-editable `uv pip install "python[tpu]"`; old .pth file still works |
| 2 | MoE expert weight not found during loading | cef4a18 needs all 145 split safetensors; only 41 present locally | Symlinked 104 files from GCSFuse mount |
| 3 | Server processes "Killed" silently | Race condition: pkill not complete before new server launch | Added sleep between kill and launch; verified clean state |
| 4 | `ValueError: devices 32 ≠ mesh_shape (1, 16)` with tp=16 | cef4a18 mesh `(data, tensor)` product must equal total chips (32). tp=16 creates 16-device mesh on 32 chips. | **tp must be 32** on this cluster. Cannot use tp=16. |
| 5 | 4-worker launch hangs indefinitely | JAX TPU pods require ALL hosts to call `jax.distributed.initialize()` | Must use all 8 workers; cannot use a subset of the pod |
| 6 | JAX `DEADLINE_EXCEEDED` on distributed init retry | Leftover JAX state from failed attempts blocks new init | `pkill -9 -f python` on all workers, wait, then retry |

**Critical constraint:** On v6e-32, `tp_size` MUST be 32. The JAX mesh spans all 32 chips as `(data=dp_size, tensor=32/dp_size)`. The `ep_size` is independent — EPMoE creates its own separate mesh.

This means we **cannot replicate the Xiaomi report's tp=16, ep=16 config on v6e-32** without code changes. Our sweep uses tp=32, ep=32, varying only dp.

---

### 6. DP Sweep (Short Context)

**Methodology:** Tested dp=1, 2, 4, 8 with tp=32, ep=32, moe_backend=epmoe.
Workload: random data, input=1024, output=512.
Batch sizes: 64, 128, 200.

For each configuration:
1. Kill existing server processes on all workers
2. Launch new server with the target dp value
3. Wait for server ready (~9 min cold start, ~2 min warm)
4. Run benchmark from worker-0

**Launch command template:**
```bash
# On all workers:
source /sglang-jax/.venv/bin/activate
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -u -m sgl_jax.launch_server \
    --model-path /models/MiMo-V2-Flash --trust-remote-code \
    --tp-size 32 --dp-size $DP --ep-size 32 \
    --moe-backend epmoe --nnodes 8 --node-rank $RANK \
    --dist-init-addr 10.202.0.54:30000 \
    --host 0.0.0.0 --port 30271 --page-size 256 \
    --context-length 262144 --disable-radix-cache \
    --chunked-prefill-size 2048 --dtype bfloat16 \
    --mem-fraction-static 0.95 --swa-full-tokens-ratio 0.2 \
    --skip-server-warmup --log-level info --max-running-requests 128
```

**Benchmark command template:**
```bash
python3 -m sgl_jax.bench_serving \
    --backend sgl-jax --port 30271 \
    --model /models/MiMo-V2-Flash \
    --dataset-name random \
    --random-input-len 1024 --random-output-len 512 \
    --num-prompts $BS --max-concurrency $BS
```

**Results:**

| Config | Mesh (data, tensor) | BS=64 Out tok/s | BS=128 Out tok/s | BS=200 Out tok/s |
|--------|--------------------:|----------------:|-----------------:|-----------------:|
| dp=1   | (1, 32)             | 1,259           | 1,506            | 1,506            |
| dp=2   | (2, 16)             | 1,325           | 1,741            | 1,600            |
| **dp=4** | **(4, 8)**        | **1,315**       | **1,808**        | **1,384**        |
| dp=8   | (8, 4)              | 1,133           | 1,608            | 1,046            |

**Peak burst output throughput:**
| dp=1 | dp=2 | dp=4 | dp=8 |
|------|------|------|------|
| 2,667 | 3,239 | 3,704 | 4,025 |

**Best sustained throughput:** dp=4 at BS=128 → **1,808 output tok/s** (+20% over dp=1)

---

### 7. Long-Context DP Sweep

Ran long-context benchmarks with dp=2, 4, 8 using Xiaomi-matched parameters to find the best long-context config.

**Benchmark command template:**
```bash
python3 -m sgl_jax.bench_serving \
    --backend sgl-jax --port 30271 \
    --model /models/MiMo-V2-Flash \
    --dataset-name random \
    --random-input-len 16384 --random-output-len 1024 \
    --num-prompts 256 --max-concurrency $BS \
    --request-rate 100 --random-range-ratio 1.0 \
    --flush-cache
```

#### 7.1 Output Throughput (tok/s)

| Config | BS=64 | BS=128 | BS=200 |
|--------|------:|-------:|-------:|
| dp=2   | 594.4 | 579.1  | 631.6  |
| dp=4   | 635.9 | —      | —      |
| dp=8   | 606.0 | 633.0  | **652.5** |

#### 7.2 Peak Output Throughput (tok/s)

| Config | BS=64 | BS=128 | BS=200 |
|--------|------:|-------:|-------:|
| dp=2   | 1,920 | 2,952  | 3,072  |
| dp=4   | 3,840 | —      | —      |
| dp=8   | 3,264 | 4,191  | **4,224** |

#### 7.3 Latency (Median ITL, ms)

| Config | BS=64 | BS=128 | BS=200 |
|--------|------:|-------:|-------:|
| dp=2   | 34.25 | 42.84  | 42.89  |
| dp=8   | **28.14** | **30.86** | **30.90** |

#### 7.4 Full Metrics

| Config | Out tok/s | Peak tok/s | Input tok/s | Total tok/s | Med ITL (ms) | P99 ITL (ms) | Med TTFT (ms) | Mean E2E (ms) |
|--------|----------:|-----------:|------------:|------------:|-------------:|-------------:|--------------:|---------------:|
| dp=2, BS=64 | 594.4 | 1,920 | 9,510.6 | 10,105.0 | 34.25 | 48.40 | 36,161 | 110,140 |
| dp=2, BS=128 | 579.1 | 2,952 | 9,266.3 | 9,845.5 | 42.84 | 69.37 | 76,314 | 204,779 |
| dp=2, BS=200 | 631.6 | 3,072 | 10,106.3 | 10,737.9 | 42.89 | 70.10 | 119,158 | 257,923 |
| dp=4, BS=64 | 635.9 | 3,840 | 10,173.6 | 10,809.5 | — | — | — | — |
| dp=8, BS=64 | 606.0 | 3,264 | 9,696.1 | 10,302.1 | 28.14 | 30.15 | 35,369 | 108,033 |
| dp=8, BS=128 | 633.0 | 4,191 | 10,127.8 | 10,760.8 | 30.86 | 37.79 | 77,024 | 191,989 |
| **dp=8, BS=200** | **652.5** | **4,224** | **10,439.6** | **11,092.1** | **30.90** | **38.32** | **114,703** | **243,134** |

#### 7.5 Analysis

- **dp=8 BS=200 achieves the highest long-context throughput: 652.5 tok/s** — matching Xiaomi's 654.9 nearly exactly (99.6%)
- **dp=8 has the best ITL** across all batch sizes (28–31 ms median) because each DP rank handles a small per-rank batch, minimizing decode latency
- **dp=2** is weaker for long context; ITL degrades significantly at BS≥128 (34→43 ms)
- Unlike short-context where dp=8 underperformed, long-context with 256 prompts gives each dp=8 rank enough work (32 prompts/rank at BS=200)
- **Peak burst throughput scales with DP** as in short context: dp=8 hits 4,224 tok/s (65% higher than Xiaomi's 2,560)

---

### 8. Comparison with Xiaomi Report

Best long-context result (dp=8 BS=200) vs Xiaomi report. Both: 256 prompts, input=16384, output=1024, rr=100, range-ratio=1.0, flush-cache.

| Metric | Xiaomi (16 chips, v6e-16) | Ours (32 chips, v6e-32) | Ratio |
|--------|------------------------:|------------------------:|------:|
| Output throughput (tok/s) | 654.9 | 652.5 | **99.6%** |
| Peak output throughput (tok/s) | 2,560 | 4,224 | **165%** |
| Input throughput (tok/s) | 10,477.8 | 10,439.6 | 99.6% |
| Total throughput (tok/s) | 11,132.7 | 11,092.1 | 99.6% |
| Per-chip output (tok/s/chip) | 40.9 | 20.4 | 50% |

**Key insight:** With dp=8 and BS=200, we match the Xiaomi report's throughput within 0.4%. Peak burst throughput is 65% higher (4,224 vs 2,560).

**Why per-chip efficiency is 2x lower (20.4 vs 40.9 tok/s/chip):**
1. TP=32 all-reduce spans 8 nodes (4x8 mesh) vs TP=16 across 4 nodes (4x4 mesh) — communication latency grows super-linearly
2. MoE all-to-all with EP=32 has 2x the routing overhead of EP=16
3. Decode is memory-bandwidth-bound: doubling chips halves per-chip work but communication cost stays constant or grows
4. Xiaomi report uses tp=16 on 16 chips where every chip is 1-2 ICI hops away; our 4x8 topology has longer paths

---

### 9. End-to-End Script

Generated: `benchmark_e2e_cef4a18.sh`

**Usage:**
```bash
bash benchmark_e2e_cef4a18.sh <CLUSTER_NAME> <ZONE> <GCS_BUCKET> <GCS_MODEL_DIR>
# Example:
bash benchmark_e2e_cef4a18.sh a6e-wangez us-east5-b 0-wangez jingnw-mimo-v2-flash-us-east5/hf-model
```

**What it does:**
1. Installs uv, Python 3.12 venv, fetches commit `cef4a18`, installs sglang-jax
2. Installs GCSFuse, mounts model from GCS, symlinks all 145 safetensors files
3. Launches server on all 8 workers with dp=4, tp=32, ep=32
4. Waits for server readiness (up to 15 min)
5. Runs DP sweep: dp=1,2,4,8 x BS=64,128 (short context)
6. Runs long-context benchmark with dp=4 (Xiaomi-matched parameters)
7. Prints summary with all results
