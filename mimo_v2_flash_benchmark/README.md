# MiMo-V2-Flash Benchmark on TPU v6e

This directory contains the complete artifacts from benchmarking Xiaomi's **MiMo-V2-Flash** (256-expert MoE, FP8) on **TPU v6e** clusters using **SGLangJax**.

---

## File Overview

### Commit cef4a18 (DP-Enabled, recommended)

| File | Purpose |
|------|---------|
| `benchmark_report_cef4a18.md` | Performance report with DP sweep, long-context results, and Xiaomi comparison |
| `operation_log_cef4a18.md` | Operation log for cef4a18 branch: setup, errors, DP sweep, benchmarks |
| `benchmark_e2e_cef4a18.sh` | E2E script for **v6e-32** (32 chips, 8 workers) with dp sweep |
| `benchmark_e2e_cef4a18_v6e16.sh` | E2E script for **v6e-16** (16 chips, 4 workers) with dp sweep — matches Xiaomi's exact HW config |

### origin/main branch (dp=1 only)

| File | Purpose |
|------|---------|
| `benchmark_report.md` | Benchmark report on main branch (dp=1) with Xiaomi comparison |
| `operation_log.md` | Operation log for main branch: setup, errors, benchmarks |
| `benchmark_e2e.sh` | E2E script for **v6e-32** (32 chips, 8 workers) using origin/main with dp=1 |
| `benchmark_e2e_v6e16.sh` | E2E script for **v6e-16** (16 chips, 4 workers) using origin/main with dp=1 |
| `install_v2.sh` | Per-worker dependency installation (used during the benchmark) |
| `install.sh` | Initial install attempt using pip (superseded by `install_v2.sh`) |
| `launch_server.sh` | Per-worker server launch command |

---

## How the Files Relate

```
Narrative flow (read in this order):
                                        
  operation_log.md          What actually happened, step by step
        |                   (errors, fixes, commands, raw output)
        v
  benchmark_report.md       Polished results + analysis + comparison
                            to Xiaomi's v6e-64 reference report


Execution flow (run in this order):

  install.sh                v1 install attempt (pip-based, hit dependency
        |                   issues with jax 0.8.1 and Python 3.10)
        v
  install_v2.sh             v2 install (uses uv + Python 3.12 venv,
        |                   resolves all dependency issues)
        v
  launch_server.sh          Launches SGLangJax server on a single worker
        |                   (run once per worker with different --node-rank)
        v
  benchmark_e2e.sh          All-in-one script that wraps install, model
                            setup, server launch, and both benchmarks
```

---

## File Details

### `operation_log.md`
The raw operational record written during the benchmark session. Documents:
- Cluster info (worker IPs, topology, runtime version)
- Software installation steps and dependency versions
- Model setup (GCSFuse mount, hybrid local/GCS directory)
- Every error encountered and how it was resolved (9 errors total)
- Server launch command and startup timeline
- Performance benchmark results (BS=32/64/128)
- Accuracy benchmark results (GSM8K)

**Use this to** understand what went wrong, what was tried, and how issues were resolved. Essential for debugging if reproducing on a different cluster.

### `benchmark_report.md`
The formal benchmark report. Contains:
- Test environment and configuration details
- Performance results across multiple batch sizes and input lengths
- Accuracy results (GSM8K)
- Side-by-side comparison with Xiaomi's reference report on v6e-64
- Analysis of why our per-chip efficiency is 1.75x higher
- Improvement recommendations (parallelism, kernel optimizations, infrastructure)

**Use this to** understand the results and plan next steps.

### `benchmark_e2e.sh`
A parameterized script that automates the entire benchmark pipeline. Takes cluster name, zone, GCS bucket, and model directory as arguments. Runs:
1. Dependency installation on all workers
2. GCSFuse model mount and hybrid directory setup
3. Multi-worker server launch with distributed init
4. Performance benchmark (random data, throughput/latency)
5. Accuracy benchmark (GSM8K via evalscope)
6. Summary printout

**Use this to** reproduce the benchmark on a new cluster:
```bash
bash benchmark_e2e.sh <CLUSTER_NAME> <ZONE> <GCS_BUCKET> <GCS_MODEL_DIR>
```

### `install_v2.sh`
The working installation script deployed to all 8 workers. Uses `uv` package manager with a Python 3.12 virtual environment at `/sglang-jax/.venv`. Installs sglang-jax from `origin/main` branch and evalscope for accuracy testing.

**Supersedes** `install.sh` which used system pip and failed because pip couldn't resolve `jax==0.8.1` from the TPU-specific package index, and the system Python 3.10 was too old for sglang-jax.

### `install.sh`
The initial installation attempt using pip. Kept for reference. It clones sglang-jax from `epic/mimo-v2-flash` branch and installs via pip, but this approach hit multiple issues:
- `jax[tpu]==0.8.1` not resolvable with pip
- Python 3.10 incompatible with sglang-jax (requires >=3.12)
- The `epic/mimo-v2-flash` branch had code compatibility issues

### `launch_server.sh`
The server launch command for a single worker. Accepts `RANK` as an environment variable or first argument. Key parameters:
- `--tp-size 32 --ep-size 32 --dp-size 1` (all 32 chips for one serving group)
- `--nnodes 8` (8 workers with 4 chips each)
- `--moe-backend epmoe` (expert-parallel MoE)
- `--context-length 262144` (256K context window)

Run on each worker with a different rank:
```bash
RANK=0 bash launch_server.sh   # on worker 0
RANK=1 bash launch_server.sh   # on worker 1
# ... etc
```

---

## Prerequisites

- GCP project with a TPU v6e cluster already provisioned:
  - **v6e-16** (16 chips, 4 workers) — matches Xiaomi's reference config
  - **v6e-32** (32 chips, 8 workers) — our primary test cluster
- `gcloud` CLI authenticated with SSH access to TPU workers
- Model weights uploaded to a GCS bucket (313 GB, 145 safetensors files)
- Use the `*_v6e16.sh` scripts for v6e-16 clusters, the base scripts for v6e-32
