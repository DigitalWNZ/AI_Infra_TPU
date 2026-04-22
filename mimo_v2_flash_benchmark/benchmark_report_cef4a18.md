# MiMo-V2-Flash Benchmark Report — Commit cef4a18 (DP-Enabled)

**Date:** 2026-04-22
**Author:** Automated benchmark via SGLangJax
**Cluster:** a6e-wangez (TPU v6e-32, us-east5-b)
**Commit:** `cef4a18` (orphaned — PR #213 data parallelism)

---

## 1. Executive Summary

We benchmarked **MiMo-V2-Flash** (256-expert MoE, FP8) on a **TPU v6e-32** cluster using **SGLangJax** at commit `cef4a18`, which contains a full data parallelism (DP) implementation. We swept dp=1,2,4,8 with short-context workloads and ran long-context benchmarks with the best config, including an apples-to-apples comparison against the Xiaomi reference report.

**Key findings:**

| Finding | Detail |
|---------|--------|
| **Best short-context config** | dp=4, tp=32, ep=32 at BS=128 → **1,808 output tok/s** (+20% over dp=1) |
| **Best long-context config** | dp=8, tp=32, ep=32 at BS=200 → **652.5 output tok/s** (matches Xiaomi's 654.9 within 0.4%) |
| **Peak burst throughput** | dp=8 BS=200: 4,224 tok/s — **65% higher** than Xiaomi report's 2,560 |
| **Best decode latency** | dp=8: 28–31 ms median ITL across all long-context batch sizes |
| **Per-chip efficiency** | 20.4 tok/s/chip vs Xiaomi's 40.9 — 2x gap due to TP=32 cross-node overhead |

---

## 2. Test Environment

| Component | Detail |
|-----------|--------|
| **Cluster** | a6e-wangez |
| **Zone** | us-east5-b (GCP) |
| **Accelerator** | TPU v6e-32 (4x8 topology) |
| **Chips** | 32 (8 workers x 4 chips) |
| **Runtime** | v2-alpha-tpuv6e |
| **Framework** | SGLangJax (commit `cef4a18`, PR #213 DP) |
| **JAX** | 0.8.1 (jaxlib 0.8.1, libtpu 0.0.30) |
| **Model** | MiMo-V2-Flash (FP8 static quantization) |
| **Model size** | 313 GB (145 split safetensors files) |
| **Storage** | GCSFuse mount from `gs://0-wangez/` |
| **MoE backend** | EPMoE (Expert Parallel) |
| **Page size** | 256 |
| **Context length** | 262,144 tokens |
| **Chunked prefill** | 2,048 tokens |
| **SWA full-tokens ratio** | 0.2 |
| **mem-fraction-static** | 0.95 |

### 2.1 Parallelism Configuration

On this v6e-32 cluster, `tp_size` must equal the total chip count (32). The JAX device mesh is created as:

```
mesh = (data=dp_size, tensor=tp_size/dp_size)
```

All tested configurations use tp=32, ep=32. Only dp varies.

| Config | JAX Mesh (data, tensor) | Tensor chips per DP rank |
|--------|------------------------:|-------------------------:|
| dp=1   | (1, 32)                 | 32                       |
| dp=2   | (2, 16)                 | 16                       |
| dp=4   | (4, 8)                  | 8                        |
| dp=8   | (8, 4)                  | 4                        |

EPMoE creates its own separate `(expert, tensor)` mesh for expert routing, independent of the DP mesh.

---

## 3. DP Sweep Results (Short Context)

**Workload:** Random data, input=1024, output=512

### 3.1 Output Throughput (tok/s)

| Config | BS=64 | BS=128 | BS=200 |
|--------|------:|-------:|-------:|
| dp=1   | 1,259 | 1,506  | 1,506  |
| dp=2   | 1,325 | 1,741  | 1,600  |
| **dp=4** | **1,315** | **1,808** | **1,384** |
| dp=8   | 1,133 | 1,608  | 1,046  |

### 3.2 Peak Burst Output Throughput (tok/s)

| dp=1 | dp=2 | dp=4 | dp=8 |
|-----:|-----:|-----:|-----:|
| 2,667 | 3,239 | 3,704 | 4,025 |

### 3.3 Analysis

- **dp=4 at BS=128** achieves the highest sustained output throughput: **1,808 tok/s** (+20% over dp=1's 1,506)
- **dp=2** is the most consistent: strong across all batch sizes (1,325 / 1,741 / 1,600), minimal degradation at BS=200
- **dp=4** degrades at BS=200 (1,384 tok/s) because each DP rank only handles 50 requests, leading to underutilization
- **dp=8** underperforms everywhere: only 4 tensor chips per rank provides insufficient compute, and 8-way coordination overhead dominates
- **Peak burst throughput scales monotonically with DP** (2,667 → 4,025) — more DP ranks absorb burst traffic better, even though sustained throughput drops at dp=8

---

## 4. Long-Context DP Sweep Results

**Workload:** 256 prompts, input=16384, output=1024, request-rate=100, random-range-ratio=1.0, flush-cache (Xiaomi-matched parameters)

### 4.1 Output Throughput (tok/s)

| Config | BS=64 | BS=128 | BS=200 |
|--------|------:|-------:|-------:|
| dp=2   | 594.4 | 579.1  | 631.6  |
| dp=4   | 635.9 | —      | —      |
| dp=8   | 606.0 | 633.0  | **652.5** |

### 4.2 Peak Output Throughput (tok/s)

| Config | BS=64 | BS=128 | BS=200 |
|--------|------:|-------:|-------:|
| dp=2   | 1,920 | 2,952  | 3,072  |
| dp=4   | 3,840 | —      | —      |
| dp=8   | 3,264 | 4,191  | **4,224** |

### 4.3 Decode Latency — Median ITL (ms)

| Config | BS=64 | BS=128 | BS=200 |
|--------|------:|-------:|-------:|
| dp=2   | 34.25 | 42.84  | 42.89  |
| dp=8   | **28.14** | **30.86** | **30.90** |

### 4.4 Full Metrics

| Config | Out tok/s | Peak tok/s | Input tok/s | Total tok/s | Med ITL (ms) | P99 ITL (ms) | Med TTFT (ms) | Mean E2E (ms) |
|--------|----------:|-----------:|------------:|------------:|-------------:|-------------:|--------------:|---------------:|
| dp=2, BS=64 | 594.4 | 1,920 | 9,510.6 | 10,105.0 | 34.25 | 48.40 | 36,161 | 110,140 |
| dp=2, BS=128 | 579.1 | 2,952 | 9,266.3 | 9,845.5 | 42.84 | 69.37 | 76,314 | 204,779 |
| dp=2, BS=200 | 631.6 | 3,072 | 10,106.3 | 10,737.9 | 42.89 | 70.10 | 119,158 | 257,923 |
| dp=4, BS=64 | 635.9 | 3,840 | 10,173.6 | 10,809.5 | — | — | — | — |
| dp=8, BS=64 | 606.0 | 3,264 | 9,696.1 | 10,302.1 | 28.14 | 30.15 | 35,369 | 108,033 |
| dp=8, BS=128 | 633.0 | 4,191 | 10,127.8 | 10,760.8 | 30.86 | 37.79 | 77,024 | 191,989 |
| **dp=8, BS=200** | **652.5** | **4,224** | **10,439.6** | **11,092.1** | **30.90** | **38.32** | **114,703** | **243,134** |

### 4.5 Analysis

- **dp=8 BS=200 achieves the highest long-context throughput: 652.5 tok/s** — matching Xiaomi's 654.9 within 0.4%
- **dp=8 has the best ITL** across all batch sizes (28–31 ms median) — 8 DP ranks keep per-rank batches small
- **dp=2** is weaker for long context; ITL degrades at BS≥128 (34→43 ms)
- Unlike short-context where dp=8 underperformed, long-context with 256 prompts gives each dp=8 rank enough work (32 prompts/rank at BS=200)
- **Short-context vs long-context best configs differ:** dp=4 BS=128 for short; dp=8 BS=200 for long

---

## 5. Comparison with Xiaomi Report

**Reference:** "基于 SGLangJax 的 mimo_v2_flash 报告" (2026-04-10)
- **Hardware:** TPU v6e-16, topo 4x4, 4 nodes x 4 chips = 16 chips
- **Branch:** `epic/mimo-v2-flash` (commit `cef4a18`)
- **Parallelism:** TP=16, EP=16, DP=4
- **Benchmark:** 256 prompts, rr=100, `--random-range-ratio 1.0`, `--flush-cache`

### 5.1 Configuration Differences

| Parameter | Xiaomi Report | Our Best (dp=8, BS=200) |
|-----------|:-------------|:---------|
| **Chips** | 16 (4x4, 4 nodes) | 32 (4x8, 8 nodes) |
| **TP / EP** | 16 / 16 | 32 / 32 |
| **DP** | 4 | 8 |
| **Mesh** | (4, 4) | (8, 4) |
| **Commit** | cef4a18 | cef4a18 |
| **Num prompts** | 256 | 256 |
| **Max concurrency** | 64 | 200 |
| **Request rate** | 100 | 100 |
| **random-range-ratio** | 1.0 | 1.0 |
| **flush-cache** | Yes | Yes |

### 5.2 Performance Comparison (best result vs Xiaomi)

| Metric | Xiaomi (16 chips) | Ours dp=8 BS=200 (32 chips) | Ratio |
|--------|------------------:|----------------------------:|------:|
| Output throughput (tok/s) | 654.9 | 652.5 | **99.6%** |
| Peak output throughput (tok/s) | 2,560 | 4,224 | **165%** |
| Input throughput (tok/s) | 10,477.8 | 10,439.6 | 99.6% |
| Total throughput (tok/s) | 11,132.7 | 11,092.1 | 99.6% |
| **Per-chip output (tok/s/chip)** | **40.9** | **20.4** | **50%** |

### 5.3 Why Absolute Throughput is Nearly Identical (0.4% gap)

Despite having 2x the chips, our throughput is within 3% of the Xiaomi report. The primary reason is **TP=32 communication overhead**:

1. **All-reduce scales super-linearly.** Each decode step requires an all-reduce across all TP chips. TP=32 across 8 nodes (4x8 topology) has much higher latency than TP=16 across 4 nodes (4x4 topology):
   - More chips = more synchronization barriers
   - 4x8 topology has longer ICI paths than 4x4
   - Maximum hop distance roughly doubles

2. **MoE all-to-all overhead doubles.** EP=32 has 2x the expert routing traffic of EP=16.

3. **Decode is memory-bandwidth-bound.** Doubling chips halves per-chip work, but communication cost stays constant or grows. The ratio of useful work to communication drops.

4. **DP rank tensor parallelism.** Each DP rank uses `tp_size/dp_size` tensor chips:
   - Xiaomi: 16/4 = 4 chips per rank (all within one node, fastest ICI)
   - Ours: 32/4 = 8 chips per rank (spanning 2 nodes, cross-node ICI)

### 5.4 Why Peak Burst is 65% Higher (4,224 vs 2,560)

Our 2x chip count and higher DP help absorb traffic spikes:
- More aggregate HBM bandwidth for concurrent KV cache reads
- 8 DP ranks (vs 4) can handle higher instantaneous concurrency
- Prefill (compute-bound) benefits more from additional chips than decode (memory-bandwidth-bound)

---

## 6. DP Recommendations

| Workload Profile | Best Config | Sustained tok/s | Why |
|-----------------|-------------|----------------:|-----|
| **Short context, high throughput** | **dp=4, BS=128** | **1,808** | Best short-context sustained throughput |
| **Long context, high throughput** | **dp=8, BS=200** | **652.5** | Matches Xiaomi's 654.9; best long-context result |
| Long context, low latency | dp=8, BS=64 | 606.0 | Best ITL (28.14 ms median) |
| Short context, variable BS | dp=2, BS=128 | 1,741 | Most consistent across BS=64-200 |
| Short context, burst absorption | dp=8, BS=128 | 1,608 | Highest peak burst (4,025 tok/s short; 4,224 long) |

---

## 7. Areas for Improvement

### 7.1 Reduce TP Degree

The biggest performance bottleneck is TP=32 across 8 nodes. If the cluster were v6e-16 (16 chips, 4 nodes), tp=16 with dp=4 would match the Xiaomi report config exactly and likely achieve similar per-chip efficiency (40.9 tok/s/chip).

**Recommendation:** Test on a v6e-16 cluster with tp=16, ep=16, dp=4 to verify.

### 7.2 Optimize for Long Context

The dp=4 long-context throughput (635.9 tok/s) is 65% lower than short-context (1,808 tok/s). Long-context workloads are dominated by prefill, where DP provides less benefit because each rank still processes the full 16K-token input in chunks.

**Options:**
- Increase `--chunked-prefill-size` from 2048 to 4096/8192 to reduce prefill latency
- Profile SWA vs full attention allocation with long-context workloads

### 7.3 Scale Beyond 32 Chips

On larger clusters (v6e-64, v6e-128), the constraint that tp=total_chips could be avoided by using multi-pod mesh configurations. This would allow tp=16 with dp=8 or dp=16, potentially unlocking much higher throughput.

### 7.4 Merge DP Implementation to main

The DP implementation at cef4a18 (PR #213) works correctly but is orphaned. Merging it to `origin/main` would:
- Make dp>1 available without fetching an orphaned commit
- Allow combining DP with the improved MiMo-V2-Flash model code on main (dedicated `mimo_v2_flash.py` with proper v_head_dim, attention_sink_bias, SWA support)

---

## 8. Startup Characteristics

| Phase | Duration |
|-------|----------|
| GCSFuse cache warmup | ~4 min (291.6 GB read) |
| Weight loading (145 split safetensors) | ~1 min |
| KV cache allocation | ~1 sec |
| JIT precompilation | ~3.5 min |
| **Total cold start** | **~9 min** |

---

## 9. Summary

### 9.1 Short-Context Results (input=1024, output=512)

| Benchmark | Config | Output tok/s | Peak tok/s | Notes |
|-----------|--------|-------------:|-----------:|-------|
| BS=64 | dp=1 | 1,259 | 2,667 | Baseline |
| BS=128 | dp=1 | 1,506 | — | |
| BS=64 | dp=2 | 1,325 | 3,239 | |
| BS=128 | dp=2 | 1,741 | — | Most consistent |
| BS=200 | dp=2 | 1,600 | — | |
| **BS=128** | **dp=4** | **1,808** | **3,704** | **Best short-context** |
| BS=200 | dp=4 | 1,384 | — | |
| BS=128 | dp=8 | 1,608 | 4,025 | |

### 9.2 Long-Context Results (input=16384, output=1024, Xiaomi-matched)

| Benchmark | Config | Output tok/s | Peak tok/s | Med ITL (ms) | Notes |
|-----------|--------|-------------:|-----------:|-------------:|-------|
| BS=64 | dp=2 | 594.4 | 1,920 | 34.25 | |
| BS=128 | dp=2 | 579.1 | 2,952 | 42.84 | |
| BS=200 | dp=2 | 631.6 | 3,072 | 42.89 | |
| BS=64 | dp=4 | 635.9 | 3,840 | — | |
| BS=64 | dp=8 | 606.0 | 3,264 | 28.14 | Best ITL |
| BS=128 | dp=8 | 633.0 | 4,191 | 30.86 | |
| **BS=200** | **dp=8** | **652.5** | **4,224** | **30.90** | **Best long-context (99.6% of Xiaomi)** |

### 9.3 Xiaomi Comparison

| Metric | Xiaomi (16 chips) | Ours best (32 chips) | Ratio |
|--------|------------------:|---------------------:|------:|
| Output tok/s | 654.9 | 652.5 (dp=8, BS=200) | 99.6% |
| Peak tok/s | 2,560 | 4,224 (dp=8, BS=200) | 165% |
| Per-chip tok/s | 40.9 | 20.4 | 50% |

**Bottom line:** Different DP configs are optimal for different workloads. For short context, dp=4 BS=128 is best (1,808 tok/s). For long context, dp=8 BS=200 matches the Xiaomi report's throughput within 0.4% (652.5 vs 654.9) with 65% higher peak burst — despite 2x TP communication overhead from running on 32 chips instead of 16.

---

## 10. Raw Data Reference

All benchmark JSON results on worker-0:
- Short-context DP sweep: `/tmp/bench_dp{1,2,4,8}_bs{64,128,200}.json`
- Long-context DP sweep: `/tmp/bench_dp{2,4,8}_long_bs{64,128,200}.json`

Operation log: `operation_log_cef4a18.md`
E2E script: `benchmark_e2e_cef4a18.sh`
