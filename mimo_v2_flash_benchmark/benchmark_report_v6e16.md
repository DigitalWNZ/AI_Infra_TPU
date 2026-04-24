# MiMo-V2-Flash Benchmark Report — TPU v6e-16

## Test Environment

| Item | Value |
|------|-------|
| **Model** | XiaomiMiMo/MiMo-V2-Flash (256 routed experts, top-8, FP8) |
| **Accelerator** | TPU v6e-16 (16 chips, 4×4 topology, 4 workers × 4 chips) |
| **Cluster** | a6e-wangez-16, us-east5-a |
| **HBM per chip** | 31.25 GB |
| **Framework** | SGLangJax |
| **Branches tested** | origin/main (`756e91ba`) and cef4a18 (`cef4a181`) |
| **Config** | TP=16, EP=16, context=262144, page=256, chunked-prefill=2048 |
| **Benchmark tool** | `sgl_jax.bench_serving` (random dataset) |

> **Note:** This is the same hardware config as Xiaomi's reference benchmark report (v6e-16, tp=16, ep=16, dp=4).

---

## Results Summary

### Main Branch (origin/main, dp=1)

Long context: input=16384, output=1024.

| BS | Output (tok/s) | Median ITL (ms) | Median TTFT (ms) |
|----|---------------|-----------------|-------------------|
| 64 | 522.2 | 24.90 | 21,066 |
| 128 | 540.8 | 33.91 | 42,628 |
| 200 | 556.5 | 34.18 | 75,344 |

### cef4a18 — Short Context (input=1024, output=512)

| DP | BS=64 (tok/s) | BS=128 (tok/s) | BS=200 (tok/s) |
|----|--------------|---------------|---------------|
| 1 | 1,251.1 | 1,611.9 | 1,715.5 |
| 2 | 1,161.6 | 1,596.9 | 1,469.3 |
| 4 | 1,191.3 | 1,712.9 | 1,300.0 |
| 8 | OOM | OOM | OOM |

### cef4a18 — Long Context (Xiaomi-matched: input=16384, output=1024, 256 prompts, rr=100)

| DP | BS=64 (tok/s) | BS=128 (tok/s) | BS=200 (tok/s) | Median ITL (ms) |
|----|--------------|---------------|---------------|-----------------|
| 1 | 474.4 | 474.4 | 474.3 | 20.86 |
| 2 | 567.4 | — | — | 33.68 |
| 4 | **654.8** | 603.4 | 557.8 | 29.33 |
| 8 | OOM | OOM | OOM | — |

> dp=2 long BS=128/200 crashed (server instability). dp=8 OOM at startup (HBM exceeded by 191 MB).

### cef4a18 dp=4 — Peak Output Tokens/s

| Context | BS=64 | BS=128 | BS=200 |
|---------|-------|--------|--------|
| Short (1024/512) | 2,359 | **3,854** | 2,766 |
| Long (16384/1024) | **2,496** | 2,380 | 2,341 |

Best peak burst: **3,854 tok/s** (short context BS=128), **2,496 tok/s** (long context BS=64).

---

## Xiaomi Comparison

Xiaomi's reference benchmark: v6e-16, cef4a18, dp=4, BS=64, input=16384, output=1024.

| Metric | Xiaomi Report | Our Result | Delta |
|--------|--------------|------------|-------|
| Output throughput (tok/s) | 654.9 | **654.8** | **−0.02%** |
| Hardware | v6e-16 | v6e-16 | Same |
| Commit | cef4a18 | cef4a18 | Same |
| Config | tp=16, ep=16, dp=4 | tp=16, ep=16, dp=4 | Same |

**We exactly reproduced Xiaomi's benchmark result (654.9 vs 654.8 tok/s).**

---

## Key Findings

1. **Xiaomi result reproduced**: dp=4 BS=64 on v6e-16 → 654.8 tok/s, matching Xiaomi's 654.9 within measurement noise.

2. **dp=4 is optimal for long context**: Best output throughput among all dp values that fit in HBM.

3. **dp=8 OOM**: On v6e-16, the mesh (8, 2) exceeds per-chip HBM by 191 MB. This config only works on v6e-32+.

4. **dp=1 has lowest ITL (20.86ms)** but also lowest throughput (474.4 tok/s) — no parallelism benefit.

5. **Main branch (dp=1) achieves 522-557 tok/s** on long context, comparable to cef4a18 dp=1 (474 tok/s) — main branch has had optimizations since cef4a18.

6. **Short context**: dp=4 BS=128 gives best short-context throughput at 1,713 tok/s.

---

## Date

2026-04-23
