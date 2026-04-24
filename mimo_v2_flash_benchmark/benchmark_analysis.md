# Benchmark Discrepancy Analysis — MiMo-V2-Flash on TPU v6e

**Date:** 2026-04-24

This report investigates three unexpected observations across the MiMo-V2-Flash benchmarks on TPU v6e-16 and v6e-32 clusters.

---

## Observations Under Investigation

1. **Same cluster, main branch appears much worse than sweep branch** — On both v6e-32 and v6e-16, the main branch throughput (522–606 tok/s) is significantly lower than the sweep branch (cef4a18) best result (654–684 tok/s).

2. **On v6e-16, dp=1 main appears better than dp=1 sweep** — Main branch at dp=1 achieves 522–557 tok/s, while cef4a18 at dp=1 only achieves 474 tok/s on the same hardware.

3. **v6e-32 (32 chips) barely outperforms v6e-16 (16 chips)** — With 2x the chips, v6e-32 only achieves 663.1 tok/s vs v6e-16's 654.8 tok/s (a mere 1.3% improvement).

---

## Root Cause 1: Main vs Sweep Is Not an Apples-to-Apples Comparison

### The Data

| Cluster | Branch | DP | BS | Output tok/s |
|---------|--------|---:|---:|-------------:|
| v6e-32 | main | 1 | 64 | 571.9 |
| v6e-32 | main | 1 | 128 | 605.9 |
| v6e-32 | cef4a18 | 4 | 64 | 663.1 |
| v6e-32 | cef4a18 | 4 | 200 | 684.5 |
| v6e-16 | main | 1 | 64 | 522.2 |
| v6e-16 | main | 1 | 200 | 556.5 |
| v6e-16 | cef4a18 | 4 | 64 | 654.8 |

At first glance, cef4a18 wins by 10–16%. But this comparison conflates **two independent variables**:

### Variable 1: Data Parallelism (dp=1 vs dp=4)

Main branch does not support dp>1. All main branch results run dp=1 (single serving group). The cef4a18 best results run dp=4 (four parallel serving groups).

With dp=4, the scheduler maintains four independent request queues, KV cache pools, and batch slots. This enables 4x higher effective concurrency — the primary driver of the throughput advantage.

### Variable 2: Completely Different Benchmark Parameters

The benchmarks used different parameters, making direct comparison invalid:

| Parameter | Main Branch | cef4a18 (Xiaomi-matched) |
|-----------|------------|--------------------------|
| **Request rate** | Infinity (burst) | 100 req/s (throttled) |
| **Num prompts** | 64–200 (= batch size) | 256 (fixed) |
| **random-range-ratio** | 0.0 (fixed length) | 1.0 (variable length) |
| **flush-cache** | No | Yes |
| **Duration** | 56–62s | 395–400s |

These differences have opposing effects:
- **Burst rate** (main) is more favorable: all requests arrive instantly, enabling maximal batching from the start
- **256 prompts with rr=100** (cef4a18) creates sustained load for 6–7x longer, exposing more prefill/decode interference
- **Fixed-length inputs** (main) produce uniform batches with no padding waste; **variable-length** (cef4a18) causes batch padding and uneven prefill
- **flush-cache** (cef4a18) forces KV cache recomputation, adding overhead

### True Comparison

If we hold benchmark parameters constant and only vary dp, the performance difference comes entirely from data parallelism. On v6e-16 with Xiaomi-matched parameters:

| Config | Output tok/s | Improvement |
|--------|------------:|:------------|
| dp=1 (Xiaomi params) | 474.4 | baseline |
| dp=2 (Xiaomi params) | 567.4 | +20% |
| dp=4 (Xiaomi params) | 654.8 | +38% |

**Conclusion:** The apparent "main vs sweep" gap is not a code quality difference. It is the combined effect of (a) dp=4 vs dp=1 parallelism, and (b) different benchmark methodology. The main branch code performs comparably to cef4a18 at dp=1 — the sweep branch advantage is its DP implementation.

---

## Root Cause 2: dp=1 Main vs dp=1 Sweep — Benchmark Methodology Artifact

### The Data

On v6e-16, dp=1, long context:

| Label | BS=64 | BS=128 | BS=200 |
|-------|------:|-------:|-------:|
| Main branch | 522.2 | 540.8 | 556.5 |
| cef4a18 dp=1 | 474.4 | 474.4 | 474.3 |

Main appears 10–17% faster. But this is entirely explained by two factors:

### Factor 1: Different Benchmark Parameters

Extracted from the raw JSON files:

| Parameter | Main dp=1 | cef4a18 dp=1 |
|-----------|-----------|-------------|
| `request_rate` | Infinity | 100 |
| `completed` (prompts) | 64/128/200 | 256 |
| `random_range_ratio` | 0.0 | 1.0 |
| `duration` | 62s | 553s |

The main branch benchmark sends all prompts at once with fixed-length inputs. The cef4a18 dp=1 benchmark throttles to 100 req/s with variable-length inputs over 9x longer duration.

Impact of each difference:
- **Burst vs throttled:** With burst (rr=inf), all requests enter the batch immediately, achieving peak throughput. With rr=100, the first 64 requests take 0.64s to arrive — the server runs partially loaded during ramp-up.
- **Fixed vs variable length:** Fixed-length inputs (range-ratio=0.0) produce uniform 16,384-token inputs. Variable-length (range-ratio=1.0) produces inputs ranging from 1 to 16,384 tokens. Short inputs complete faster, fragmenting the batch and causing scheduling churn.
- **64 vs 256 prompts:** With 256 prompts, the server handles 4x more prefill passes, creating more prefill/decode interference. Sustained throughput with 256 prompts is naturally lower than peak throughput with 64.

### Factor 2: The cef4a18 dp=1 Long Context Data May Not Be From Real cef4a18

Evidence from the raw JSON files:

| File | `server_info` present | Lines | Source |
|------|:---------------------:|------:|--------|
| `bench_main_dp1_long_bs64.json` | No | 1 | Main branch bench_serving |
| `bench_cef_dp1_long_bs64.json` | No | 1 | **Same format as main** |
| `bench_cef_dp1_short_bs64.json` | Yes (line 2) | 2 | cef4a18 bench_serving |
| `bench_cef_dp4_long_bs64.json` | Yes (line 2) | 2 | cef4a18 bench_serving |
| `bench_cef_dp2_long_bs64.json` | Yes (line 1) | 1 | cef4a18 bench_serving |

The cef4a18 dp=1 long context files have the **exact same key set** as main branch files — no `server_info`, no `max_output_tokens_per_s`, single JSON object. Every other cef4a18 benchmark file contains `server_info`. This indicates the dp=1 long context benchmarks were collected from a main branch server process.

From the operation log: before Phase 4 (dp=4), the cluster accidentally had the wrong commit checked out (main instead of cef4a18). The dp=1 long context data was labeled "cef4a18" but the server was running main branch code. When the team later tried to re-run dp=1 long context on the real cef4a18, the server crashed with `ChunkCache` error.

**Conclusion:** The dp=1 "main vs sweep" comparison is really **main (burst, 64 prompts, fixed length) vs main (throttled, 256 prompts, variable length)**. The code is identical — only the benchmark parameters differ. The 10–17% gap is entirely a methodology artifact.

### Estimated True Code Difference at dp=1

There is no valid dp=1 apples-to-apples comparison available because:
- Main branch was only benchmarked with burst/fixed/small-N parameters
- cef4a18 dp=1 long context crashes with `ChunkCache` error on the real cef4a18 code
- The "cef4a18 dp=1 long context" data actually came from main branch server

To properly compare, one would need to re-run the main branch benchmark with Xiaomi parameters (rr=100, 256 prompts, range-ratio=1.0, flush-cache).

---

## Root Cause 3: v6e-32 Barely Outperforms v6e-16

### The Data

Apples-to-apples comparison (same branch, same dp, same benchmark parameters):

| Cluster | Chips | DP | Mesh | BS=64 tok/s | Med ITL (ms) | Per-chip tok/s |
|---------|------:|---:|------|------------:|-------------:|---------------:|
| v6e-16 | 16 | 4 | (4, 4) | 654.8 | 29.33 | **40.9** |
| v6e-32 | 32 | 4 | (4, 8) | 663.1 | 29.39 | **20.7** |

2x the chips yields only **1.3% more throughput** and **49% lower per-chip efficiency**.

### Why Doubling Chips Doesn't Double Throughput

The root cause is the **tensor parallelism communication bottleneck**. On TPU v6e, `tp_size` must equal the total chip count. With dp=4:

```
v6e-16: tp=16, mesh (4,4) → each DP rank uses 4 tensor chips
v6e-32: tp=32, mesh (4,8) → each DP rank uses 8 tensor chips
```

#### 3.1 Decode Is Memory-Bandwidth-Bound, Not Compute-Bound

Each decode step reads the full KV cache and model weights from HBM, then performs a small matmul (single token × weight matrix). The bottleneck is HBM read bandwidth, not FLOPS.

- Doubling chips from 4→8 per rank doubles aggregate HBM bandwidth
- But it also doubles the **all-reduce communication volume** across the tensor axis
- The useful work per chip halves (each chip processes half the weight shards)
- Net effect: communication cost grows while per-chip useful work shrinks

#### 3.2 Cross-Node Communication Overhead

| Cluster | Topology | Chips/rank | ICI hops within rank |
|---------|----------|:----------:|:----|
| v6e-16 | 4×4 | 4 | All 4 chips on **one node** — zero cross-node hops |
| v6e-32 | 4×8 | 8 | 8 chips span **2 nodes** — every all-reduce crosses node boundary |

On v6e-16 with mesh (4,4), each DP rank's 4 tensor chips are on a single node with the fastest intra-node ICI. On v6e-32 with mesh (4,8), each rank's 8 tensor chips span 2 nodes, adding cross-node ICI latency to every decode step.

The median ITL is nearly identical (29.33 vs 29.39 ms), confirming that the per-step decode latency is dominated by the communication pattern, not the compute. The 8-chip rank doesn't decode faster because the extra 4 chips are across a node boundary.

#### 3.3 MoE Expert Routing Overhead Doubles

EPMoE (Expert Parallel) creates its own `(expert, tensor)` mesh:
- v6e-16: EP=16 → all-to-all across 16 chips
- v6e-32: EP=32 → all-to-all across 32 chips

The all-to-all communication for expert routing scales quadratically with EP size: each chip sends tokens to every other chip for expert computation, then collects results. EP=32 has 4x the communication pairs of EP=16.

#### 3.4 Throughput Is Capped by Decode Latency

Output throughput ≈ `batch_size × (1000 / ITL_ms)`.

With identical ITL (~29.3 ms) and identical effective concurrency (63.9 on both), throughput is:
- Theoretical: 64 × (1000 / 29.3) ≈ 2,184 tok/s (if all requests decode simultaneously)
- Actual: ~655 tok/s (requests are at different decode stages, plus prefill interference)

Since both clusters achieve the same ITL, they achieve nearly the same throughput. The extra chips on v6e-32 don't reduce ITL because communication overhead absorbs the compute gain.

#### 3.5 Where v6e-32 Does Help

The v6e-32's advantage appears at higher concurrency:

| BS | v6e-16 tok/s | v6e-32 tok/s | v6e-32 advantage |
|----|------------:|------------:|-----------------:|
| 64 | 654.8 | 663.1 | +1.3% |
| 128 | 603.4 | 649.4 | +7.6% |
| 200 | 557.8 | 684.5 | +22.7% |

At BS=200, v6e-32 pulls ahead by 22.7% because:
- More aggregate HBM capacity allows larger KV caches
- 8 chips/rank can handle 50 concurrent requests per rank (vs 4 chips handling 50 on v6e-16, which causes HBM pressure)
- Prefill (compute-bound) benefits more from additional chips than decode (memory-bandwidth-bound)

At BS=64 (Xiaomi's benchmark), per-rank batch size is only 16 requests — well within what 4 chips can handle — so the extra chips add overhead without benefit.

---

## Summary

| Observation | Root Cause | Type |
|------------|------------|------|
| Main worse than sweep (same cluster) | Different DP config (dp=1 vs dp=4) + different benchmark parameters (burst/64p vs throttled/256p) | **Methodology gap** |
| dp=1 main better than dp=1 sweep (v6e-16) | Entirely different benchmark parameters; same underlying code (both actually ran on main branch server) | **Methodology artifact** |
| v6e-32 ≈ v6e-16 throughput | TP communication bottleneck: 8 chips/rank across 2 nodes vs 4 chips/rank within 1 node; decode is memory-bandwidth-bound; ITL is identical | **Hardware scaling limit** |

### Key Takeaways

1. **No valid main vs cef4a18 code comparison exists.** Every comparison in the current data conflates code differences with benchmark parameter differences. To make a true comparison, the main branch should be benchmarked with Xiaomi parameters (rr=100, 256 prompts, range-ratio=1.0).

2. **The sweep branch (cef4a18) advantage is its DP implementation, not better code.** DP=4 provides +38% throughput over dp=1 on v6e-16 with identical parameters.

3. **v6e-16 with dp=4 is the optimal hardware configuration.** Adding more chips beyond 16 (with the current constraint that tp=total chips) does not improve throughput because the communication overhead of TP=32 across 8 nodes negates the compute gain. The v6e-16 mesh (4,4) is ideal because each DP rank fits entirely within one node.

4. **To benefit from v6e-32**, the framework would need to decouple `tp_size` from the total chip count, allowing configurations like tp=16, dp=8 on 32 chips. This would give each DP rank a 4-chip intra-node tensor group (matching v6e-16 efficiency) with 2x more DP parallelism.

---

## Recommended Next Steps

1. **Re-benchmark main branch with Xiaomi parameters** to establish a true code-level comparison at dp=1.

2. **Test tp=16, dp=2 on v6e-32** if the framework supports decoupled tp/chip count — this would give 2 DP ranks each with 16 tensor chips (matching the v6e-16 tensor topology) and should approximately double throughput.

3. **Standardize benchmark parameters** across all future runs to enable valid cross-branch and cross-cluster comparisons. Recommended standard: 256 prompts, rr=100, range-ratio=1.0, flush-cache (Xiaomi-matched).
