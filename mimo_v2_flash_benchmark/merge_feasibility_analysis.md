# Merge Feasibility Analysis: cef4a18 (DP) → origin/main

**Date:** 2026-04-24
**Repo:** sgl-project/sglang-jax
**Target:** Merge data parallelism (DP) from commit `cef4a18` into `origin/main`

---

## 1. Branch Divergence Summary

| Metric | Value |
|--------|-------|
| Merge base | `56ff2af8` (2026-04-09, PR #897) |
| Commits on main since base | 21 |
| Commits on cef4a18 since base | 18 |
| Days diverged | 15 |
| Total files changed (diff) | 103 |
| Lines changed | +9,571 / −10,930 |
| Merge conflict files | **21** |
| Total conflict markers | **158** |

---

## 2. What cef4a18 Contains (18 commits)

### 2.1 The Core DP Implementation — PR #213 (1 commit, 67 files, +8,652 / −2,157)

This is the primary merge target. It implements single-controller data parallelism:

| Component | File | Changes |
|-----------|------|---------|
| **Entry point** | `engine.py` | Remove `dp_size == 1` guard (+24/−57 lines) |
| **Scheduler** | `scheduler.py` | Per-DP request queues, DP-aware scheduling, round-robin/shortest-queue dispatch (+677 lines) |
| **Batch management** | `schedule_batch.py` | DP-indexed batch management, input reordering by DP rank, DP-aware metadata (+2,453 lines) |
| **Scheduling policy** | `schedule_policy.py` | DP-aware prefill admission, per-DP capacity tracking (+249 lines) |
| **KV cache allocator** | `allocator.py` | Per-DP allocation/deallocation, DP-indexed free lists (+537 lines) |
| **Memory pool** | `memory_pool.py` | DP-sharded token pools, `PartitionSpec('data', ...)` (+200 lines) |
| **Radix cache** | `radix_cache.py`, `swa_radix_cache.py` | Per-DP prefix trees (+295 lines) |
| **Model runner** | `model_runner.py` | DP mesh creation `(dp, tp/dp)`, `attention_tp_size` (+198 lines) |
| **Attention kernel** | `flashattention_backend.py` | `shard_map` with DP partition specs (+151 lines) |
| **Native attention** | `native_backend.py` | DP-aware native attention path (+112 lines) |
| **Sampler** | `sampler.py` | `P('data', None)` sharding (+10 lines) |
| **Logits** | `logits_processor.py` | DP-aware logprobs (+38 lines) |
| **Embeddings** | `embeddings.py` | DP-aware embedding layer (+126 lines) |
| **MoE layer** | `fused_moe.py`, `moe.py` | DP-compatible EPMoE (+203 lines) |
| **KV cache update** | `update_kv_cache.py` | DP-aware cache writes (+88 lines) |
| **Output processing** | `scheduler_output_processor_mixin.py` | Per-DP result extraction (+448 lines) |
| **Tests** | 6 new test files | DP-specific unit tests (+1,726 lines) |
| **Design doc** | `data_parallelism.md` | Architecture documentation (+159 lines) |
| **Bench tool** | `bench_serving.py` | Enhanced benchmarking with `server_info` and peak metrics (+1,693 lines) |

### 2.2 Supporting Commits (17 commits)

| Category | Commits | Description |
|----------|---------|-------------|
| MiMo model | 1 | Basic `mimo.py` (Qwen2 wrapper) — superseded by main's `mimo_v2_flash.py` |
| SWA / Attention sink | 5 | `attention_sink_bias`, SWA dual-pool, SWA OOB fixes |
| Forward metadata optimization | 2 | 31x speedup for `get_forward_metadata` |
| KV cache shape | 2 | Shape changes for rpav3 compatibility |
| Bug fixes | 5 | DP correctness fixes, astype removal, stale v_head_dim |
| Process guard removal | 1 | Remove `jax.process_count() == 1` check |

---

## 3. What main Has That cef4a18 Lacks (21 commits)

| Feature | PRs | Impact on DP Merge |
|---------|-----|-------------------|
| **MLA (Multi-Latent Attention)** | #895, #911, #926, #941 | New attention backend — DP partition specs needed |
| **Simple GLA (Linear Attention)** | #899 | New attention backend — DP partition specs needed |
| **DeepSeek v3 model** | (dedicated model) | Deleted in cef4a18 — keep main's version |
| **MiMo-V2-Flash (dedicated)** | #922, #936 | 946-line dedicated implementation vs cef4a18's 49-line wrapper |
| **SWA improvements** | #921, #930, #938, #944 | OOM handling, paged allocation — must integrate with DP allocator |
| **FP8 FusedEPMoE** | #914 | Kernel optimizations — need DP compatibility check |
| **RPAv3 integration** | #913 | New attention kernel — DP partition specs needed |
| **Tuned MoE block configs** | #931, #932, #942 | Config-only, no conflict |
| **Sampler RNG fix** | #940 | cef4a18 removed sampler RNG; main fixed it differently |
| **Misc** | #918, #920, #924 | Profile endpoint, dependency fixes — no DP impact |

---

## 4. Merge Conflict Analysis

### 4.1 Conflict Breakdown by Severity

**High severity (core DP logic — require careful manual resolution):**

| File | Conflicts | Issue |
|------|:---------:|-------|
| `schedule_batch.py` | 8 | Heaviest DP file (2,453 lines changed). Both branches modified batch management extensively. |
| `memory_pool.py` | 14 | DP sharding vs main's SWA dual-pool changes. Both restructured pool allocation. |
| `model_runner.py` | 7 | DP mesh setup vs main's SWA `adjust_layer_num` refactoring. |
| `allocator.py` | 5 | DP-indexed allocation vs main's SWA eviction improvements. |
| `flashattention_backend.py` | 7 | DP `shard_map` specs vs main's RPAv3 + SWA metadata changes. |
| `native_backend.py` | 9 | DP native attention vs main's SWA native backend changes. |
| `ragged_paged_attention_v3.py` | 7 | DP partition specs vs main's new RPAv3 kernel. |

**Medium severity (functional but less complex):**

| File | Conflicts | Issue |
|------|:---------:|-------|
| `mimo_v2_flash.py` | 24 | cef4a18 has basic `mimo.py`; main has 946-line dedicated implementation. Resolution: keep main's version. |
| `fused_moe.py` | 3 | DP-compatible MoE vs main's FP8 optimizations. |
| `kernel.py` (fused_moe) | 5 | MoE kernel changes on both sides. |
| `update_kv_cache.py` | 3 | DP-aware cache writes vs main's cache format changes. |

**Low severity (tests, benchmarks, config):**

| File | Conflicts | Issue |
|------|:---------:|-------|
| `test_swa_allocator.py` | 44 | Most conflicts by count, but test code — use main's tests + add DP tests |
| `test_kv_cache.py` | 6 | Test code |
| `test_flashattention.py` | 2 | Test code |
| `bench_flashattention.py` | 6 | Benchmark code |
| `bench_fused_moe.py` | 4 | Benchmark code |
| `model_config.py` | 1 | Config change |
| `schedule_policy.py` | 1 | Minor conflict |
| `common.py` | 1 | Minor conflict |
| `run_suite.py` | 1 | Test suite config |

### 4.2 Key Semantic Conflicts (Not Just Textual)

Beyond textual merge conflicts, these areas have **semantic conflicts** — both branches changed the same logic in incompatible ways:

1. **SWA memory management**: Main's `#921` rewrote SWA memory pooling with paged allocation and eviction. cef4a18's DP adds per-DP allocation. These need to be combined: DP-indexed SWA paged allocation.

2. **Sampler RNG**: Main's `#940` fixed sampler RNG with `fold_in(base_key, step)`. cef4a18 removed RNG from the sampler entirely. Need to decide which approach and make it DP-compatible.

3. **MiMo model**: cef4a18 uses a 49-line Qwen2 wrapper. Main has a 946-line dedicated implementation with `v_head_dim`, `attention_sink_bias`, SWA support. Main's version is strictly better — but the DP changes to model layers (embeddings, linear, MoE) need to work with main's model code.

4. **RPAv3 attention kernel**: Main integrated a new RPAv3 kernel (`#913`). cef4a18's DP partition specs are written for the old attention kernel. Need to add DP partition specs to RPAv3.

5. **MLA attention**: Main added MLA (`#911, #926`). cef4a18 doesn't know about MLA. Need to add DP partition specs to MLA attention.

---

## 5. Feasibility Assessment

### 5.1 Approach Options

**Option A: Direct merge cef4a18 → main**
- Resolve 21 files, 158 conflict markers
- Estimated effort: **3–5 engineer-days**
- Risk: **High** — semantic conflicts in memory management and attention kernels are complex. Silent bugs likely (wrong DP sharding, incorrect cache indexing).
- Testing: Need to verify dp=1 (regression), dp=2/4/8 on v6e-16 and v6e-32.

**Option B: Cherry-pick PR #213 only, rebase onto current main**
- Extract the DP-specific changes from PR #213 (the 67-file, 8,652-line commit) and port them to current main
- Skip the 17 supporting commits (SWA, MiMo wrapper, etc.) that are either superseded or already on main
- Estimated effort: **5–8 engineer-days**
- Risk: **Medium** — more work upfront but cleaner result. Can integrate with main's newer SWA, MLA, RPAv3 from scratch.
- Benefit: No baggage from cef4a18's older SWA/MiMo code.

**Option C: Incremental port of DP logic onto main**
- Port DP changes file-by-file onto current main in logical groups:
  1. Mesh + model runner (2 days)
  2. Memory pool + allocator + cache (3 days)
  3. Scheduler + batch management (3 days)
  4. Attention kernels + sampler (2 days)
  5. Tests + benchmarks (1 day)
- Estimated effort: **8–12 engineer-days**
- Risk: **Low** — each increment can be tested independently
- Benefit: Best code quality, proper integration with all main features

### 5.2 Recommendation

**Option B (cherry-pick + rebase)** is the best balance of effort and risk.

Rationale:
- The 17 supporting commits on cef4a18 are either superseded (MiMo model, SWA) or already on main (forward metadata optimization). Cherry-picking only PR #213 avoids re-introducing older code.
- PR #213 is a single merge commit that contains all DP logic. It can be analyzed as one coherent change set.
- The core DP architecture (single-controller, `shard_map`, DP-indexed allocator) is sound and well-documented in the design doc.
- Main has 3 new attention backends (MLA, RPAv3, GLA) that need DP partition specs — these are additive work regardless of merge approach.

### 5.3 Specific Challenges

| Challenge | Difficulty | Notes |
|-----------|:----------:|-------|
| DP + SWA paged allocation | Hard | Main's `#921` heavily restructured SWA memory. Must combine DP indexing with paged eviction. |
| DP + RPAv3 kernel | Medium | Need `shard_map` in/out specs for the new kernel. Follow cef4a18's pattern for old kernel. |
| DP + MLA attention | Medium | New attention backend needs DP partition specs. |
| DP + FP8 FusedEPMoE | Easy | EPMoE creates its own mesh — mostly independent of DP. |
| DP + sampler RNG | Easy | Small decision: keep main's `fold_in` approach, make it DP-aware. |
| `schedule_batch.py` rewrite | Hard | 2,453-line diff. Both branches restructured this file. Most time-consuming conflict. |
| DP + ChunkCache bug | Medium | cef4a18 has a known `ChunkCache` crash at dp=1 with long context. Must fix during port. |
| Testing on TPU | Medium | Need v6e-16 or v6e-32 cluster for integration testing. Unit tests can run on CPU. |

### 5.4 Risk Summary

| Risk | Likelihood | Impact | Mitigation |
|------|:----------:|:------:|------------|
| Silent DP sharding bugs | High | High | Comprehensive unit tests + numerical parity checks (dp=1 output == dp=N output) |
| SWA + DP memory corruption | Medium | High | Test with MiMo-V2-Flash specifically (hybrid SWA model) |
| Performance regression at dp=1 | Medium | Medium | Benchmark dp=1 before/after with identical parameters |
| ChunkCache bug not fixed | Low | Medium | Reproduce and fix the `full_lru_list_evictable_size` error |
| MLA/GLA incompatibility | Low | Low | These are new features; DP support can be added incrementally |

---

## 6. Estimated Timeline

| Phase | Duration | Activities |
|-------|:--------:|------------|
| 1. Analysis & planning | 1 day | Identify all DP touchpoints in current main, create test plan |
| 2. Core DP port | 3–4 days | Port scheduler, batch mgmt, allocator, memory pool from PR #213 onto main |
| 3. Kernel integration | 2 days | Add DP partition specs to RPAv3, MLA, native attention, SWA backends |
| 4. SWA + DP integration | 1–2 days | Combine main's paged SWA allocation with DP indexing |
| 5. Testing | 2–3 days | Unit tests (CPU), integration tests (v6e-16/32), benchmark parity |
| **Total** | **9–12 days** | |

---

## 7. Conclusion

Merging DP from cef4a18 to main is **feasible but non-trivial**. The core DP architecture is well-designed (single-controller, JAX SPMD, DP-indexed caches) and the design doc provides clear guidance. The main challenges are:

1. **21 conflict files with 158 conflict markers** — most are resolvable but `schedule_batch.py` and `memory_pool.py` require deep understanding of both branches.
2. **3 new attention backends on main** (MLA, RPAv3, GLA) that need DP partition specs — additive work.
3. **SWA memory management** was rewritten on both branches — must be carefully combined.
4. **Known bug**: cef4a18's `ChunkCache` crashes at dp=1 with long context — must be fixed during the port.

The recommended approach is **Option B: cherry-pick PR #213 and rebase onto main**, estimated at 5–8 engineer-days for the port plus 2–3 days for testing. This avoids re-introducing cef4a18's older SWA and MiMo code while preserving all DP logic.
