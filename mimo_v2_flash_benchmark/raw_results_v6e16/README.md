# Raw Benchmark Results — TPU v6e-16

Cluster: **a6e-wangez-16** (us-east5-a, 4 workers x 4 chips = 16 chips)
Model: **MiMo-V2-Flash** (256 experts, FP8)
Date: 2026-04-23

All files are JSON output from `sgl_jax.bench_serving` with `--dataset-name random`.

---

## Main Branch (origin/main, dp=1, tp=16, ep=16)

Long context benchmarks (input=16384, output=1024, burst rate):

| File | BS | Output tok/s |
|------|----|-------------|
| `bench_main_dp1_long_bs64.json` | 64 | 522.2 |
| `bench_main_dp1_long_bs128.json` | 128 | 540.8 |
| `bench_main_dp1_long_bs200.json` | 200 | 556.5 |

---

## cef4a18 Branch (dp=1, tp=16, ep=16)

Short context (input=1024, output=512, burst rate):

| File | BS | Output tok/s |
|------|----|-------------|
| `bench_cef_dp1_short_bs64.json` | 64 | 1,251.1 |
| `bench_cef_dp1_short_bs128.json` | 128 | 1,611.9 |
| `bench_cef_dp1_short_bs200.json` | 200 | 1,715.5 |

Long context (input=16384, output=1024, 256 prompts, rr=100):

| File | BS | Output tok/s |
|------|----|-------------|
| `bench_cef_dp1_long_bs64.json` | 64 | 474.4 |
| `bench_cef_dp1_long_bs128.json` | 128 | 474.4 |
| `bench_cef_dp1_long_bs200.json` | 200 | 474.3 |

> Note: dp=1 long context files were collected while the cluster was accidentally on origin/main (not true cef4a18). They lack `max_output_tokens_per_s` and `server_info` fields.

---

## cef4a18 Branch (dp=2, tp=16, ep=16)

Short context (input=1024, output=512, burst rate):

| File | BS | Output tok/s |
|------|----|-------------|
| `bench_cef_dp2_short_bs64.json` | 64 | 1,161.6 |
| `bench_cef_dp2_short_bs128.json` | 128 | 1,596.9 |
| `bench_cef_dp2_short_bs200.json` | 200 | 1,469.3 |

Long context (input=16384, output=1024, 256 prompts, rr=100):

| File | BS | Output tok/s |
|------|----|-------------|
| `bench_cef_dp2_long_bs64.json` | 64 | 567.4 |

> dp=2 long BS=128/200 crashed (server instability, TransferEncodingError).

---

## cef4a18 Branch (dp=4, tp=16, ep=16) — Xiaomi-matched config

Short context (input=1024, output=512, burst rate):

| File | BS | Output tok/s | Peak tok/s |
|------|----|-------------|-----------|
| `bench_cef_dp4_short_bs64.json` | 64 | 1,191.3 | 2,359 |
| `bench_cef_dp4_short_bs128.json` | 128 | 1,712.9 | 3,854 |
| `bench_cef_dp4_short_bs200.json` | 200 | 1,300.0 | 2,766 |

Long context (input=16384, output=1024, 256 prompts, rr=100):

| File | BS | Output tok/s | Peak tok/s |
|------|----|-------------|-----------|
| `bench_cef_dp4_long_bs64.json` | 64 | **654.8** | 2,496 |
| `bench_cef_dp4_long_bs128.json` | 128 | 603.4 | 2,380 |
| `bench_cef_dp4_long_bs200.json` | 200 | 557.8 | 2,341 |

> dp=4 BS=64 long context: **654.8 tok/s** — matches Xiaomi's reported 654.9 tok/s.

---

## Notes

- cef4a18 files with `server_info` contain two JSON objects per file (line 1: standard metrics, line 2: metrics with server config and peak throughput).
- dp=8 was attempted but OOM at startup (exceeded 31.25GB HBM by 191MB). No result files.
- dp=1 short context files (`bench_cef_dp1_short_*.json`) also have two JSON objects per file.
