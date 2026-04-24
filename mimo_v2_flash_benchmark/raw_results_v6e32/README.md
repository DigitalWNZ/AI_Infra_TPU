# Raw Benchmark Results — TPU v6e-32

Cluster: **a6e-wangez** (us-east5-b, 8 workers x 4 chips = 32 chips)
Model: **MiMo-V2-Flash** (256 experts, FP8)
Date: 2026-04-21 (main branch), 2026-04-22/23 (cef4a18 DP sweep)

All bench files are JSON output from `sgl_jax.bench_serving` with `--dataset-name random`.

---

## Main Branch (origin/main, dp=1, tp=32, ep=32)

Long context (input=16384, output=1024, burst rate):

| File | BS | Output tok/s |
|------|----|-------------|
| `bench_bs32.json` | 32 | 519.4 |
| `bench_result.json` | 64 | 571.9 |
| `bench_bs128.json` | 128 | 605.9 |

Short context (input=1024, output=1024, burst rate):

| File | BS | Output tok/s |
|------|----|-------------|
| `bench_short.json` | 64 | 1,523.6 |

---

## Main Branch — TP Variation Experiments (dp=1)

Long context (input=16384, output=1024, burst rate), testing reduced TP on 32-chip cluster:

| File | TP | BS | Output tok/s |
|------|----|----|-------------|
| `bench_tp16_bs64.json` | 16 | 64 | 570.0 |
| `bench_tp16_bs128.json` | 16 | 128 | 610.1 |
| `bench_tp8_bs64.json` | 8 | 64 | 441.8 |
| `bench_tp8_bs128.json` | 8 | 128 | 471.4 |

> These experiments tested whether using fewer chips per TP group would improve per-chip efficiency. TP=16 performed comparably to TP=32; TP=8 was significantly worse.

---

## cef4a18 Branch — DP Sweep (tp=32, ep=32)

Long context (input=16384, output=1024, 256 prompts, rr=100):

| File | DP | BS | Output tok/s |
|------|----|----|-------------|
| `bench_dp2_long_bs64.json` | 2 | 64 | 594.4 |
| `bench_dp2_long_bs128.json` | 2 | 128 | 579.1 |
| `bench_dp2_long_bs200.json` | 2 | 200 | 631.6 |
| `bench_dp4_long_bs64.json` | 4 | 64 | 663.1 |
| `bench_dp4_long_bs128.json` | 4 | 128 | 649.4 |
| `bench_dp4_long_bs200.json` | 4 | 200 | 684.5 |
| `bench_dp8_long_bs64.json` | 8 | 64 | 606.0 |
| `bench_dp8_long_bs128.json` | 8 | 128 | 633.0 |
| `bench_dp8_long_bs200.json` | 8 | 200 | 652.5 |

> Best sustained: dp=4 BS=200 at 684.5 tok/s.

---

## Accuracy Result

| File | Benchmark | Samples | Accuracy |
|------|-----------|---------|----------|
| `gsm8k.json` | GSM8K | 200 | 97.5% |

Evaluated using `evalscope` with greedy decoding on main branch (dp=1, tp=32).

---

## Notes

- Main branch files (`bench_result.json`, `bench_bs*.json`, `bench_short.json`, `bench_tp*.json`) contain a single JSON object with standard metrics only.
- cef4a18 DP sweep files (`bench_dp*.json`) contain two JSON objects per file (line 1: standard metrics, line 2: metrics with `server_info` and `max_output_tokens_per_s`).
- `gsm8k.json` is evalscope output format, not bench_serving format.
