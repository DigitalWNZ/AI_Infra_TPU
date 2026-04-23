# Operation Log — MiMo-V2-Flash on TPU v6e-16

## Cluster Info

- **Name**: a6e-wangez-16
- **Zone**: us-east5-a
- **Accelerator**: TPU v6e-16 (16 chips, 4×4 topology)
- **Workers**: 4 (w-0 through w-3)
- **Worker IPs**: 10.202.0.101, 10.202.0.52, 10.202.0.100, 10.202.0.8
- **Master IP**: 10.202.0.101 (w-0)
- **HBM per chip**: 31.25 GB

## Timeline

### Phase 1: Setup (03:47 – 06:22 UTC)

1. **SSH connectivity verified** to all 4 workers.

2. **Main branch installed** on all workers using `uv` + Python 3.12 venv.
   - `uv pip install -e "python[tpu]"` from origin/main (`756e91ba`)

3. **GCSFuse model mount** — encountered gzip-compressed config files:
   - `config.json` and `tokenizer_config.json` in the GCS bucket were gzip-compressed
   - Fix: used `zcat` to decompress before copying to model dir
   - Also needed to download `configuration_mimo_v2_flash.py` and `modeling_mimo_v2_flash.py` from HuggingFace (not in GCS bucket)

4. **Model directory**: 145 split safetensors symlinked from GCSFuse mount, config files decompressed locally.

### Phase 2: Main Branch Benchmark (06:22 – 07:37 UTC)

- **Server startup**: ~29 minutes (weight loading 20 min + JIT precompile 3.5 min)
- dp=1, tp=16, ep=16
- Benchmarks: BS=64, 128, 200 with input=16384, output=1024

Results:
| BS | Output tok/s | Median ITL (ms) |
|----|-------------|-----------------|
| 64 | 522.2 | 24.90 |
| 128 | 540.8 | 33.91 |
| 200 | 556.5 | 34.18 |

### Phase 3: Switch to cef4a18 (07:37 – 09:10 UTC)

**Critical issue: wrong commit checked out.**

- `git fetch origin cef4a18 && git checkout cef4a18` fetched `756e91ba` (main) instead of the orphaned commit
- Root cause: short hash `cef4a18` didn't resolve the orphaned commit via `git fetch origin`
- **Fix**: used full hash `cef4a181508ef22b452a38fe4210478e2e3672b1` for `git fetch origin` and `git checkout`
- Verified: no `dp_size == 1` branching in `engine.py` (correct cef4a18 behavior)
- Non-editable install: `uv pip install "python[tpu]"` (cef4a18 lacks build_editable hook)

### Phase 4: cef4a18 dp=4 Benchmark (09:10 – 09:57 UTC)

- **Xiaomi's exact config**: v6e-16, tp=16, ep=16, dp=4
- Server startup: ~11 minutes
- Mesh: (data=4, tensor=4)

Results — Short context (input=1024, output=512):
| BS | Output tok/s | Median ITL (ms) |
|----|-------------|-----------------|
| 64 | 1,191.3 | 23.34 |
| 128 | 1,712.9 | 29.98 |
| 200 | 1,300.0 | 30.91 |

Results — Long context (Xiaomi-matched, input=16384, output=1024, 256 prompts, rr=100):
| BS | Output tok/s | Median ITL (ms) |
|----|-------------|-----------------|
| 64 | **654.8** | 29.33 |
| 128 | 603.4 | 29.17 |
| 200 | 557.8 | 29.37 |

**dp=4 BS=64 → 654.8 tok/s — matches Xiaomi's reported 654.9 tok/s.**

### Phase 5: cef4a18 dp=1 Benchmark (09:57 – 10:29 UTC)

- Server startup: ~8 minutes (JIT cache from dp=4 partially reused)

Results — Long context:
| BS | Output tok/s | Median ITL (ms) |
|----|-------------|-----------------|
| 64 | 474.4 | 20.86 |
| 128 | 474.4 | 20.85 |
| 200 | 474.3 | 20.86 |

### Phase 6: cef4a18 dp=2 Benchmark (10:29 – 11:08 UTC)

- Server startup: ~8.5 minutes

Results — Short context:
| BS | Output tok/s |
|----|-------------|
| 64 | 1,161.6 |
| 128 | 1,596.9 |
| 200 | 1,469.3 |

Results — Long context:
| BS | Output tok/s |
|----|-------------|
| 64 | 567.4 |
| 128 | Server crashed (TransferEncodingError) |
| 200 | Server crashed |

### Phase 7: cef4a18 dp=8 — OOM (11:08 – 11:22 UTC)

- **RESOURCE_EXHAUSTED**: XLA compile OOM
- Used 31.43G of 31.25G HBM, exceeded by 191 MB
- Breakdown: reserved 260M + program 2.89G + arguments 28.54G
- dp=8 mesh (8, 2) requires more per-chip memory than available
- **Conclusion**: dp=8 is not viable on v6e-16

## Errors Encountered

1. **Gzip config files**: config.json in GCS was gzip-compressed → `OSError: not a valid JSON file`. Fix: `zcat` decompress.

2. **Missing model Python files**: `configuration_mimo_v2_flash.py` not in GCS → `OSError: file not found`. Fix: download from HuggingFace.

3. **Wrong commit on v6e-16**: `git fetch origin cef4a18` resolved to main, not orphaned commit. Fix: use full hash.

4. **dp=8 OOM**: HBM exceeded by 191 MB. Not fixable on v6e-16.

5. **dp=2 long BS=128/200 crash**: Server instability (aiohttp TransferEncodingError). Partial results collected.

## Conclusions

- **Xiaomi reproduction successful**: dp=4 BS=64 on v6e-16 → 654.8 tok/s (Xiaomi: 654.9)
- **Best config on v6e-16**: dp=4, tp=16, ep=16 (same as Xiaomi)
- **dp=8 not viable on v6e-16** (OOM), only works on v6e-32+
- **Main branch (origin/main)** performs well at dp=1 (522-557 tok/s) thanks to recent optimizations

## Date

2026-04-23
