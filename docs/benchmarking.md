# Benchmarking Guide

## Overview

The benchmark suite lives in `src/benchmarks/` and covers three dimensions:

| Module | What it measures | Why it matters |
|--------|-----------------|----------------|
| `speed.py` | Tokens/sec, TTFT, TBT percentiles, context scaling | User-facing latency |
| `perplexity.py` | Perplexity (PPL) on held-out text | Model quality / training progress |
| `memory.py` | Peak RSS, memory growth over context | Stays within 32 GB target |
| `run_all.py` | All three + structured report | Checkpoint comparison |

---

## Full Suite (recommended)

```bash
python -m src.benchmarks.run_all \
    --checkpoint checkpoints/small-int8/ \
    --test-data data/raw/test.txt \
    --output reports/ \
    --speed-tokens 200 \
    --speed-runs 10 \
    --memory-tokens 500 \
    --context-sweep
```

Writes two files to `reports/`:
- `bench_<checkpoint>_<timestamp>.json` — machine-readable, for tracking over time
- `bench_<checkpoint>_<timestamp>.md`  — human-readable Markdown summary

If you don't have test data yet, skip perplexity:
```bash
python -m src.benchmarks.run_all \
    --checkpoint checkpoints/small-int8/ \
    --no-perplexity
```

---

## Individual Benchmarks

### Speed

```bash
python -m src.benchmarks.speed \
    --checkpoint checkpoints/small-int8/ \
    --tokens 200 \
    --runs 10 \
    --context-sweep      # optional: shows how TBT grows with prompt length
```

**Metrics explained:**

| Metric | Definition |
|--------|-----------|
| TTFT (time-to-first-token) | Latency from prompt → first output token. Includes full prompt prefill. |
| TBT (time-between-tokens) | Per-token latency during generation (excludes first token). |
| Throughput (tok/s) | `1000 / mean_TBT`. The headline number. |
| TBT p95 | 95th-percentile latency — shows tail behavior. |

**Expected ranges on 8th-gen i7 (INT8, small config):**
- TTFT: 200–500 ms (depends on prompt length)
- TBT: 30–70 ms (≈ 15–30 tok/s)

---

### Perplexity

```bash
python -m src.benchmarks.perplexity \
    --checkpoint checkpoints/small-int8/ \
    --data data/raw/test.txt \
    --stride 512
```

**What it means:** Perplexity is `exp(average cross-entropy loss)` over the test set. Lower = better.

| PPL range | Interpretation |
|-----------|---------------|
| < 20 | Excellent — model has strong grasp of the domain |
| 20–50 | Good for a small model on general text |
| 50–100 | Acceptable early in training |
| > 100 | Model is still largely random |

Use `--stride 512` (half the context window) for the standard sliding-window evaluation. Using stride = max_seq_len gives a faster but optimistic estimate.

Track PPL across checkpoints to confirm training is making progress.

---

### Memory

```bash
python -m src.benchmarks.memory \
    --checkpoint checkpoints/small-int8/ \
    --tokens 500
```

Prints RSS (resident set size) sampled every 50 tokens, plus headroom vs 32 GB. If headroom goes negative the model will start swapping — reduce `max_seq_len` or switch to a smaller config.

---

## Comparing Checkpoints

Run `run_all.py` after each major training milestone and save the JSON outputs. Compare PPL and throughput over time:

```bash
# Quick comparison across all saved reports
python - <<'EOF'
import json, glob
for f in sorted(glob.glob("reports/*.json")):
    d = json.load(open(f))
    ppl = d["results"].get("perplexity", {}).get("perplexity", "—")
    tps = d["results"].get("speed", {}).get("throughput_tok_per_sec", "—")
    print(f"{f:60s}  PPL={ppl:>10}  tok/s={tps}")
EOF
```

---

## Adding to `psutil` dependency

`memory.py` requires `psutil`. It is already in `requirements.txt`. Confirm with:

```bash
pip show psutil
```
