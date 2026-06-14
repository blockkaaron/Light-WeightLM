# Inference Guide

## Running Inference

```bash
python -m src.inference.generate \
    --checkpoint checkpoints/small-int8/ \
    --prompt "The capital of France is" \
    --max-tokens 200 \
    --temperature 0.8 \
    --top-p 0.9
```

## Sampling Strategies

| Strategy | Flag | Notes |
|----------|------|-------|
| Greedy | `--temperature 0.0` | Deterministic; picks highest-prob token |
| Top-k | `--top-k 40` | Sample from top 40 candidates |
| Top-p (nucleus) | `--top-p 0.9` | Sample from tokens summing to p=0.9 |
| Temperature | `--temperature 0.8` | Scales logits before softmax |

For factual queries use `temperature 0.2`. For creative tasks use `temperature 0.9`.

## KV Cache

The inference engine pre-allocates a KV cache tensor on startup:

```
cache shape: [n_layers, 2, batch_size, max_seq_len, n_heads, head_dim]
```

This avoids repeated computation of past tokens. At `small` + context 1024:
- KV cache size ≈ 12 × 2 × 1 × 1024 × 12 × 64 × 4 bytes ≈ **72 MB**

## Interactive Shell

```bash
python -m src.inference.shell --checkpoint checkpoints/small-int8/
```

Starts a REPL loop. Type `exit` or Ctrl-C to quit.

## Quantization Modes

| Mode | File size | Speed | Quality |
|------|-----------|-------|---------|
| FP32 | ~500 MB | 1× | Baseline |
| FP16 | ~250 MB | 1.2× CPU | Same |
| INT8 (weights only) | ~130 MB | 1.5–2× | ≈FP32 |
| INT8 (weights + activations) | ~130 MB | 2–3× | Slight loss |

Recommended for air-gapped CPU deployment: **INT8 weights-only**.

## Benchmarking

```bash
python -m src.inference.benchmark \
    --checkpoint checkpoints/small-int8/ \
    --prompt "Benchmark prompt here" \
    --tokens 200 \
    --runs 10
```

Outputs:
- Mean / median / p95 latency per token (ms)
- Tokens per second
- Peak RSS (MB)

## Air-Gap Bundle

To create a fully self-contained bundle for offline deployment:

```bash
python -m src.inference.bundle \
    --checkpoint checkpoints/small-int8/ \
    --tokenizer tokenizer/ \
    --output dist/bloxslm-bundle/
```

The bundle contains:
- `model.safetensors` — quantized weights
- `tokenizer.model` — BPE tokenizer
- `config.yaml` — all hyperparameters
- `generate.py` — standalone inference script (no extra imports beyond PyTorch)
- `requirements-runtime.txt` — minimal deps for inference only
