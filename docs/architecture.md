# Architecture

## Design Goals

1. **Air-gapped**: zero network calls at runtime; all assets ship with the model
2. **CPU-first**: primary target is 8th-gen i7 with no discrete GPU
3. **Memory-bounded**: peak RSS ≤ 24 GB during inference (leaves 8 GB for OS)
4. **Deterministic output**: reproducible inference given the same seed

## Model Family

LWLM is a **decoder-only transformer** (autoregressive, left-to-right).  
This matches GPT-2 / LLaMA lineage and is the most mature architecture for text generation.

### Parameter Targets

| Config | Params | Approx RAM (FP32) | Approx RAM (INT8) |
|--------|--------|--------------------|-------------------|
| `tiny`   | 20M  | ~80 MB             | ~40 MB            |
| `small`  | 125M | ~500 MB            | ~250 MB           |
| `medium` | 350M | ~1.4 GB            | ~700 MB           |
| `large`  | 750M | ~3 GB              | ~1.5 GB           |

Start with `small` (125M). It fits easily in 32 GB and inference is fast on CPU.

## Transformer Block

Each layer contains:
1. **RMSNorm** (pre-norm) — more stable than LayerNorm, faster on CPU
2. **Multi-Head Self-Attention** — scaled dot-product, optional causal mask
3. **SwiGLU Feed-Forward Network** — better than GELU in practice
4. **Residual connections** around both sub-layers

```
x → RMSNorm → MHA → x (residual)
x → RMSNorm → FFN → x (residual)
```

### Attention

- Standard multi-head attention (not GQA yet — keep it simple)
- Causal mask for autoregressive decoding
- No positional embeddings — use **RoPE** (Rotary Position Embedding) instead
  - RoPE is baked into Q/K projections, not added to embeddings
  - Generalizes better to longer sequences than sinusoidal embeddings

### Feed-Forward

SwiGLU: `FFN(x) = (xW₁ ⊙ SiLU(xW₃)) W₂`  
Hidden dim = 4/3 × model_dim (keeps param count equal to classic 4×FFN).

## Tokenizer

Byte-Pair Encoding (BPE) trained on the target corpus.

- Vocab size: 8192–16384 (small vocab = faster embedding lookups on CPU)
- Byte fallback: every byte is representable, so unknown characters never fail
- Special tokens: `<|bos|>`, `<|eos|>`, `<|pad|>`

## Quantization Strategy

For CPU inference, INT8 post-training quantization (PTQ) is the primary path:

1. Train in FP32 / BF16
2. Calibrate on a representative dataset (≈500 samples)
3. Quantize linear layers to INT8 (weights only, or weights + activations)
4. PyTorch `torch.ao.quantization` or `bitsandbytes` CPU backend

Expected throughput on 8th-gen i7 at INT8:
- `small` (125M): ~15–30 tokens/sec
- `medium` (350M): ~6–12 tokens/sec

## Inference Engine

See [docs/inference.md](inference.md) for the full breakdown.

Key decisions:
- **KV cache**: pre-allocate a fixed-size tensor; avoid Python-level list growth
- **Greedy / top-k / top-p sampling**: configurable at runtime
- **Batch size**: default 1 for interactive use; up to 4 on 32 GB with `small`
