# Configuration Reference

All config files live in `configs/` as YAML. Load with OmegaConf.

## Model Config (`model.*`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `model.n_layers` | int | 12 | Number of transformer layers |
| `model.n_heads` | int | 12 | Number of attention heads |
| `model.d_model` | int | 768 | Hidden / embedding dimension |
| `model.d_ffn` | int | 2048 | FFN inner dimension (SwiGLU) |
| `model.vocab_size` | int | 8192 | Must match tokenizer |
| `model.max_seq_len` | int | 1024 | Maximum context length |
| `model.dropout` | float | 0.0 | Dropout (0 for inference; 0.1 for training) |
| `model.rope_theta` | float | 10000.0 | RoPE base frequency |

## Training Config (`train.*`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `train.batch_size` | int | 4 | Micro-batch size |
| `train.grad_accum` | int | 4 | Gradient accumulation steps (effective batch = 16) |
| `train.lr` | float | 3e-4 | Peak learning rate |
| `train.lr_schedule` | str | `cosine` | `cosine` or `constant` |
| `train.warmup_steps` | int | 1000 | Linear warmup duration |
| `train.max_steps` | int | 100000 | Total training steps |
| `train.grad_clip` | float | 1.0 | Gradient clipping norm |
| `train.weight_decay` | float | 0.1 | AdamW weight decay |
| `train.dtype` | str | `float32` | `float32` or `bfloat16` |
| `train.save_every` | int | 1000 | Checkpoint interval (steps) |
| `train.seed` | int | 42 | Global RNG seed |

## Inference Config (`inference.*`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `inference.max_new_tokens` | int | 200 | Max tokens to generate |
| `inference.temperature` | float | 0.8 | Sampling temperature |
| `inference.top_k` | int | 0 | Top-k (0 = disabled) |
| `inference.top_p` | float | 0.9 | Nucleus sampling |
| `inference.repetition_penalty` | float | 1.0 | 1.0 = no penalty |
| `inference.device` | str | `cpu` | `cpu` or `cuda` |
| `inference.dtype` | str | `int8` | Quantization mode |

## Predefined Configs

| File | Params | Use Case |
|------|--------|----------|
| `configs/tiny.yaml` | ~20M | Unit tests, fast iteration |
| `configs/small.yaml` | ~125M | Primary dev target |
| `configs/medium.yaml` | ~350M | Better quality, still fits 32 GB |
