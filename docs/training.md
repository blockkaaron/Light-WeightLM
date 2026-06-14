# Training Guide

## Overview

Training proceeds in three stages:

1. **Tokenizer training** — learn vocabulary from your corpus
2. **Pre-training** — next-token prediction on raw text
3. **Fine-tuning** (optional) — task-specific or instruction tuning

## 1. Prepare Your Data

Place raw `.txt` files in `data/raw/`. The pipeline reads UTF-8 text.

```
data/
└── raw/
    ├── corpus_part1.txt
    ├── corpus_part2.txt
    └── ...
```

Large binary files (`.jsonl`, `.parquet`) are gitignored — document their sources in `data/README.md`.

## 2. Train the Tokenizer

```bash
python -m src.tokenizer.train \
    --data data/raw/ \
    --vocab-size 8192 \
    --output tokenizer/
```

This runs BPE training and writes:
- `tokenizer/vocab.json`
- `tokenizer/merges.txt`

These are also gitignored (large binary blobs). Bundle them manually for distribution.

## 3. Tokenize the Corpus

```bash
python -m src.training.tokenize_corpus \
    --tokenizer tokenizer/ \
    --data data/raw/ \
    --output data/tokenized/ \
    --seq-len 1024
```

Produces memory-mapped `.bin` files that the DataLoader streams without loading everything into RAM.

## 4. Pre-Training

```bash
python -m src.training.train \
    --config configs/small.yaml \
    --data data/tokenized/ \
    --output checkpoints/small-pretrain/
```

Key config options in `configs/small.yaml`:

| Option | Default | Notes |
|--------|---------|-------|
| `model.n_layers` | 12 | Transformer depth |
| `model.n_heads` | 12 | Attention heads |
| `model.d_model` | 768 | Hidden dimension |
| `model.d_ffn` | 2048 | FFN hidden dimension |
| `model.vocab_size` | 8192 | Must match tokenizer |
| `model.max_seq_len` | 1024 | Context window |
| `train.batch_size` | 4 | Reduce if OOM on 32 GB |
| `train.lr` | 3e-4 | AdamW learning rate |
| `train.warmup_steps` | 1000 | Linear warmup |
| `train.grad_clip` | 1.0 | Gradient clipping |
| `train.dtype` | `float32` | Use `bfloat16` if GPU available |

## 5. Checkpointing

Checkpoints are saved every N steps (configurable). Structure:

```
checkpoints/small-pretrain/
├── step-001000/
│   ├── model.safetensors
│   ├── optimizer.pt
│   └── config.yaml
└── latest -> step-001000/   (symlink)
```

Resume from a checkpoint:

```bash
python -m src.training.train \
    --config configs/small.yaml \
    --resume checkpoints/small-pretrain/latest/
```

## 6. Quantization (post-training)

After pre-training, produce an INT8 version for CPU inference:

```bash
python -m src.inference.quantize \
    --checkpoint checkpoints/small-pretrain/latest/ \
    --calibration data/raw/calibration.txt \
    --output checkpoints/small-int8/
```

## Training Time Estimates (CPU only, 8th-gen i7)

These are rough estimates for the `small` config (125M params):

| Corpus size | Steps | Est. time |
|-------------|-------|-----------|
| 100 MB text | 10K   | ~6 hours  |
| 1 GB text   | 100K  | ~60 hours |

CPU training is slow — use for prototyping/research. For real pre-training, use a GPU machine then transfer weights.
