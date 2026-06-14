# BloxSLM

A small language model built from scratch, designed for **air-gapped, low-resource deployment**.

## Target Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | Intel 8th-gen i7 (8 cores) | Intel 10th-gen i7+ or AMD Ryzen 7+ |
| RAM | 32 GB DDR4 | 32–64 GB DDR4/DDR5 |
| GPU | None (CPU-only mode) | NVIDIA with ≥ 8 GB VRAM (optional) |
| Storage | 50 GB SSD | 200 GB NVMe SSD |
| OS | Windows 10 / Ubuntu 20.04 | Ubuntu 22.04 LTS |

> **Air-gapped**: BloxSLM is designed to run with **zero internet access** post-installation. All weights, tokenizer assets, and dependencies are bundled locally.

## Architecture Overview

BloxSLM uses a decoder-only transformer architecture tuned for CPU inference:

- **Parameters**: ~125M–350M (configurable via `configs/`)
- **Attention**: Multi-head attention with optional sliding window for memory efficiency
- **Quantization**: INT8 weight quantization for CPU inference (no accuracy loss at this scale)
- **Context length**: 1024–2048 tokens (tunable)
- **Tokenizer**: Byte-Pair Encoding (BPE) trained from scratch

## Project Structure

```
BloxSLM/
├── src/
│   ├── model/          # Transformer architecture
│   ├── tokenizer/      # BPE tokenizer
│   ├── training/       # Training loop & data pipeline
│   ├── inference/      # CPU-optimized inference engine
│   └── utils/          # Shared utilities
├── configs/            # YAML configuration files
├── data/               # Data preparation (raw data excluded from git)
├── docs/               # In-depth documentation
├── tests/              # Unit & integration tests
├── requirements.txt    # Python dependencies
└── CHANGELOG.md        # Full change history
```

## Quick Start

```bash
# 1. Clone
git clone https://github.com/Bloxware/BloxSLM.git
cd BloxSLM

# 2. Create virtual environment (Python 3.11+)
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.\.venv\Scripts\Activate.ps1     # Windows PowerShell

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the tokenizer on your corpus
python -m src.tokenizer.train --data data/raw/ --vocab-size 8192

# 5. Train the model
python -m src.training.train --config configs/small.yaml

# 6. Run inference
python -m src.inference.generate --checkpoint checkpoints/latest/ --prompt "Hello"
```

## Documentation

- [Architecture](docs/architecture.md) — model design decisions and rationale
- [Hardware & Performance](docs/hardware.md) — profiling on target hardware
- [Training Guide](docs/training.md) — data prep, tokenizer training, model training
- [Inference Guide](docs/inference.md) — deployment, quantization, benchmarks
- [Configuration Reference](docs/config-reference.md) — all config options explained

## Development

```bash
# Run tests
pytest tests/

# Type check
mypy src/

# Lint
ruff check src/
```

## License

Private — all rights reserved. See repository settings.
