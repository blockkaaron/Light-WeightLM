# Changelog

All notable changes to Light-WeightLM are documented here.  
Format: `## [vX.Y.Z] - YYYY-MM-DD` followed by sections: **Added**, **Changed**, **Fixed**, **Removed**.

---

## [Unreleased]

## [0.1.1] - 2026-06-14

### Added
- Full benchmark suite (`src/benchmarks/`) — speed, perplexity, memory, and aggregate report
- `docs/benchmarking.md` — guide covering all benchmark modes and how to compare checkpoints
- `psutil` dependency for RSS memory tracking
- `tests/test_benchmarks.py` — smoke tests for all three benchmark modules

### Changed
- `src/inference/benchmark.py` replaced with redirect shim pointing to new suite
- `README.md` updated to reference benchmarking docs
- Project renamed to Light-WeightLM throughout

## [0.1.0] - 2026-06-14

### Added
- Initial project scaffold
- Core transformer model skeleton (`src/model/`)
- BPE tokenizer skeleton (`src/tokenizer/`)
- Training loop skeleton (`src/training/`)
- CPU-optimized inference engine skeleton (`src/inference/`)
- Base configuration system (`configs/`)
- Project documentation (`docs/`)
- Hardware profile targeting 8th-gen i7 / 32 GB RAM / no GPU
