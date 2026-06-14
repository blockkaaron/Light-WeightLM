# Hardware & Performance

## Target Platform

| Component | Spec |
|-----------|------|
| CPU | Intel Core i7-8700 (6C/12T, 3.2–4.6 GHz) or better |
| RAM | 32 GB DDR4-2666 dual-channel |
| GPU | None (optional: any CUDA-capable card with ≥ 8 GB VRAM) |
| OS | Windows 10 64-bit / Ubuntu 20.04+ |
| Python | 3.11+ |

## Memory Budget (32 GB)

```
OS + background processes:   ~4–6 GB
Python runtime + PyTorch:    ~1–2 GB
Model weights (INT8 small):  ~250 MB
KV cache (2048 ctx, 12 head): ~200 MB
Batch buffers & overhead:    ~500 MB
                             --------
Total (small, INT8):         ~8 GB peak  ← well within 32 GB
```

Even `medium` (350M INT8) stays under 10 GB peak, leaving comfortable headroom.

## CPU Optimization Notes

- **AVX2** is available on all 8th-gen i7s — PyTorch uses it automatically for BLAS ops
- **OpenMP** thread count: set `OMP_NUM_THREADS` to physical core count (not logical)  
  e.g., `OMP_NUM_THREADS=6` for i7-8700
- **NUMA**: not relevant for desktop chips, skip
- **Intel Extension for PyTorch (IPEX)**: optional drop-in that accelerates CPU inference  
  ~1.3–1.8× speedup over stock PyTorch on Intel hardware

## Optional GPU Path

If a GPU is available (even a budget GTX 1060 6 GB):
- Load model in FP16 onto GPU
- Falls back to CPU automatically for layers that don't fit

The codebase detects `torch.cuda.is_available()` and moves tensors accordingly.

## Benchmarking

Run the built-in benchmark after training:

```bash
python -m src.inference.benchmark \
    --checkpoint checkpoints/small-int8/ \
    --tokens 200 \
    --runs 5
```

Outputs: mean latency, tokens/sec, peak RSS, VRAM used (if GPU).
