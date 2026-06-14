"""
Memory benchmark: peak RSS and KV-cache growth over context length.

Tracks how memory usage scales as generation continues — critical for
confirming the model stays within 32 GB on the target hardware.

Usage:
    python -m src.benchmarks.memory \
        --checkpoint checkpoints/small-int8/ \
        --tokens 500
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import psutil
import torch

from src.inference.engine import InferenceEngine, _sample


def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1024 ** 2


def run_memory_benchmark(engine: InferenceEngine, prompt: str, max_new_tokens: int = 500) -> dict:
    model = engine.model
    device = engine.device
    tokenizer = engine.tokenizer

    gc.collect()
    torch.cuda.empty_cache() if device.type == "cuda" else None
    baseline_mb = _rss_mb()

    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    kv_caches = None

    samples: list[dict] = []
    peak_mb = baseline_mb

    model.eval()
    with torch.no_grad():
        for step in range(max_new_tokens):
            logits, kv_caches = model(tokens, kv_caches)
            next_id = _sample(logits[:, -1, :], temperature=0.0, top_k=0, top_p=1.0)

            if next_id == tokenizer.eos_id:
                break

            tokens = torch.tensor([[next_id]], dtype=torch.long, device=device)

            # Sample memory every 50 steps
            if step % 50 == 0 or step == max_new_tokens - 1:
                rss = _rss_mb()
                peak_mb = max(peak_mb, rss)
                samples.append({"token": step + 1, "rss_mb": round(rss, 1)})

    # KV cache theoretical size
    cfg = engine.model_cfg
    kv_bytes = (
        cfg.n_layers * 2 * 1 * max_new_tokens * cfg.n_heads * cfg.head_dim() * 4
    )  # float32

    return {
        "baseline_rss_mb": round(baseline_mb, 1),
        "peak_rss_mb": round(peak_mb, 1),
        "delta_mb": round(peak_mb - baseline_mb, 1),
        "kv_cache_theoretical_mb": round(kv_bytes / 1024 ** 2, 2),
        "samples": samples,
        "headroom_mb": round(32768 - peak_mb, 1),  # vs 32 GB target
    }


def main():
    parser = argparse.ArgumentParser(description="Memory benchmark for Light-WeightLM")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default="Once upon a time in a land far away")
    parser.add_argument("--tokens", type=int, default=500)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    engine = InferenceEngine(args.checkpoint, device=args.device)

    print(f"Memory benchmark — {args.tokens} tokens of generation\n")
    result = run_memory_benchmark(engine, args.prompt, args.tokens)

    print(f"Baseline RSS   : {result['baseline_rss_mb']} MB")
    print(f"Peak RSS       : {result['peak_rss_mb']} MB")
    print(f"Delta          : +{result['delta_mb']} MB")
    print(f"KV cache (est) : {result['kv_cache_theoretical_mb']} MB")
    print(f"Headroom vs 32G: {result['headroom_mb']} MB")
    print()
    print(f"  {'Token':>8} | {'RSS (MB)':>10}")
    print("  " + "-" * 22)
    for s in result["samples"]:
        print(f"  {s['token']:>8} | {s['rss_mb']:>10}")


if __name__ == "__main__":
    main()
