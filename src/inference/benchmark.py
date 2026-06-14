"""Benchmark inference throughput: python -m src.inference.benchmark"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from statistics import mean, median, quantiles

import psutil
import torch

from .engine import InferenceEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default="The quick brown fox")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    engine = InferenceEngine(args.checkpoint, device=args.device)
    process = psutil.Process()

    latencies: list[float] = []
    for run in range(args.runs):
        torch.manual_seed(run)
        mem_before = process.memory_info().rss / 1024 ** 2
        t0 = time.perf_counter()
        _ = engine.generate(args.prompt, max_new_tokens=args.tokens, temperature=0.0)
        elapsed = time.perf_counter() - t0
        mem_after = process.memory_info().rss / 1024 ** 2
        ms_per_tok = elapsed * 1000 / args.tokens
        latencies.append(ms_per_tok)
        print(f"Run {run + 1}: {ms_per_tok:.1f} ms/tok  |  {1000/ms_per_tok:.1f} tok/s  |  RSS {mem_after:.0f} MB")

    print(f"\nSummary ({args.runs} runs, {args.tokens} tokens each):")
    print(f"  Mean:   {mean(latencies):.1f} ms/tok  ({1000/mean(latencies):.1f} tok/s)")
    print(f"  Median: {median(latencies):.1f} ms/tok")
    if args.runs >= 4:
        p95 = quantiles(latencies, n=20)[18]
        print(f"  p95:    {p95:.1f} ms/tok")


if __name__ == "__main__":
    main()
