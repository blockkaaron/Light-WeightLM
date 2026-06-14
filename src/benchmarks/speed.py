"""
Speed benchmark: latency, throughput, first-token vs subsequent-token timing.

Measures:
  - Time-to-first-token (TTFT) — latency the user feels before output starts
  - Time-between-tokens (TBT)  — per-token latency during generation
  - Throughput (tok/s)         — sustained generation speed
  - Context-length scaling     — how speed degrades as prompt grows

Usage:
    python -m src.benchmarks.speed \
        --checkpoint checkpoints/small-int8/ \
        --tokens 200 \
        --runs 10 \
        --context-sweep
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from statistics import mean, median, stdev, quantiles

import torch

from src.inference.engine import InferenceEngine, _sample


def _time_generation(engine: InferenceEngine, prompt_ids: list[int], max_new: int):
    """Returns (ttft_ms, per_token_ms_list)."""
    model = engine.model
    device = engine.device
    tokenizer = engine.tokenizer

    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    kv_caches = None
    per_token: list[float] = []

    model.eval()
    with torch.no_grad():
        for i in range(max_new):
            t0 = time.perf_counter()
            logits, kv_caches = model(tokens, kv_caches)
            next_id = _sample(logits[:, -1, :], temperature=0.0, top_k=0, top_p=1.0)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            per_token.append(elapsed_ms)

            if next_id == tokenizer.eos_id:
                break
            tokens = torch.tensor([[next_id]], dtype=torch.long, device=device)

    ttft = per_token[0] if per_token else 0.0
    tbt = per_token[1:] if len(per_token) > 1 else per_token
    return ttft, tbt


def run_speed_benchmark(
    engine: InferenceEngine,
    prompt: str,
    max_new_tokens: int = 200,
    n_runs: int = 10,
) -> dict:
    tokenizer = engine.tokenizer
    prompt_ids = tokenizer.encode(prompt, add_bos=True)

    ttfts, tbts_all = [], []
    for run in range(n_runs):
        torch.manual_seed(run)
        ttft, tbt = _time_generation(engine, prompt_ids, max_new_tokens)
        ttfts.append(ttft)
        tbts_all.extend(tbt)

    def _stats(vals):
        if not vals:
            return {}
        s = {"mean": mean(vals), "median": median(vals)}
        if len(vals) >= 4:
            s["p95"] = quantiles(vals, n=20)[18]
            s["p99"] = quantiles(vals, n=100)[98]
        if len(vals) >= 2:
            s["stdev"] = stdev(vals)
        return {k: round(v, 2) for k, v in s.items()}

    tbt_stats = _stats(tbts_all)
    throughput = round(1000 / tbt_stats["mean"], 2) if tbt_stats.get("mean") else 0

    return {
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": max_new_tokens,
        "n_runs": n_runs,
        "ttft_ms": _stats(ttfts),
        "tbt_ms": tbt_stats,
        "throughput_tok_per_sec": throughput,
    }


def run_context_sweep(engine: InferenceEngine, max_new_tokens: int = 50) -> list[dict]:
    """Measure TBT at increasing prompt lengths to show KV-cache scaling."""
    base = "The quick brown fox jumped over the lazy dog. "
    results = []
    for ctx_len in [32, 64, 128, 256, 512, 768, 1024]:
        if ctx_len > engine.model_cfg.max_seq_len - max_new_tokens:
            break
        # Build a prompt that's approximately ctx_len tokens
        prompt = (base * (ctx_len // len(base.split()) + 1))[:ctx_len * 4]  # rough char estimate
        prompt_ids = engine.tokenizer.encode(prompt, add_bos=True)[:ctx_len]

        _, tbt = _time_generation(engine, prompt_ids, max_new_tokens)
        avg_tbt = round(mean(tbt), 2) if tbt else 0
        results.append({"prompt_tokens": len(prompt_ids), "avg_tbt_ms": avg_tbt, "tok_per_sec": round(1000 / avg_tbt, 1) if avg_tbt else 0})

    return results


def main():
    parser = argparse.ArgumentParser(description="Speed benchmark for Light-WeightLM")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default="The history of artificial intelligence began")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--context-sweep", action="store_true", help="Also sweep over prompt lengths")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    engine = InferenceEngine(args.checkpoint, device=args.device)

    print(f"Speed benchmark — {args.runs} runs × {args.tokens} tokens\n")
    result = run_speed_benchmark(engine, args.prompt, args.tokens, args.runs)

    print(f"Prompt tokens      : {result['prompt_tokens']}")
    print(f"Time-to-first-token: {result['ttft_ms']}")
    print(f"Time-between-tokens: {result['tbt_ms']} ms")
    print(f"Throughput         : {result['throughput_tok_per_sec']} tok/s")

    if args.context_sweep:
        print("\nContext-length scaling:")
        print(f"  {'Prompt tokens':>14} | {'Avg TBT (ms)':>12} | {'tok/s':>8}")
        print("  " + "-" * 40)
        for row in run_context_sweep(engine, max_new_tokens=50):
            print(f"  {row['prompt_tokens']:>14} | {row['avg_tbt_ms']:>12} | {row['tok_per_sec']:>8}")


if __name__ == "__main__":
    main()
