"""
Perplexity benchmark.

Perplexity = exp(mean cross-entropy loss over held-out tokens).
Lower is better. The primary quality metric for a language model —
use this to compare checkpoints and track training progress.

Usage:
    python -m src.benchmarks.perplexity \
        --checkpoint checkpoints/small-int8/ \
        --data data/raw/test.txt \
        --stride 512
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.inference.engine import InferenceEngine
from src.tokenizer import BPETokenizer


def compute_perplexity(
    engine: InferenceEngine,
    text: str,
    stride: int = 512,
    max_tokens: int | None = None,
) -> dict:
    """
    Sliding-window perplexity over `text`.

    Uses a stride < max_seq_len so every token (except the first window)
    has full left context. Matches the standard PPL evaluation methodology.
    """
    tokenizer = engine.tokenizer
    model = engine.model
    device = engine.device
    max_len = engine.model_cfg.max_seq_len

    token_ids = tokenizer.encode(text, add_bos=True)
    if max_tokens:
        token_ids = token_ids[:max_tokens]

    if len(token_ids) < 2:
        raise ValueError("Text too short to evaluate perplexity (need at least 2 tokens).")

    total_nll = 0.0
    total_tokens = 0
    n_windows = 0

    model.eval()
    with torch.no_grad():
        for begin in tqdm(range(0, len(token_ids) - 1, stride), desc="PPL windows", unit="win"):
            end = min(begin + max_len, len(token_ids))
            window = torch.tensor([token_ids[begin:end]], dtype=torch.long, device=device)

            logits, _ = model(window[:, :-1])

            # Only score tokens after the first stride position (avoid double-counting)
            target_start = 0 if begin == 0 else stride
            target_ids = window[:, 1:][:, target_start:]
            pred_logits = logits[:, target_start:, :]

            nll = F.cross_entropy(
                pred_logits.reshape(-1, pred_logits.size(-1)),
                target_ids.reshape(-1),
                reduction="sum",
            )
            total_nll += nll.item()
            total_tokens += target_ids.numel()
            n_windows += 1

            if end >= len(token_ids):
                break

    avg_nll = total_nll / total_tokens
    ppl = math.exp(avg_nll)

    return {
        "perplexity": round(ppl, 4),
        "avg_nll": round(avg_nll, 6),
        "total_tokens": total_tokens,
        "n_windows": n_windows,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute perplexity on a text file")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help=".txt file to evaluate on")
    parser.add_argument("--stride", type=int, default=512, help="Sliding window stride")
    parser.add_argument("--max-tokens", type=int, default=None, help="Cap token count for fast eval")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    engine = InferenceEngine(args.checkpoint, device=args.device)
    text = args.data.read_text(encoding="utf-8", errors="replace")

    print(f"Evaluating perplexity on {args.data} ...")
    result = compute_perplexity(engine, text, stride=args.stride, max_tokens=args.max_tokens)

    print(f"\nPerplexity  : {result['perplexity']}")
    print(f"Avg NLL     : {result['avg_nll']}")
    print(f"Tokens eval : {result['total_tokens']:,}")
    print(f"Windows     : {result['n_windows']}")


if __name__ == "__main__":
    main()
