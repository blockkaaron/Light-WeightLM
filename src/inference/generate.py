"""CLI entry point: python -m src.inference.generate"""

from __future__ import annotations

import argparse
from pathlib import Path

from .engine import InferenceEngine


def main():
    parser = argparse.ArgumentParser(description="Generate text with LWLM")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default="Hello")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    engine = InferenceEngine(args.checkpoint, device=args.device)

    print(f"\n--- LWLM ---\nPrompt: {args.prompt}\n")
    for token_text in engine.stream(
        args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    ):
        print(token_text, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    main()
