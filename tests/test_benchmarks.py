"""Smoke tests for the benchmark suite (no checkpoint required)."""

import math
import torch
import pytest

from src.model import BloxSLM, BloxSLMConfig
from src.tokenizer.bpe import BPETokenizer, SPECIAL_TOKENS


def _tiny_engine():
    """Build a minimal InferenceEngine-like object without loading from disk."""
    from src.inference.engine import InferenceEngine
    import types

    cfg = BloxSLMConfig(
        vocab_size=256, n_layers=2, n_heads=4, d_model=64, d_ffn=128, max_seq_len=64
    )
    model = BloxSLM(cfg)
    model.eval()

    chars = list("abcdefghijklmnopqrstuvwxyz .,")
    vocab = dict(SPECIAL_TOKENS)
    nid = len(vocab)
    for ch in chars:
        if ch not in vocab:
            vocab[ch] = nid; nid += 1
    tokenizer = BPETokenizer(vocab=vocab, merges=[])

    engine = types.SimpleNamespace(
        model=model,
        model_cfg=cfg,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
    )
    return engine


def test_perplexity_runs():
    from src.benchmarks.perplexity import compute_perplexity
    engine = _tiny_engine()
    result = compute_perplexity(engine, "hello world foo bar baz", stride=16, max_tokens=20)
    assert "perplexity" in result
    assert result["perplexity"] > 1.0
    assert math.isfinite(result["perplexity"])


def test_speed_runs():
    from src.benchmarks.speed import run_speed_benchmark
    engine = _tiny_engine()
    result = run_speed_benchmark(engine, "hello world", max_new_tokens=5, n_runs=2)
    assert result["throughput_tok_per_sec"] > 0
    assert "ttft_ms" in result
    assert "tbt_ms" in result


def test_memory_runs():
    from src.benchmarks.memory import run_memory_benchmark
    engine = _tiny_engine()
    result = run_memory_benchmark(engine, "hello world", max_new_tokens=10)
    assert result["peak_rss_mb"] >= result["baseline_rss_mb"]
    assert "headroom_mb" in result
