"""Unit tests for the transformer model."""

import torch
import pytest

from src.model import LWLM, LWLMConfig


@pytest.fixture
def tiny_cfg():
    return LWLMConfig(
        vocab_size=256,
        n_layers=2,
        n_heads=4,
        d_model=64,
        d_ffn=128,
        max_seq_len=32,
    )


def test_forward_shape(tiny_cfg):
    model = LWLM(tiny_cfg)
    input_ids = torch.randint(0, tiny_cfg.vocab_size, (2, 16))
    logits, kv_caches = model(input_ids)
    assert logits.shape == (2, 16, tiny_cfg.vocab_size)
    assert len(kv_caches) == tiny_cfg.n_layers


def test_kv_cache_consistency(tiny_cfg):
    """Single-step inference with KV cache must match full-sequence output."""
    model = LWLM(tiny_cfg)
    model.eval()
    torch.manual_seed(0)

    seq = torch.randint(0, tiny_cfg.vocab_size, (1, 8))

    with torch.no_grad():
        logits_full, _ = model(seq)
        last_full = logits_full[:, -1, :]

        logits_prefix, kv = model(seq[:, :-1])
        logits_step, _ = model(seq[:, -1:], kv)
        last_cached = logits_step[:, -1, :]

    assert torch.allclose(last_full, last_cached, atol=1e-5), "KV cache output mismatch"


def test_param_count(tiny_cfg):
    model = LWLM(tiny_cfg)
    assert model.param_count() > 0


def test_weight_tying(tiny_cfg):
    model = LWLM(tiny_cfg)
    assert model.lm_head.weight is model.token_emb.weight
