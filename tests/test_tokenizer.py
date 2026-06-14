"""Unit tests for the BPE tokenizer."""

import pytest
from src.tokenizer.bpe import BPETokenizer, SPECIAL_TOKENS


def _make_tiny_tokenizer() -> BPETokenizer:
    """Build a minimal tokenizer without running BPE training."""
    chars = list("abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ.,!?")
    vocab = dict(SPECIAL_TOKENS)
    next_id = len(vocab)
    for ch in chars:
        if ch not in vocab:
            vocab[ch] = next_id
            next_id += 1
    # No merges — character-level only
    return BPETokenizer(vocab=vocab, merges=[])


def test_encode_decode_roundtrip():
    tok = _make_tiny_tokenizer()
    text = "Hello world"
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    assert "Hello" in decoded or "hello" in decoded.lower()


def test_special_token_ids():
    tok = _make_tiny_tokenizer()
    assert tok.bos_id == SPECIAL_TOKENS["<|bos|>"]
    assert tok.eos_id == SPECIAL_TOKENS["<|eos|>"]
    assert tok.pad_id == SPECIAL_TOKENS["<|pad|>"]


def test_encode_adds_bos_eos():
    tok = _make_tiny_tokenizer()
    ids = tok.encode("hi", add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id


def test_vocab_size():
    tok = _make_tiny_tokenizer()
    assert len(tok) > 0
