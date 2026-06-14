"""Train a BPE tokenizer from a directory of .txt files."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

from .bpe import BPETokenizer, SPECIAL_TOKENS, _pre_tokenize


def _count_pairs(vocab_counts: dict[tuple, int]) -> dict[tuple, int]:
    pairs: dict[tuple, int] = defaultdict(int)
    for word, freq in vocab_counts.items():
        symbols = list(word)
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def _merge_vocab(pair: tuple[str, str], vocab_counts: dict[tuple, int]) -> dict[tuple, int]:
    merged = pair[0] + pair[1]
    new_vocab: dict[tuple, int] = {}
    for word, freq in vocab_counts.items():
        symbols = list(word)
        i = 0
        new_symbols = []
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                new_symbols.append(merged)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_vocab[tuple(new_symbols)] = freq
    return new_vocab


def train_bpe(corpus_dir: Path, vocab_size: int, output_dir: Path):
    print(f"Reading corpus from {corpus_dir} ...")

    # Count word frequencies
    word_freq: dict[str, int] = defaultdict(int)
    for txt_file in sorted(corpus_dir.glob("**/*.txt")):
        text = txt_file.read_text(encoding="utf-8", errors="replace")
        for word in _pre_tokenize(text):
            word_freq[word] += 1

    print(f"  {len(word_freq):,} unique word-forms")

    # Start with character-level vocab (bytes)
    # Represent each word as a tuple of characters
    vocab_counts: dict[tuple, int] = {tuple(w): c for w, c in word_freq.items()}

    # Seed vocab with all characters seen
    char_vocab: set[str] = set()
    for word in vocab_counts:
        char_vocab.update(word)

    vocab: dict[str, int] = dict(SPECIAL_TOKENS)
    next_id = len(vocab)
    for ch in sorted(char_vocab):
        if ch not in vocab:
            vocab[ch] = next_id
            next_id += 1

    merges: list[tuple[str, str]] = []
    n_merges = vocab_size - len(vocab)

    print(f"Running {n_merges} BPE merges (target vocab size {vocab_size}) ...")
    for i in range(n_merges):
        pairs = _count_pairs(vocab_counts)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        merges.append(best)
        merged_token = best[0] + best[1]
        vocab[merged_token] = next_id
        next_id += 1
        vocab_counts = _merge_vocab(best, vocab_counts)

        if (i + 1) % 500 == 0:
            print(f"  merge {i + 1}/{n_merges}")

    tokenizer = BPETokenizer(vocab=vocab, merges=merges)
    tokenizer.save(output_dir)
    print(f"Tokenizer saved to {output_dir}  (vocab size: {len(vocab):,})")


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer")
    parser.add_argument("--data", type=Path, required=True, help="Directory of .txt corpus files")
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--output", type=Path, default=Path("tokenizer/"))
    args = parser.parse_args()
    train_bpe(args.data, args.vocab_size, args.output)


if __name__ == "__main__":
    main()
