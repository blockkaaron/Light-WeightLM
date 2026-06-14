"""
Byte-Pair Encoding tokenizer.

Trained from scratch on a corpus. Vocab sizes 8192–16384 work well for
small models — larger vocab = larger embedding table = more RAM with no
quality benefit at this parameter scale.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterator

# GPT-2 pre-tokenization pattern (split on whitespace/punctuation before BPE)
_GPT2_PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+""",
    re.UNICODE,
)

SPECIAL_TOKENS = {"<|bos|>": 0, "<|eos|>": 1, "<|pad|>": 2}


class BPETokenizer:
    """Minimal BPE tokenizer compatible with training from scratch."""

    def __init__(self, vocab: dict[str, int], merges: list[tuple[str, str]]):
        self.vocab = vocab
        self.merges = {pair: i for i, pair in enumerate(merges)}
        self.inv_vocab = {v: k for k, v in vocab.items()}
        # Special token ids
        self.bos_id = vocab.get("<|bos|>", 0)
        self.eos_id = vocab.get("<|eos|>", 1)
        self.pad_id = vocab.get("<|pad|>", 2)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        tokens: list[int] = []
        if add_bos:
            tokens.append(self.bos_id)
        for word in _pre_tokenize(text):
            word_tokens = self._encode_word(word)
            tokens.extend(word_tokens)
        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        pieces = []
        for i in ids:
            token = self.inv_vocab.get(i, "")
            if skip_special and token in SPECIAL_TOKENS:
                continue
            pieces.append(token)
        # BPE uses Ġ (U+0120) as a space prefix — replace back to space
        return "".join(pieces).replace("Ġ", " ")

    def _encode_word(self, word: str) -> list[int]:
        chars = list(word)
        # Keep merging the highest-priority pair until none remain
        while len(chars) > 1:
            pairs = [(chars[i], chars[i + 1]) for i in range(len(chars) - 1)]
            best = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if best not in self.merges:
                break
            merged = best[0] + best[1]
            new_chars = []
            i = 0
            while i < len(chars):
                if i < len(chars) - 1 and (chars[i], chars[i + 1]) == best:
                    new_chars.append(merged)
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            chars = new_chars
        return [self.vocab.get(c, self.vocab.get("<|unk|>", 3)) for c in chars]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, directory: str | Path):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        with open(directory / "merges.txt", "w", encoding="utf-8") as f:
            f.write("#version: 0.2\n")
            for a, b in sorted(self.merges, key=lambda x: self.merges[x]):
                f.write(f"{a} {b}\n")

    @classmethod
    def load(cls, directory: str | Path) -> "BPETokenizer":
        directory = Path(directory)
        with open(directory / "vocab.json", encoding="utf-8") as f:
            vocab = json.load(f)
        merges = []
        with open(directory / "merges.txt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    merges.append((parts[0], parts[1]))
        return cls(vocab=vocab, merges=merges)

    def __len__(self) -> int:
        return len(self.vocab)


# ------------------------------------------------------------------
# Pre-tokenization helper
# ------------------------------------------------------------------

def _pre_tokenize(text: str) -> Iterator[str]:
    """Split text into words before BPE, using GPT-2 pattern."""
    for match in _GPT2_PAT.finditer(text):
        word = match.group()
        # Mark space-prefixed tokens with Ġ
        if word.startswith(" "):
            word = "Ġ" + word[1:]
        yield word
