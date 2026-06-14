"""
Token dataset backed by memory-mapped binary files.

Each .bin file is a flat uint16 array of token ids produced by tokenize_corpus.py.
Using mmap avoids loading the full corpus into RAM — critical for large datasets.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    """Streams fixed-length token sequences from memory-mapped .bin files."""

    def __init__(self, data_dir: Path, seq_len: int):
        self.seq_len = seq_len
        self.chunks: list[np.ndarray] = []
        total_tokens = 0

        for bin_file in sorted(data_dir.glob("*.bin")):
            arr = np.memmap(bin_file, dtype=np.uint16, mode="r")
            self.chunks.append(arr)
            total_tokens += len(arr)

        if not self.chunks:
            raise RuntimeError(f"No .bin files found in {data_dir}. Run tokenize_corpus.py first.")

        print(f"Dataset: {total_tokens:,} tokens across {len(self.chunks)} file(s)")
        self._total_tokens = total_tokens

    def __len__(self) -> int:
        return (self._total_tokens - 1) // (self.seq_len + 1)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Find which chunk and offset this index maps to
        offset = idx * (self.seq_len + 1)
        for chunk in self.chunks:
            if offset < len(chunk) - self.seq_len - 1:
                tokens = chunk[offset : offset + self.seq_len + 1]
                return torch.from_numpy(tokens.astype(np.int64))
            offset -= len(chunk)
        # Fallback: wrap around to start
        tokens = self.chunks[0][: self.seq_len + 1]
        return torch.from_numpy(tokens.astype(np.int64))
