"""
Tokenize a raw text corpus into binary .bin files for training.

Usage:
    python -m src.training.tokenize_corpus \
        --tokenizer tokenizer/ \
        --data data/raw/ \
        --output data/tokenized/ \
        --seq-len 1024
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.tokenizer import BPETokenizer


def tokenize_corpus(tokenizer_dir: Path, data_dir: Path, output_dir: Path, seq_len: int):
    tokenizer = BPETokenizer.load(tokenizer_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(data_dir.glob("**/*.txt"))
    if not txt_files:
        raise RuntimeError(f"No .txt files found in {data_dir}")

    print(f"Tokenizing {len(txt_files)} file(s) ...")

    for txt_file in tqdm(txt_files, unit="file"):
        text = txt_file.read_text(encoding="utf-8", errors="replace")
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)

        arr = np.array(ids, dtype=np.uint16)
        out_file = output_dir / (txt_file.stem + ".bin")
        arr.tofile(out_file)

    print(f"Done. Tokenized files written to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, default=Path("tokenizer/"))
    parser.add_argument("--data", type=Path, default=Path("data/raw/"))
    parser.add_argument("--output", type=Path, default=Path("data/tokenized/"))
    parser.add_argument("--seq-len", type=int, default=1024)
    args = parser.parse_args()
    tokenize_corpus(args.tokenizer, args.data, args.output, args.seq_len)


if __name__ == "__main__":
    main()
