"""
CPU-optimized inference engine for BloxSLM.

Key decisions:
- KV cache pre-allocated as a fixed tensor (no list growth)
- Supports greedy, top-k, and nucleus (top-p) sampling
- INT8 quantization loaded via safetensors
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from safetensors.torch import load_file

from src.model import BloxSLM, BloxSLMConfig
from src.tokenizer import BPETokenizer


class InferenceEngine:
    def __init__(self, checkpoint_dir: Path, device: str = "cpu"):
        checkpoint_dir = Path(checkpoint_dir)
        cfg = OmegaConf.load(checkpoint_dir / "config.yaml")

        self.model_cfg = BloxSLMConfig(
            vocab_size=cfg.model.vocab_size,
            n_layers=cfg.model.n_layers,
            n_heads=cfg.model.n_heads,
            d_model=cfg.model.d_model,
            d_ffn=cfg.model.d_ffn,
            max_seq_len=cfg.model.max_seq_len,
        )

        self.device = torch.device(device)
        self.model = BloxSLM(self.model_cfg)
        state = load_file(checkpoint_dir / "model.safetensors")
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        tokenizer_dir = checkpoint_dir / "tokenizer"
        if not tokenizer_dir.exists():
            tokenizer_dir = Path("tokenizer/")
        self.tokenizer = BPETokenizer.load(tokenizer_dir)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 0,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        seed: Optional[int] = None,
    ) -> str:
        if seed is not None:
            torch.manual_seed(seed)

        input_ids = self.tokenizer.encode(prompt, add_bos=True)
        tokens = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        kv_caches = None
        generated: list[int] = []

        for _ in range(max_new_tokens):
            logits, kv_caches = self.model(tokens, kv_caches)
            next_logits = logits[:, -1, :]  # [1, vocab]

            if repetition_penalty != 1.0:
                for token_id in set(input_ids + generated):
                    next_logits[0, token_id] /= repetition_penalty

            next_token_id = _sample(next_logits, temperature, top_k, top_p)

            if next_token_id == self.tokenizer.eos_id:
                break

            generated.append(next_token_id)
            tokens = torch.tensor([[next_token_id]], dtype=torch.long, device=self.device)

        return self.tokenizer.decode(generated)

    @torch.no_grad()
    def stream(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 0,
        top_p: float = 0.9,
    ) -> Iterator[str]:
        """Yield decoded text one token at a time."""
        input_ids = self.tokenizer.encode(prompt, add_bos=True)
        tokens = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        kv_caches = None

        for _ in range(max_new_tokens):
            logits, kv_caches = self.model(tokens, kv_caches)
            next_token_id = _sample(logits[:, -1, :], temperature, top_k, top_p)

            if next_token_id == self.tokenizer.eos_id:
                break

            yield self.tokenizer.decode([next_token_id])
            tokens = torch.tensor([[next_token_id]], dtype=torch.long, device=self.device)


def _sample(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> int:
    if temperature == 0.0:
        return int(logits.argmax(dim=-1).item())

    logits = logits / temperature

    if top_k > 0:
        kth = torch.topk(logits, top_k).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cumprobs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove_mask = cumprobs - F.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits[remove_mask] = float("-inf")
        logits = torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())
