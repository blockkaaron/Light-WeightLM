"""Checkpoint loading utilities shared between training and inference."""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file

from src.model import LWLM, LWLMConfig


def load_model(checkpoint_dir: Path, device: str = "cpu") -> tuple[LWLM, object]:
    """Load model and config from a checkpoint directory."""
    checkpoint_dir = Path(checkpoint_dir)

    # Support 'latest.txt' pointer written by train.py (Windows-compatible)
    if (checkpoint_dir / "latest.txt").exists():
        latest_name = (checkpoint_dir / "latest.txt").read_text().strip()
        checkpoint_dir = checkpoint_dir / latest_name

    cfg = OmegaConf.load(checkpoint_dir / "config.yaml")
    model_cfg = LWLMConfig(
        vocab_size=cfg.model.vocab_size,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        d_model=cfg.model.d_model,
        d_ffn=cfg.model.d_ffn,
        max_seq_len=cfg.model.max_seq_len,
    )
    model = LWLM(model_cfg)
    state = load_file(checkpoint_dir / "model.safetensors")
    model.load_state_dict(state)
    model.to(torch.device(device))
    model.eval()
    return model, cfg
