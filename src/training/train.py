"""
Pre-training loop for LWLM.

Usage:
    python -m src.training.train --config configs/small.yaml --data data/tokenized/
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.optim import AdamW

from src.model import LWLM, LWLMConfig
from .dataset import TokenDataset


def build_lr_schedule(cfg, optimizer: AdamW):
    """Cosine decay with linear warmup."""
    warmup = cfg.train.warmup_steps
    max_steps = cfg.train.max_steps
    min_lr = cfg.train.lr * 0.1

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, max_steps - warmup)
        return min_lr / cfg.train.lr + 0.5 * (1 - min_lr / cfg.train.lr) * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model: LWLM, optimizer: AdamW, step: int, cfg, output_dir: Path):
    from safetensors.torch import save_file
    step_dir = output_dir / f"step-{step:07d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), step_dir / "model.safetensors")
    torch.save(optimizer.state_dict(), step_dir / "optimizer.pt")
    OmegaConf.save(cfg, step_dir / "config.yaml")
    # Update latest symlink (Windows: use a plain text pointer instead of symlink)
    (output_dir / "latest.txt").write_text(str(step_dir.name))
    print(f"  checkpoint saved → {step_dir}")


def train(cfg_path: Path, data_dir: Path, output_dir: Path, resume: Path | None):
    cfg = OmegaConf.load(cfg_path)
    torch.manual_seed(cfg.train.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if cfg.train.dtype == "bfloat16" and device.type == "cuda" else torch.float32
    print(f"Device: {device}  dtype: {dtype}")

    model_cfg = LWLMConfig(
        vocab_size=cfg.model.vocab_size,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        d_model=cfg.model.d_model,
        d_ffn=cfg.model.d_ffn,
        max_seq_len=cfg.model.max_seq_len,
        dropout=cfg.model.get("dropout", 0.1),
    )

    model = LWLM(model_cfg).to(device=device, dtype=dtype)
    print(f"Parameters: {model.param_count() / 1e6:.1f}M")

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = build_lr_schedule(cfg, optimizer)

    dataset = TokenDataset(data_dir, seq_len=cfg.model.max_seq_len)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=0,  # 0 workers for Windows compatibility
        pin_memory=(device.type == "cuda"),
    )

    start_step = 0
    if resume:
        from safetensors.torch import load_file
        model.load_state_dict(load_file(resume / "model.safetensors"))
        optimizer.load_state_dict(torch.load(resume / "optimizer.pt"))
        # Infer step from directory name
        start_step = int(resume.name.replace("step-", ""))
        print(f"Resumed from step {start_step}")

    model.train()
    step = start_step
    accum_loss = 0.0
    t0 = time.time()

    output_dir.mkdir(parents=True, exist_ok=True)

    for batch_ids in _infinite_loader(loader):
        if step >= cfg.train.max_steps:
            break

        batch_ids = batch_ids.to(device)
        inputs = batch_ids[:, :-1]
        targets = batch_ids[:, 1:]

        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
            logits, _ = model(inputs)
            loss = nn.functional.cross_entropy(logits.reshape(-1, model_cfg.vocab_size), targets.reshape(-1))

        (loss / cfg.train.grad_accum).backward()
        accum_loss += loss.item()

        if (step + 1) % cfg.train.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if step % 100 == 0:
                elapsed = time.time() - t0
                lr = scheduler.get_last_lr()[0]
                print(f"step {step:7d} | loss {accum_loss / cfg.train.grad_accum:.4f} | lr {lr:.2e} | {elapsed:.1f}s")
                accum_loss = 0.0
                t0 = time.time()

            if step % cfg.train.save_every == 0 and step > 0:
                save_checkpoint(model, optimizer, step, cfg, output_dir)

        step += 1

    save_checkpoint(model, optimizer, step, cfg, output_dir)
    print("Training complete.")


def _infinite_loader(loader):
    while True:
        yield from loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/tokenized/"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/"))
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()
    train(args.config, args.data, args.output, args.resume)


if __name__ == "__main__":
    main()
