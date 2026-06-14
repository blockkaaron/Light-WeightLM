"""
Aggregate benchmark report.

Runs all benchmarks and writes a structured JSON + human-readable Markdown
report to disk. Useful for comparing checkpoints over time.

Usage:
    python -m src.benchmarks.run_all \
        --checkpoint checkpoints/small-int8/ \
        --test-data data/raw/test.txt \
        --output reports/
"""

from __future__ import annotations

import json
import platform
from datetime import datetime
from pathlib import Path

import psutil
import torch


def hardware_info() -> dict:
    cpu = platform.processor() or platform.machine()
    ram_gb = round(psutil.virtual_memory().total / 1024 ** 3, 1)
    has_cuda = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if has_cuda else "none"
    return {
        "cpu": cpu,
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "ram_gb": ram_gb,
        "gpu": gpu_name,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
    }


def write_report(
    checkpoint: Path,
    results: dict,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_name = checkpoint.name

    report = {
        "timestamp": timestamp,
        "checkpoint": str(checkpoint),
        "hardware": hardware_info(),
        "results": results,
    }

    json_path = output_dir / f"bench_{ckpt_name}_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    md_path = output_dir / f"bench_{ckpt_name}_{timestamp}.md"
    _write_markdown(md_path, report)

    return json_path, md_path


def _write_markdown(path: Path, report: dict):
    hw = report["hardware"]
    r = report["results"]
    ts = report["timestamp"]
    ckpt = report["checkpoint"]

    lines = [
        f"# Benchmark Report",
        f"",
        f"**Checkpoint**: `{ckpt}`  ",
        f"**Date**: {ts[:4]}-{ts[4:6]}-{ts[6:8]} {ts[9:11]}:{ts[11:13]}:{ts[13:15]}",
        f"",
        f"## Hardware",
        f"",
        f"| | |",
        f"|---|---|",
        f"| CPU | {hw['cpu']} |",
        f"| Cores | {hw['physical_cores']}P / {hw['logical_cores']}L |",
        f"| RAM | {hw['ram_gb']} GB |",
        f"| GPU | {hw['gpu']} |",
        f"| PyTorch | {hw['torch']} |",
        f"",
    ]

    if "speed" in r:
        s = r["speed"]
        lines += [
            f"## Speed",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Throughput | **{s.get('throughput_tok_per_sec', '—')} tok/s** |",
            f"| Time-to-first-token (mean) | {s.get('ttft_ms', {}).get('mean', '—')} ms |",
            f"| Time-between-tokens (mean) | {s.get('tbt_ms', {}).get('mean', '—')} ms |",
            f"| TBT p95 | {s.get('tbt_ms', {}).get('p95', '—')} ms |",
            f"| Prompt tokens | {s.get('prompt_tokens', '—')} |",
            f"| Generated tokens | {s.get('generated_tokens', '—')} |",
            f"| Runs | {s.get('n_runs', '—')} |",
            f"",
        ]

    if "perplexity" in r:
        p = r["perplexity"]
        lines += [
            f"## Perplexity",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Perplexity | **{p.get('perplexity', '—')}** |",
            f"| Avg NLL | {p.get('avg_nll', '—')} |",
            f"| Tokens evaluated | {p.get('total_tokens', '—'):,} |",
            f"",
        ]

    if "memory" in r:
        m = r["memory"]
        lines += [
            f"## Memory",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Baseline RSS | {m.get('baseline_rss_mb', '—')} MB |",
            f"| Peak RSS | {m.get('peak_rss_mb', '—')} MB |",
            f"| Delta | +{m.get('delta_mb', '—')} MB |",
            f"| KV cache (est) | {m.get('kv_cache_theoretical_mb', '—')} MB |",
            f"| Headroom vs 32 GB | {m.get('headroom_mb', '—')} MB |",
            f"",
        ]

    path.write_text("\n".join(lines), encoding="utf-8")
