"""
Full benchmark suite — runs speed, perplexity, and memory, then writes a report.

Usage:
    python -m src.benchmarks.run_all \
        --checkpoint checkpoints/small-int8/ \
        --test-data data/raw/test.txt \
        --output reports/

    # Skip perplexity (no test data yet):
    python -m src.benchmarks.run_all \
        --checkpoint checkpoints/small-int8/ \
        --no-perplexity
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.inference.engine import InferenceEngine
from .perplexity import compute_perplexity
from .speed import run_speed_benchmark, run_context_sweep
from .memory import run_memory_benchmark
from .report import write_report

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Full benchmark suite for Light-WeightLM")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, default=None, help=".txt file for perplexity eval")
    parser.add_argument("--output", type=Path, default=Path("reports/"))
    parser.add_argument("--prompt", type=str, default="The history of artificial intelligence began in")
    parser.add_argument("--speed-tokens", type=int, default=200)
    parser.add_argument("--speed-runs", type=int, default=10)
    parser.add_argument("--memory-tokens", type=int, default=500)
    parser.add_argument("--ppl-stride", type=int, default=512)
    parser.add_argument("--ppl-max-tokens", type=int, default=None)
    parser.add_argument("--no-perplexity", action="store_true")
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--context-sweep", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    console.rule("[bold]Light-WeightLM Benchmark Suite[/]")
    console.print(f"Checkpoint : [dim]{args.checkpoint}[/]")
    console.print(f"Device     : [dim]{args.device}[/]\n")

    engine = InferenceEngine(args.checkpoint, device=args.device)
    results: dict = {}

    # ── Speed ─────────────────────────────────────────────────────────────────
    console.rule("[cyan]Speed[/]")
    speed = run_speed_benchmark(engine, args.prompt, args.speed_tokens, args.speed_runs)
    results["speed"] = speed

    t = Table(show_header=True, header_style="bold cyan")
    t.add_column("Metric"); t.add_column("Value", justify="right")
    t.add_row("Throughput", f"[bold]{speed['throughput_tok_per_sec']} tok/s[/]")
    t.add_row("TTFT mean", f"{speed['ttft_ms'].get('mean', '—')} ms")
    t.add_row("TBT mean", f"{speed['tbt_ms'].get('mean', '—')} ms")
    t.add_row("TBT p95", f"{speed['tbt_ms'].get('p95', '—')} ms")
    t.add_row("TBT stdev", f"{speed['tbt_ms'].get('stdev', '—')} ms")
    console.print(t)

    if args.context_sweep:
        console.rule("[cyan]Context-length scaling[/]")
        sweep = run_context_sweep(engine, max_new_tokens=50)
        results["context_sweep"] = sweep
        st = Table(show_header=True, header_style="bold cyan")
        st.add_column("Prompt tokens", justify="right")
        st.add_column("Avg TBT (ms)", justify="right")
        st.add_column("tok/s", justify="right")
        for row in sweep:
            st.add_row(str(row["prompt_tokens"]), str(row["avg_tbt_ms"]), str(row["tok_per_sec"]))
        console.print(st)

    # ── Perplexity ─────────────────────────────────────────────────────────────
    if not args.no_perplexity:
        if args.test_data is None:
            console.print("[yellow]⚠ --test-data not provided, skipping perplexity. Pass --no-perplexity to suppress this warning.[/]")
        else:
            console.rule("[cyan]Perplexity[/]")
            text = args.test_data.read_text(encoding="utf-8", errors="replace")
            ppl = compute_perplexity(engine, text, stride=args.ppl_stride, max_tokens=args.ppl_max_tokens)
            results["perplexity"] = ppl

            pt = Table(show_header=True, header_style="bold cyan")
            pt.add_column("Metric"); pt.add_column("Value", justify="right")
            pt.add_row("Perplexity", f"[bold]{ppl['perplexity']}[/]")
            pt.add_row("Avg NLL", str(ppl["avg_nll"]))
            pt.add_row("Tokens evaluated", f"{ppl['total_tokens']:,}")
            console.print(pt)

    # ── Memory ────────────────────────────────────────────────────────────────
    if not args.no_memory:
        console.rule("[cyan]Memory[/]")
        mem = run_memory_benchmark(engine, args.prompt, args.memory_tokens)
        results["memory"] = mem

        mt = Table(show_header=True, header_style="bold cyan")
        mt.add_column("Metric"); mt.add_column("Value", justify="right")
        mt.add_row("Baseline RSS", f"{mem['baseline_rss_mb']} MB")
        mt.add_row("Peak RSS", f"[bold]{mem['peak_rss_mb']} MB[/]")
        mt.add_row("Delta", f"+{mem['delta_mb']} MB")
        mt.add_row("KV cache (est)", f"{mem['kv_cache_theoretical_mb']} MB")
        mt.add_row("Headroom vs 32 GB", f"[green]{mem['headroom_mb']} MB[/]")
        console.print(mt)

    # ── Report ────────────────────────────────────────────────────────────────
    console.rule("[cyan]Saving report[/]")
    json_path, md_path = write_report(args.checkpoint, results, args.output)
    console.print(f"JSON : [dim]{json_path}[/]")
    console.print(f"MD   : [dim]{md_path}[/]")
    console.rule("[bold green]Done[/]")


if __name__ == "__main__":
    main()
