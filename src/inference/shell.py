"""Interactive REPL shell: python -m src.inference.shell"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt

from .engine import InferenceEngine

console = Console()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=300)
    args = parser.parse_args()

    console.print(f"[bold green]LWLM[/] loading from [dim]{args.checkpoint}[/] ...")
    engine = InferenceEngine(args.checkpoint, device=args.device)
    console.print("[bold green]Ready.[/] Type [bold]exit[/] to quit.\n")

    while True:
        try:
            prompt = Prompt.ask("[bold blue]>>>[/]")
        except (EOFError, KeyboardInterrupt):
            break

        if prompt.strip().lower() in {"exit", "quit", "q"}:
            break

        console.print("[dim]Generating...[/]")
        for token_text in engine.stream(prompt, max_new_tokens=args.max_tokens, temperature=args.temperature):
            console.print(token_text, end="")
        console.print()


if __name__ == "__main__":
    main()
