"""Thin shim — full benchmarking suite moved to src/benchmarks/."""

import sys

print(
    "NOTE: The benchmark has moved.\n"
    "Run the full suite with:\n\n"
    "    python -m src.benchmarks.run_all --checkpoint <path>\n\n"
    "Or individual benchmarks:\n"
    "    python -m src.benchmarks.speed\n"
    "    python -m src.benchmarks.perplexity\n"
    "    python -m src.benchmarks.memory\n",
    file=sys.stderr,
)
sys.exit(1)
