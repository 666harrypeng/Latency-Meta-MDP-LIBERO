"""Benchmark the three final L3 JEPA checkpoints on the local Belief runtime path."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from latency_meta_mdp.belief.jepa.ar import benchmark as runtime_benchmark


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = runtime_benchmark.execute_l3_runtime_benchmark(
        project_root=args.project_root,
        device=args.device,
        output_path=args.output,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
