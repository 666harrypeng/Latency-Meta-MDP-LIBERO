"""Summarize one complete Action-Conditioned JEPA temporal-selection sweep."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from latency_meta_mdp.belief.jepa.ar.report import (
    write_temporal_selection_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-seed", type=int, default=7)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = write_temporal_selection_report(
        args.selection_root,
        args.output_dir,
        model_seed=args.model_seed,
    )
    print(paths.summary_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
