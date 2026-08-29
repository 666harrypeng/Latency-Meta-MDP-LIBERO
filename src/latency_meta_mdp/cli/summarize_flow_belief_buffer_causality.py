"""Combine L1/L2/L3 Flow Belief action-buffer causality reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from latency_meta_mdp.belief.flow.buffer_causality_summary import (
    summarize_buffer_causality_levels,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-manifests", nargs=3, type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = summarize_buffer_causality_levels(
        project_root=Path.cwd(),
        evaluation_manifests=tuple(args.evaluation_manifests),
        output_dir=args.output_dir,
    )
    print(manifest)


if __name__ == "__main__":
    main()
