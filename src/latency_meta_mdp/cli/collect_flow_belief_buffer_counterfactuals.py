"""Collect exact RoboSuite action-buffer counterfactual targets for Flow Belief."""

from __future__ import annotations

import argparse
from pathlib import Path

from latency_meta_mdp.belief.flow.buffer_causality_sim import (
    collect_buffer_counterfactual_simulation_run,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bulk-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--levels", nargs="+", type=int, default=(1, 2, 3))
    parser.add_argument("--contexts-per-phase", type=int)
    parser.add_argument(
        "--split-config",
        type=Path,
        default=Path("configs/data/formal_belief_train_val_v1.yaml"),
    )
    parser.add_argument(
        "--temporal-config",
        type=Path,
        default=Path("configs/temporal/h50_e25_d20_k6_v1.yaml"),
    )
    parser.add_argument(
        "--analysis-config",
        type=Path,
        default=Path("configs/analysis/dinov3_flow_belief_buffer_causality_v1.yaml"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    levels = tuple(args.levels)
    manifest = collect_buffer_counterfactual_simulation_run(
        project_root=Path.cwd(),
        source_bulk_manifest=args.source_bulk_manifest,
        split_config_path=args.split_config,
        temporal_config_path=args.temporal_config,
        analysis_config_path=args.analysis_config,
        output_dir=args.output_dir,
        levels=levels,
        contexts_per_phase=args.contexts_per_phase,
    )
    print(manifest)


if __name__ == "__main__":
    main()
