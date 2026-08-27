"""Render selected Flow Belief future-state samples in the agent view."""

from __future__ import annotations

import argparse
from pathlib import Path

from latency_meta_mdp.belief.flow.ghost_run import render_flow_belief_quality_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bulk-manifest", type=Path, required=True)
    parser.add_argument("--quality-sample-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--task-config",
        type=Path,
        default=Path("configs/task/dynamic_grasp_lift_l0.yaml"),
    )
    parser.add_argument(
        "--control-config",
        type=Path,
        default=Path("configs/control/panda_osc_pose_delta_v1.yaml"),
    )
    parser.add_argument(
        "--temporal-config",
        type=Path,
        default=Path("configs/temporal/h50_e25_d20_k6_v1.yaml"),
    )
    parser.add_argument(
        "--ghost-config",
        type=Path,
        default=Path("configs/analysis/flow_belief_agentview_ghost_v1.yaml"),
    )
    parser.add_argument("--levels", nargs="+", type=int, default=(1, 2, 3))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = render_flow_belief_quality_run(
        project_root=Path.cwd(),
        source_bulk_manifest=args.source_bulk_manifest,
        quality_sample_manifest=args.quality_sample_manifest,
        task_config_path=args.task_config,
        control_config_path=args.control_config,
        temporal_config_path=args.temporal_config,
        ghost_config_path=args.ghost_config,
        output_dir=args.output_dir,
        levels=tuple(args.levels),
    )
    print(manifest)


if __name__ == "__main__":
    main()
