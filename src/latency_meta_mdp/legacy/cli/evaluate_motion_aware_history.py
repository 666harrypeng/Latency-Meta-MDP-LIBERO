"""Evaluate motion-aware history encoders against matched historical contexts."""

from __future__ import annotations

import argparse
from pathlib import Path

from latency_meta_mdp.legacy.belief.causal_return.motion_aware_run import (
    evaluate_motion_aware_history_run,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bulk-manifest", type=Path, required=True)
    parser.add_argument("--cache-run-manifest", type=Path, required=True)
    parser.add_argument("--training-run-manifest", type=Path, required=True)
    parser.add_argument("--baseline-evaluation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--levels", nargs="+", type=int, default=(1, 2, 3))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--vision-config",
        type=Path,
        default=Path("configs/models/vision/dinov3_vits16_lvd1689m_224_v1.yaml"),
    )
    parser.add_argument(
        "--temporal-config",
        type=Path,
        default=Path("configs/contracts/temporal/h50_e25_d20_k6_v1.yaml"),
    )
    parser.add_argument(
        "--split-config",
        type=Path,
        default=Path("configs/legacy/data/formal_belief_train_val_v1.yaml"),
    )
    parser.add_argument(
        "--motion-aware-config",
        type=Path,
        default=Path("configs/legacy/belief/causal_return/motion_aware_history.yaml"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = evaluate_motion_aware_history_run(
        project_root=Path.cwd(),
        source_bulk_manifest=args.source_bulk_manifest,
        cache_run_manifest=args.cache_run_manifest,
        training_run_manifest=args.training_run_manifest,
        baseline_evaluation_manifest=args.baseline_evaluation_manifest,
        vision_config_path=args.vision_config,
        temporal_config_path=args.temporal_config,
        split_config_path=args.split_config,
        motion_aware_config_path=args.motion_aware_config,
        output_dir=args.output_dir,
        levels=tuple(args.levels),
        device=args.device,
    )
    print(manifest)


if __name__ == "__main__":
    main()
