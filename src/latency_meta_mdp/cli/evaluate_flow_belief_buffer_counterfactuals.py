"""Evaluate one frozen Flow Belief checkpoint on exact action-buffer branches."""

from __future__ import annotations

import argparse
from pathlib import Path

from latency_meta_mdp.belief.flow.buffer_causality_evaluation import (
    evaluate_buffer_counterfactual_level,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation-manifest", type=Path, required=True)
    parser.add_argument("--source-bulk-manifest", type=Path, required=True)
    parser.add_argument("--cache-run-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--level", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--vision-config",
        type=Path,
        default=Path("configs/vision/dinov3_vits16_lvd1689m_224_v1.yaml"),
    )
    parser.add_argument(
        "--temporal-config",
        type=Path,
        default=Path("configs/temporal/h50_e25_d20_k6_v1.yaml"),
    )
    parser.add_argument(
        "--nominal-law",
        type=Path,
        default=Path("configs/latency/truncated_beta_8_65_400ms_v1.yaml"),
    )
    parser.add_argument(
        "--family-config",
        type=Path,
        default=Path("configs/latency/truncated_beta_family_8_65_400ms_v1.yaml"),
    )
    parser.add_argument(
        "--split-config",
        type=Path,
        default=Path("configs/data/formal_belief_train_val_v1.yaml"),
    )
    parser.add_argument(
        "--analysis-config",
        type=Path,
        default=Path("configs/analysis/dinov3_flow_belief_buffer_causality_v1.yaml"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = evaluate_buffer_counterfactual_level(
        project_root=Path.cwd(),
        simulation_manifest=args.simulation_manifest,
        source_bulk_manifest=args.source_bulk_manifest,
        cache_run_manifest=args.cache_run_manifest,
        checkpoint_dir=args.checkpoint_dir,
        vision_config_path=args.vision_config,
        temporal_config_path=args.temporal_config,
        nominal_law_path=args.nominal_law,
        family_config_path=args.family_config,
        split_config_path=args.split_config,
        analysis_config_path=args.analysis_config,
        output_dir=args.output_dir,
        level=args.level,
        device=args.device,
    )
    print(manifest)


if __name__ == "__main__":
    main()
