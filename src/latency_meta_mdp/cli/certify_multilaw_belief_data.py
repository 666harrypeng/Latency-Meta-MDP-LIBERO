"""Certify the derived episode-level multi-law Flow Belief data view."""

from __future__ import annotations

import argparse
from pathlib import Path

from latency_meta_mdp.belief.flow.multilaw_data_artifact import (
    certify_multilaw_belief_data,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bulk-manifest", type=Path, required=True)
    parser.add_argument("--cache-run-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
        default=Path("configs/latency/truncated_beta_5_26_400ms_v1.yaml"),
    )
    parser.add_argument(
        "--family-config",
        type=Path,
        default=Path("configs/latency/truncated_beta_family_5_26_400ms_v1.yaml"),
    )
    parser.add_argument(
        "--split-config",
        type=Path,
        default=Path("configs/data/formal_belief_train_val_v1.yaml"),
    )
    parser.add_argument(
        "--multilaw-config",
        type=Path,
        default=Path("configs/belief/dinov3_flow_belief_multilaw_v2.yaml"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = certify_multilaw_belief_data(
        project_root=Path.cwd(),
        source_bulk_manifest=args.source_bulk_manifest,
        cache_run_manifest=args.cache_run_manifest,
        vision_config_path=args.vision_config,
        temporal_config_path=args.temporal_config,
        nominal_law_path=args.nominal_law,
        family_config_path=args.family_config,
        split_config_path=args.split_config,
        multilaw_config_path=args.multilaw_config,
        output_dir=args.output_dir,
    )
    print(manifest)


if __name__ == "__main__":
    main()
