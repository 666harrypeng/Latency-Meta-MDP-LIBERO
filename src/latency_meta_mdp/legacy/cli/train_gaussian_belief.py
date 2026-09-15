"""Train independent L1/L2/L3 Gaussian predictive return-belief models."""

from __future__ import annotations

import argparse
from pathlib import Path

from latency_meta_mdp.legacy.belief.gaussian.run import train_gaussian_belief_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bulk-manifest", type=Path, required=True)
    parser.add_argument("--cache-run-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
        "--latency-law",
        type=Path,
        default=Path("configs/runtime/latency/truncated_beta_5_26_400ms_v1.yaml"),
    )
    parser.add_argument(
        "--belief-config",
        type=Path,
        default=Path("configs/legacy/belief/dinov3_gaussian_belief_v1.yaml"),
    )
    parser.add_argument("--levels", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = train_gaussian_belief_run(
        project_root=Path.cwd(),
        source_bulk_manifest=args.source_bulk_manifest,
        cache_run_manifest=args.cache_run_manifest,
        vision_config_path=args.vision_config,
        temporal_config_path=args.temporal_config,
        latency_law_path=args.latency_law,
        belief_config_path=args.belief_config,
        output_dir=args.output_dir,
        levels=tuple(args.levels),
        device=args.device,
    )
    print(manifest)


if __name__ == "__main__":
    main()
