"""Certify deterministic episode-level latency-law assignments."""

from __future__ import annotations

import argparse
from pathlib import Path

from latency_meta_mdp.runtime.latency_law_family_artifact import (
    certify_episode_latency_law_family,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bulk-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--nominal-law",
        type=Path,
        default=Path("configs/runtime/latency/truncated_beta_8_65_400ms_v1.yaml"),
    )
    parser.add_argument(
        "--family-config",
        type=Path,
        default=Path("configs/runtime/latency/truncated_beta_family_8_65_400ms_v1.yaml"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = certify_episode_latency_law_family(
        project_root=Path.cwd(),
        source_bulk_manifest=args.source_bulk_manifest,
        nominal_law_path=args.nominal_law,
        family_config_path=args.family_config,
        output_dir=args.output_dir,
    )
    print(manifest)


if __name__ == "__main__":
    main()
