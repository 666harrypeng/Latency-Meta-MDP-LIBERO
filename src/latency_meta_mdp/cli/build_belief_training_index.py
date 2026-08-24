"""Build a lazy, episode-split index for return-belief training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from latency_meta_mdp.belief_training_artifact import (
    write_belief_training_index_artifact,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-bulk-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split-config",
        type=Path,
        default=Path("configs/data/belief_training_split_v1.yaml"),
    )
    parser.add_argument(
        "--bulk-plan",
        type=Path,
        default=Path("configs/collection/panda_ball_bulk_v1.yaml"),
    )
    parser.add_argument(
        "--view-config",
        type=Path,
        default=Path("configs/data/belief_data_view_h50_v2.yaml"),
    )
    parser.add_argument(
        "--latency-law",
        type=Path,
        default=Path("configs/latency/truncated_beta_5_26_400ms_v1.yaml"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = write_belief_training_index_artifact(
        project_root=args.project_root,
        source_bulk_manifest=args.source_bulk_manifest,
        split_config_path=args.split_config,
        bulk_plan_path=args.bulk_plan,
        view_config_path=args.view_config,
        latency_law_path=args.latency_law,
        output_dir=args.output_dir,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    print(json.dumps({"eligible": payload["eligible"], "manifest": str(manifest)}))
    return 0 if payload["eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
