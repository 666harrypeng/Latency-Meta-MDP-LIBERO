"""Certify BELIEF raw/index readiness on a bulk source run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from latency_meta_mdp.belief_data_artifact import certify_belief_data_contract


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-bulk-manifest", type=Path, required=True)
    parser.add_argument(
        "--view-config",
        type=Path,
        default=Path("configs/data/belief_data_view_v1.yaml"),
    )
    parser.add_argument(
        "--latency-law",
        type=Path,
        default=Path("configs/latency/truncated_beta_5_26_400ms_v1.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = certify_belief_data_contract(
        project_root=args.project_root,
        source_bulk_manifest=args.source_bulk_manifest,
        view_config_path=args.view_config,
        latency_law_path=args.latency_law,
        output_dir=args.output_dir,
    )
    report = json.loads(manifest.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "eligible": report["eligible"],
                "manifest": manifest.as_posix(),
                "raw_index_ready": report["raw_index_ready"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
