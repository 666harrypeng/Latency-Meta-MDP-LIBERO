"""Audit physical trajectory coverage before Flow Belief data scaling."""

from __future__ import annotations

import argparse
from pathlib import Path

from latency_meta_mdp.belief.flow.data_sufficiency import (
    write_flow_belief_physical_data_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bulk-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split-config",
        type=Path,
        default=Path("configs/data/formal_belief_train_val_v1.yaml"),
    )
    parser.add_argument(
        "--audit-config",
        type=Path,
        default=Path("configs/analysis/dinov3_flow_belief_data_scaling_v1.yaml"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = write_flow_belief_physical_data_audit(
        project_root=Path.cwd(),
        source_bulk_manifest=args.source_bulk_manifest,
        split_config_path=args.split_config,
        audit_config_path=args.audit_config,
        output_dir=args.output_dir,
    )
    print(manifest)


if __name__ == "__main__":
    main()
