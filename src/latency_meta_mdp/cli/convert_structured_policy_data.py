"""Export the grouped train view of the structured source as state-aware LeRobot data."""

from __future__ import annotations

import argparse
from pathlib import Path

from latency_meta_mdp.policy_dataset_run import convert_structured_source_to_lerobot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--profile", type=Path, default=Path("configs/policy/pi05_structured_state16_h50_v1.yaml")
    )
    parser.add_argument("--levels", type=int, choices=(1, 2, 3), nargs="+", default=[3, 1, 2])
    args = parser.parse_args()
    path = convert_structured_source_to_lerobot(
        source_root=args.source_root,
        split_manifest=args.split_manifest,
        output_dir=args.output_dir,
        profile_path=args.profile,
        levels=tuple(args.levels),
    )
    print(path)


if __name__ == "__main__":
    main()
