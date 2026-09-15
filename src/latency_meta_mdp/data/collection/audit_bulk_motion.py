"""Audit train, development, and test motion seed banks without running simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from latency_meta_mdp.data.bulk_coverage import write_bulk_motion_coverage


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("configs/data/collection/panda_ball_bulk_v1.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = write_bulk_motion_coverage(
        project_root=args.project_root,
        plan_path=args.plan,
        output_dir=args.output_dir,
    )
    print(json.dumps({"manifest": manifest.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
