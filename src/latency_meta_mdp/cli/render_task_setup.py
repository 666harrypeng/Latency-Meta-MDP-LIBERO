"""Render the synchronized initial policy-camera views for a task configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from latency_meta_mdp.rendering import render_task_setup


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-config",
        type=Path,
        default=Path("configs/task/dynamic_grasp_lift_l0.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest_path = render_task_setup(
        task_config_path=args.task_config,
        output_dir=args.output_dir,
        seed=args.seed,
        width=args.width,
        height=args.height,
    )
    print(json.dumps({"manifest": manifest_path.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
