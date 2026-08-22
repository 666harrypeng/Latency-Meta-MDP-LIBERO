"""Collect a synchronized zero-latency L1-L3 expert pilot campaign."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

from latency_meta_mdp.pilot_collection import (  # noqa: E402
    collect_expert_pilot_run,
    load_pilot_run_spec,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/collection/panda_ball_pilot_v1.yaml"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest_path = collect_expert_pilot_run(
        project_root=args.project_root,
        output_root=args.output_root,
        run_id=args.run_id,
        spec=load_pilot_run_spec(args.config),
    )
    print(json.dumps({"manifest": manifest_path.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
