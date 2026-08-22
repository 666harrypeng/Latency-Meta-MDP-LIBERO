"""Certify warm-start sharp H16/E8 execution on a real L1 rollout."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

from latency_meta_mdp.action_chunk_parity import (  # noqa: E402
    run_action_chunk_zero_delay_parity,
)
from latency_meta_mdp.action_chunk_parity_artifact import (  # noqa: E402
    preflight_action_chunk_parity_output,
    write_action_chunk_parity_artifact,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--camera-height", type=int, default=256)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    preflight_action_chunk_parity_output(args.output_dir)
    result = run_action_chunk_zero_delay_parity(
        project_root=args.project_root,
        seed=args.seed,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
    )
    manifest_path = write_action_chunk_parity_artifact(
        result=result,
        project_root=args.project_root,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(json.dumps({"manifest": manifest_path.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
