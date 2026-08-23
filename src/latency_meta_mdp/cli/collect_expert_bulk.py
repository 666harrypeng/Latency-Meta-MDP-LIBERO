"""Collect the formal Panda-ball bulk first tranche."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

from latency_meta_mdp.bulk_collection import collect_first_tranche  # noqa: E402


def _progress(index, total, result) -> None:
    print(
        f"[{index:03d}/{total:03d}] L{result.level} seed={result.seed} "
        f"{result.terminal_status.value}:{result.terminal_reason.value}",
        file=sys.stderr,
        flush=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("configs/collection/panda_ball_bulk_v1.yaml"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest_path = collect_first_tranche(
        project_root=args.project_root,
        plan_path=args.plan,
        output_root=args.output_root,
        run_id=args.run_id,
        on_attempt=_progress,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "attempt_count": manifest["attempt_count"],
                "eligible": manifest["eligible"],
                "manifest": manifest_path.as_posix(),
                "success_count": manifest["success_count"],
            },
            sort_keys=True,
        )
    )
    return 0 if manifest["eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
