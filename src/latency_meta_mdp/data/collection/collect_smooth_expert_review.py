"""Collect the bounded L1-L3 smooth-expert behavior review videos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from latency_meta_mdp.data.collection.review_collection import collect_behavior_review


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument(
        "--planner-python",
        type=Path,
        default=Path(".venv-expert-realization/bin/python"),
    )
    parser.add_argument("--maximum-task-level-groups", type=int)
    args = parser.parse_args(argv)

    manifest = collect_behavior_review(
        project_root=Path.cwd(),
        config_path=args.config,
        target=args.target,
        planner_worker_python=args.planner_python,
        maximum_task_level_groups=args.maximum_task_level_groups,
        on_progress=lambda message: print(message, file=sys.stderr, flush=True),
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
