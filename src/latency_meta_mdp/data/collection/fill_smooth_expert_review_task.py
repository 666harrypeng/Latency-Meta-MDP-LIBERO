"""Append explicit reserve realization slots to one incomplete review task group."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from latency_meta_mdp.data.collection.review_extensions import (
    fill_review_task_realizations,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--logical-task-index", type=int, required=True)
    parser.add_argument(
        "--planner-python",
        type=Path,
        default=Path(".venv-expert-realization/bin/python"),
    )
    args = parser.parse_args(argv)
    result = fill_review_task_realizations(
        project_root=Path.cwd(),
        config_path=args.config,
        target=args.target,
        level=args.level,
        logical_task_index=args.logical_task_index,
        planner_worker_python=args.planner_python,
        on_progress=lambda message: print(message, file=sys.stderr, flush=True),
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
