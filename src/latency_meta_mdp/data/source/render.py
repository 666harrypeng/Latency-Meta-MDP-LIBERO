"""Render level-wise review videos from a verified source corpus."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--level", type=int, choices=(1, 2, 3))
    parser.add_argument("--episode-range", type=int, nargs=2, metavar=("START", "STOP"))
    parser.add_argument("--speed", type=int, choices=(1, 2, 3, 5, 10), default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    load_fn: Callable[[Path], Any] | None = None,
) -> int:
    args = _parser().parse_args(argv)
    episode_range = None if args.episode_range is None else tuple(args.episode_range)
    if load_fn is None:
        from latency_meta_mdp.data.source.loader import (
            load_verified_source_corpus,
        )

        load_fn = load_verified_source_corpus
    from latency_meta_mdp.data.source.review_video import (
        render_source_review_videos,
        review_plan_summary,
    )

    if args.dry_run:
        corpus = load_fn(args.source_root)
        print(
            json.dumps(
                review_plan_summary(
                    corpus,
                    level=args.level,
                    episode_range=episode_range,
                    speed=args.speed,
                ),
                sort_keys=True,
            )
        )
        return 0
    result = render_source_review_videos(
        source_root=args.source_root,
        output_root=args.output_root,
        level=args.level,
        episode_range=episode_range,
        speed=args.speed,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
