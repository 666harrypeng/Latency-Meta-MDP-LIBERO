"""Create a deterministic external master-task split for a verified source corpus."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-id", required=True)
    parser.add_argument("--validation-master-count", type=int, required=True)
    parser.add_argument("--split-seed", type=int, required=True)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    load_fn: Callable[[Path], Any] | None = None,
) -> int:
    args = _parser().parse_args(argv)
    if load_fn is None:
        from latency_meta_mdp.data.source.loader import (
            load_verified_source_corpus,
        )

        load_fn = load_verified_source_corpus
    from latency_meta_mdp.data.source.split_view import (
        build_source_split,
        write_source_split,
    )

    corpus = load_fn(args.source_root)
    result = write_source_split(
        args.output,
        build_source_split(
            corpus,
            split_id=args.split_id,
            validation_master_count=args.validation_master_count,
            split_seed=args.split_seed,
        ),
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
