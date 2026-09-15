"""Verify and summarize one formal structured-expert source corpus."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    return parser


def _summary(corpus: Any) -> dict[str, Any]:
    manifest = corpus.manifest
    return {
        "root": str(corpus.root),
        "corpus_id": manifest.corpus_id,
        "admitted_master_task_indices": list(manifest.admitted_master_task_indices),
        "master_task_count": manifest.master_task_count,
        "level_task_instance_count": manifest.level_task_instance_count,
        "episode_count": manifest.episode_count,
        "episodes_by_level": dict(manifest.episodes_by_level),
        "frame_count": manifest.frame_count,
        "shard_count": manifest.shard_count,
        "artifact_bytes": sum(row["bytes"] for row in manifest.artifacts.values()),
    }


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
    print(json.dumps(_summary(load_fn(args.source_root)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
