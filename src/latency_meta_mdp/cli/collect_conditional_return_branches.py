"""Collect same-source-information executable-control branches."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from latency_meta_mdp.belief.conditional_return_flow.branch_collection import (
    collect_control_branch_corpus,
)
from latency_meta_mdp.belief.conditional_return_flow.branch_contracts import (
    normalize_scene_seed_ranges,
)


def _scene_seed_range(value: str) -> tuple[int, int]:
    try:
        start_text, stop_text = value.split(":", maxsplit=1)
        result = (int(start_text), int(stop_text))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("scene seed range must be START:STOP") from error
    try:
        normalized = normalize_scene_seed_ranges((result,))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    assert normalized is not None
    return normalized[0]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-bulk-manifest", type=Path, required=True)
    parser.add_argument("--split-config", type=Path, required=True)
    parser.add_argument("--temporal-config", type=Path, required=True)
    parser.add_argument("--branch-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--levels", type=int, nargs="+", required=True)
    parser.add_argument("--maximum-contexts-per-level", type=int)
    parser.add_argument("--scene-seed-range", type=_scene_seed_range, action="append")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = collect_control_branch_corpus(
        project_root=args.project_root,
        source_bulk_manifest=args.source_bulk_manifest,
        split_config_path=args.split_config,
        temporal_config_path=args.temporal_config,
        branch_config_path=args.branch_config,
        output_dir=args.output_dir,
        levels=tuple(args.levels),
        maximum_contexts_per_level=args.maximum_contexts_per_level,
        allowed_scene_seed_ranges=(
            None if args.scene_seed_range is None else tuple(args.scene_seed_range)
        ),
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
