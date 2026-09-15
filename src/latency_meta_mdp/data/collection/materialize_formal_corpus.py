"""Materialize immutable expert runs into one audited formal corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from latency_meta_mdp.data.formal_corpus import materialize_formal_corpus


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--source-manifest",
        dest="source_manifests",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest_path = materialize_formal_corpus(
        project_root=args.project_root,
        source_manifests=tuple(args.source_manifests),
        output_dir=args.output_dir,
        levels=tuple(args.levels),
        seed_start=args.seed_start,
        seed_count=args.seed_count,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "eligible": manifest["eligible"],
                "episode_count": manifest["episode_count"],
                "manifest": manifest_path.as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0 if manifest["eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
