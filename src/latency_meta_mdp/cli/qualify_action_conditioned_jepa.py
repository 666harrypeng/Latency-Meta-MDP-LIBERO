"""Qualify pinned official JEPA-WM predictors and publish one manifest."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from contextlib import redirect_stdout
from pathlib import Path

from latency_meta_mdp.belief.action_conditioned_jepa.upstream_qualification import (
    qualify_upstream_reference,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--reference-config", type=Path, required=True)
    parser.add_argument("--mirror-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    with redirect_stdout(sys.stderr):
        manifest = qualify_upstream_reference(
            reference_config_path=args.reference_config,
            project_root=args.project_root,
            mirror_root=args.mirror_root,
            output_dir=args.output_dir,
            device=args.device,
        )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
