"""Collect the fixed formal-validation L3 same-source action-effect bank."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from latency_meta_mdp.belief.jepa.diagnostics import action_effect_run


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = action_effect_run.collect_l3_j4_bank(
        project_root=args.project_root,
        device=args.device,
        output_dir=args.output_dir,
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
