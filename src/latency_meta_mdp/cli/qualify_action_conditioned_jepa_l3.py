"""Qualify one completed epoch-75 L3 Action-Conditioned JEPA checkpoint."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from latency_meta_mdp.belief.action_conditioned_jepa import qualification_run


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--model-seed", type=int, choices=(7, 17, 27), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = qualification_run.execute_l3_final_qualification(
        project_root=args.project_root,
        model_seed=args.model_seed,
        device=args.device,
        batch_size=args.batch_size,
        output_path=args.output,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
