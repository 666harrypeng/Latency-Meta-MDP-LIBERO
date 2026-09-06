"""Train the selected Action-Conditioned JEPA configuration on one task level."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from latency_meta_mdp.belief.action_conditioned_jepa.admission_runner import (
    build_jepa_admission_preflight,
    execute_jepa_admission_job,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--level", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--microbatch-size", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--qualification-max-epochs", type=int)
    parser.add_argument("--disable-wandb", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.preflight_only:
        preflight = build_jepa_admission_preflight(
            level=args.level,
            project_root=args.project_root,
            model_seed=args.model_seed,
            microbatch_size=args.microbatch_size,
            num_workers=args.num_workers,
            device=args.device,
            output_dir=args.output_dir,
            resume=args.resume,
        )
        print(json.dumps(preflight.to_mapping(), indent=2, sort_keys=True))
        return 0
    manifest = execute_jepa_admission_job(
        level=args.level,
        project_root=args.project_root,
        model_seed=args.model_seed,
        microbatch_size=args.microbatch_size,
        num_workers=args.num_workers,
        device=args.device,
        output_dir=args.output_dir,
        resume=args.resume,
        qualification_max_epochs=args.qualification_max_epochs,
        enable_wandb=not args.disable_wandb,
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
