"""Train or preflight one Action-Conditioned JEPA temporal-selection fold."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from latency_meta_mdp.belief.jepa.ar.selection import (
    build_temporal_fold_preflight,
    execute_temporal_fold_job,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--temporal-config", type=Path, required=True)
    parser.add_argument("--fold-index", type=int, required=True)
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
    preflight = build_temporal_fold_preflight(
        project_root=args.project_root,
        temporal_config_path=args.temporal_config,
        fold_index=args.fold_index,
        model_seed=args.model_seed,
        microbatch_size=args.microbatch_size,
        num_workers=args.num_workers,
        device=args.device,
        output_dir=args.output_dir,
        resume=args.resume,
    )
    if args.preflight_only:
        print(json.dumps(preflight.to_mapping(), indent=2, sort_keys=True))
        return 0
    manifest = execute_temporal_fold_job(
        project_root=args.project_root,
        temporal_config_path=args.temporal_config,
        fold_index=args.fold_index,
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
