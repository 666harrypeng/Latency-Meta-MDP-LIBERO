"""Convert one verified formal corpus into level-specific LeRobot datasets."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from latency_meta_mdp.policy_dataset_run import convert_formal_corpus_to_lerobot


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("configs/policy/pi05_panda_ball_full_sft_h50_v2.yaml"),
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    dataset_factory: Callable[..., Any] | None = None,
) -> int:
    args = _parser().parse_args(argv)
    manifest = convert_formal_corpus_to_lerobot(
        source_manifest=args.source_manifest,
        output_dir=args.output_dir,
        profile_path=args.profile,
        dataset_factory=dataset_factory,
    )
    print(json.dumps({"manifest": manifest.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
