"""Certify the nominal categorical latency law and its causal launch boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from latency_meta_mdp.runtime.latency_law_artifact import write_latency_law_certification


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/runtime/latency/truncated_beta_5_26_400ms_v1.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=100_000)
    parser.add_argument("--sampling-seed", type=int, default=2026)
    parser.add_argument("--maximum-probability-error", type=float, default=0.003)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = write_latency_law_certification(
        project_root=args.project_root,
        config_path=args.config,
        output_dir=args.output_dir,
        sample_count=args.sample_count,
        sampling_seed=args.sampling_seed,
        maximum_probability_error=args.maximum_probability_error,
    )
    print(json.dumps({"manifest": manifest.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
