"""Verify and summarize a Conditional Return Flow control-branch artifact."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from latency_meta_mdp.legacy.belief.conditional_return_flow.branch_artifacts import (
    load_verified_control_branch_corpus,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    corpus = load_verified_control_branch_corpus(args.manifest)
    nominal_errors = corpus.nominal_future_max_abs[corpus.nominal_future_valid]
    payload = {
        "manifest": str(args.manifest.resolve()),
        "scientific_gate_pass": corpus.scientific_gate_pass,
        "artifact_eligible": corpus.artifact_eligible,
        "context_count": int(len(np.unique(corpus.source_context_index))),
        "branch_count": int(len(corpus.source_context_index)),
        "branch_kinds": sorted(set(corpus.branch_kind.tolist())),
        "absorbing_target_count": int(np.sum(corpus.target_absorbing)),
        "maximum_source_replay_abs": float(np.max(corpus.source_replay_max_abs)),
        "all_source_fingerprints_match": bool(np.all(corpus.source_fingerprint_match)),
        "maximum_nominal_future_abs": (
            0.0 if len(nominal_errors) == 0 else float(np.max(nominal_errors))
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
