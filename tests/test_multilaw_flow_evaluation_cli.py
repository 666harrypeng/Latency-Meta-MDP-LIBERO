from __future__ import annotations

from pathlib import Path

from latency_meta_mdp.cli.evaluate_multilaw_flow_belief import build_parser


def test_multilaw_evaluation_cli_uses_locked_defaults() -> None:
    args = build_parser().parse_args(
        [
            "--source-bulk-manifest",
            "source.json",
            "--cache-run-manifest",
            "cache.json",
            "--multilaw-run-manifest",
            "run.json",
            "--output-dir",
            "eval-output",
        ]
    )

    assert args.evaluation_config == Path(
        "configs/analysis/dinov3_flow_belief_multilaw_evaluation_v1.yaml"
    )
    assert args.levels == (1, 2, 3)
