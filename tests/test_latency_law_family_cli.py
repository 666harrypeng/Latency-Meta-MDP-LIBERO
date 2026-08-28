from __future__ import annotations

from pathlib import Path

from latency_meta_mdp.cli.certify_latency_law_family import build_parser


def test_latency_law_family_cli_uses_locked_defaults() -> None:
    args = build_parser().parse_args(
        [
            "--source-bulk-manifest",
            "source.json",
            "--output-dir",
            "family-output",
        ]
    )

    assert args.nominal_law == Path("configs/latency/truncated_beta_5_26_400ms_v1.yaml")
    assert args.family_config == Path("configs/latency/truncated_beta_family_5_26_400ms_v1.yaml")
