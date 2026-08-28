from __future__ import annotations

from pathlib import Path

from latency_meta_mdp.cli.certify_multilaw_belief_data import build_parser


def test_multilaw_belief_data_cli_uses_locked_defaults() -> None:
    args = build_parser().parse_args(
        [
            "--source-bulk-manifest",
            "source.json",
            "--cache-run-manifest",
            "cache.json",
            "--output-dir",
            "data-output",
        ]
    )

    assert args.nominal_law == Path("configs/latency/truncated_beta_8_65_400ms_v1.yaml")
    assert args.family_config == Path("configs/latency/truncated_beta_family_8_65_400ms_v1.yaml")
    assert args.multilaw_config == Path("configs/belief/dinov3_flow_belief_multilaw_v3.yaml")
