from __future__ import annotations

from pathlib import Path

from latency_meta_mdp.legacy.cli.probe_flow_belief_law_retention import build_parser


def test_law_retention_cli_uses_locked_defaults() -> None:
    args = build_parser().parse_args(
        [
            "--source-bulk-manifest",
            "source.json",
            "--cache-run-manifest",
            "cache.json",
            "--multilaw-run-manifest",
            "run.json",
            "--level",
            "2",
            "--output-dir",
            "probe-output",
        ]
    )

    assert args.nominal_law == Path("configs/runtime/latency/truncated_beta_8_65_400ms_v1.yaml")
    assert args.family_config == Path(
        "configs/runtime/latency/truncated_beta_family_8_65_400ms_v1.yaml"
    )
    assert args.multilaw_config == Path("configs/legacy/belief/dinov3_flow_belief_multilaw_v3.yaml")
    assert args.probe_config == Path(
        "configs/legacy/analysis/dinov3_flow_belief_law_retention_probe_v1.yaml"
    )
    assert args.device == "cuda"
