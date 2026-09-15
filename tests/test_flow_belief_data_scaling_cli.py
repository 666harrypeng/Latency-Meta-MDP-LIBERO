from __future__ import annotations

from pathlib import Path

from latency_meta_mdp.legacy.cli.run_flow_belief_data_scaling import build_parser


def test_data_scaling_cli_uses_formal_defaults() -> None:
    args = build_parser().parse_args(
        [
            "--source-bulk-manifest",
            "source.json",
            "--cache-run-manifest",
            "cache.json",
            "--physical-audit-manifest",
            "audit.json",
            "--output-dir",
            "scaling-output",
        ]
    )

    assert args.flow_config == Path("configs/legacy/belief/dinov3_flow_belief_v1.yaml")
    assert args.audit_config == Path(
        "configs/legacy/analysis/dinov3_flow_belief_data_scaling_v1.yaml"
    )
    assert args.levels == (1, 2, 3)
    assert args.device == "cuda"
