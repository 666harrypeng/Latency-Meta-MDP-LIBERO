from __future__ import annotations

from pathlib import Path

from latency_meta_mdp.cli.audit_flow_belief_data_sufficiency import build_parser


def test_data_sufficiency_cli_uses_formal_defaults() -> None:
    args = build_parser().parse_args(
        [
            "--source-bulk-manifest",
            "source.json",
            "--output-dir",
            "audit-output",
        ]
    )

    assert args.split_config == Path("configs/data/formal_belief_train_val_v1.yaml")
    assert args.audit_config == Path("configs/analysis/dinov3_flow_belief_data_scaling_v1.yaml")
