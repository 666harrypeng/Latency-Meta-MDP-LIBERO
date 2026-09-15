from __future__ import annotations

from pathlib import Path

from latency_meta_mdp.legacy.cli.probe_flow_belief_encoder import build_parser


def test_fresh_decoder_cli_uses_formal_defaults() -> None:
    args = build_parser().parse_args(
        [
            "--source-bulk-manifest",
            "source.json",
            "--cache-run-manifest",
            "cache.json",
            "--flow-run-manifest",
            "flow.json",
            "--output-dir",
            "probe-output",
        ]
    )

    assert args.probe_config == Path(
        "configs/legacy/analysis/dinov3_flow_belief_fresh_decoder_probe_v1.yaml"
    )
    assert args.flow_config == Path("configs/legacy/belief/dinov3_flow_belief_v1.yaml")
    assert args.levels == (1, 2, 3)
    assert args.device == "cuda"
    assert not hasattr(args, "realized_delay")
