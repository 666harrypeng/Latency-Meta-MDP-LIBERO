from pathlib import Path

from latency_meta_mdp.cli.evaluate_flow_belief_buffer_counterfactuals import build_parser


def test_counterfactual_evaluation_cli_has_locked_defaults() -> None:
    args = build_parser().parse_args(
        [
            "--simulation-manifest",
            "simulation/manifest.json",
            "--source-bulk-manifest",
            "source.json",
            "--cache-run-manifest",
            "cache.json",
            "--checkpoint-dir",
            "checkpoint/L3",
            "--level",
            "3",
            "--output-dir",
            "evaluation/L3",
        ]
    )

    assert args.level == 3
    assert args.simulation_manifest == Path("simulation/manifest.json")
    assert args.checkpoint_dir == Path("checkpoint/L3")
    assert args.analysis_config == Path(
        "configs/analysis/dinov3_flow_belief_buffer_causality_v1.yaml"
    )
    assert args.nominal_law == Path("configs/latency/truncated_beta_8_65_400ms_v1.yaml")
    assert args.family_config == Path(
        "configs/latency/truncated_beta_family_8_65_400ms_v1.yaml"
    )
