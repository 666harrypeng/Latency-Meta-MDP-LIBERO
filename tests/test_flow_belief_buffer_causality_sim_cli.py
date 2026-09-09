from pathlib import Path

from latency_meta_mdp.cli.collect_flow_belief_buffer_counterfactuals import build_parser


def test_counterfactual_sim_cli_defaults_to_all_levels() -> None:
    args = build_parser().parse_args(
        [
            "--source-bulk-manifest",
            "source.json",
            "--output-dir",
            "output",
        ]
    )

    assert args.source_bulk_manifest == Path("source.json")
    assert args.output_dir == Path("output")
    assert args.levels == (1, 2, 3)
    assert args.contexts_per_phase is None
    assert args.split_config == Path("configs/data/formal_belief_train_val_v1.yaml")
    assert args.analysis_config == Path(
        "configs/analysis/dinov3_flow_belief_buffer_causality_v1.yaml"
    )
