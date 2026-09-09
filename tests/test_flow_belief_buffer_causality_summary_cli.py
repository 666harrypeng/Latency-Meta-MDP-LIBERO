from pathlib import Path

from latency_meta_mdp.cli.summarize_flow_belief_buffer_causality import build_parser


def test_summary_cli_requires_three_level_manifests() -> None:
    args = build_parser().parse_args(
        [
            "--evaluation-manifests",
            "L1/manifest.json",
            "L2/manifest.json",
            "L3/manifest.json",
            "--output-dir",
            "combined",
        ]
    )

    assert args.evaluation_manifests == [
        Path("L1/manifest.json"),
        Path("L2/manifest.json"),
        Path("L3/manifest.json"),
    ]
    assert args.output_dir == Path("combined")
