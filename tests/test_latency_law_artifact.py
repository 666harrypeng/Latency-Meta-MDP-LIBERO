from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from latency_meta_mdp.latency_law_artifact import write_latency_law_certification


def test_latency_law_certification_binds_theory_sampling_and_causal_visibility(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "law-certification"
    manifest_path = write_latency_law_certification(
        project_root=Path.cwd(),
        config_path=Path("configs/latency/truncated_beta_5_26_400ms_v1.yaml"),
        output_dir=output,
        sample_count=20_000,
        sampling_seed=2026,
        maximum_probability_error=0.01,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["format_id"] == "categorical_latency_law_certification_v1"
    assert manifest["law_id"] == "truncated_beta_5_26_400ms_v1"
    assert manifest["delay_ticks"] == list(range(1, 21))
    assert len(manifest["theoretical_probabilities"]) == 20
    assert len(manifest["empirical_probabilities"]) == 20
    assert manifest["sample_min_ticks"] >= 1
    assert manifest["sample_max_ticks"] <= 20
    assert manifest["maximum_probability_error"] <= 0.01
    assert manifest["launch_context_fields"] == [
        "request_id",
        "launch_formal_tick",
        "launch_time_us",
        "observation",
    ]
    assert manifest["realized_delay_visible_at_launch"] is False
    assert manifest["passed"] is True
    assert manifest["eligible"] is (not manifest["implementation_dirty"])

    with pytest.raises(FileExistsError, match="already exists"):
        write_latency_law_certification(
            project_root=Path.cwd(),
            config_path=Path(
                "configs/latency/truncated_beta_5_26_400ms_v1.yaml"
            ),
            output_dir=output,
            sample_count=20_000,
            sampling_seed=2026,
            maximum_probability_error=0.01,
        )

    cli = importlib.import_module("latency_meta_mdp.cli.certify_latency_law")
    cli_output = tmp_path / "law-certification-cli"
    assert (
        cli.main(
            [
                "--project-root",
                str(Path.cwd()),
                "--config",
                "configs/latency/truncated_beta_5_26_400ms_v1.yaml",
                "--output-dir",
                str(cli_output),
                "--sample-count",
                "20000",
                "--sampling-seed",
                "2026",
                "--maximum-probability-error",
                "0.01",
            ]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    assert Path(printed["manifest"]) == cli_output / "manifest.json"
