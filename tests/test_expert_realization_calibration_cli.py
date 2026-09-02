from __future__ import annotations

from pathlib import Path


def test_calibration_cli_prints_only_manifest_to_stdout(tmp_path: Path, capsys) -> None:
    """Break caught: progress contaminates the machine-readable manifest-path stdout contract."""
    from test_expert_realization_calibration import _row

    from latency_meta_mdp.cli.calibrate_structured_expert_timing import main

    target = tmp_path / "calibration"

    def run_attempt(request):
        return _row(request.logical_task_index, request.level)

    result = main(
        [
            "--config",
            "configs/analysis/panda_ball_structured_timing_calibration.yaml",
            "--target",
            str(target),
        ],
        attempt_runner=run_attempt,
    )
    captured = capsys.readouterr()
    assert result == 0
    assert captured.out == f"{target / 'manifest.json'}\n"
    assert "completed=150 total=150" in captured.err
