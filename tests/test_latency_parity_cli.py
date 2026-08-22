from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


def test_latency_parity_cli_runs_a_real_tiny_calibration(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = importlib.import_module("latency_meta_mdp.cli.calibrate_latency_harness")
    output_dir = tmp_path / "parity-cli"

    result = cli.main(
        [
            "--project-root",
            str(Path.cwd()),
            "--output-dir",
            str(output_dir),
            "--seed",
            "10",
            "--camera-width",
            "8",
            "--camera-height",
            "8",
        ]
    )

    assert result == 0
    output = json.loads(capsys.readouterr().out)
    manifest_path = Path(output["manifest"])
    assert manifest_path == output_dir / "manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["eligible"] is (not manifest["implementation_dirty"])
