from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


def test_action_chunk_parity_cli_runs_real_tiny_calibration(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = importlib.import_module("latency_meta_mdp.cli.calibrate_action_chunk_client")
    output_dir = tmp_path / "chunk-client-cli"

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
    manifest_path = Path(json.loads(capsys.readouterr().out)["manifest"])
    assert manifest_path == output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["eligible"] is (not manifest["implementation_dirty"])
