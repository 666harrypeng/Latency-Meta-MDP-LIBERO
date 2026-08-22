from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from latency_meta_mdp.cli.calibrate_control import main


def test_control_calibration_cli_preflights_before_running_simulation() -> None:
    with tempfile.TemporaryDirectory() as directory:
        output_root = Path(directory)
        existing = output_root / "calibration" / "control" / "existing"
        existing.mkdir(parents=True)

        with (
            patch(
                "latency_meta_mdp.cli.calibrate_control.run_control_calibration",
                side_effect=AssertionError("calibration must not start"),
            ),
            pytest.raises(FileExistsError, match="control calibration run already exists"),
        ):
            main(["--run-id", "existing", "--output-root", str(output_root)])
