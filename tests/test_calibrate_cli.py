from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from latency_meta_mdp.cli.calibrate_timing import main


def test_cli_preflights_before_running_expensive_calibration() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "artifact_index.json").write_text(
            json.dumps(
                {
                    "gates": {
                        "g1": {"manifest": "existing/manifest.json", "sha256": "0" * 64}
                    },
                    "schema_version": 1,
                }
            )
        )

        with (
            patch(
                "latency_meta_mdp.cli.calibrate_timing.run_g1_calibration",
                side_effect=AssertionError("calibration must not start"),
            ),
            pytest.raises(FileExistsError, match="g1 artifact index entry already exists"),
        ):
            main(["--run-id", "other", "--output-root", str(root)])
