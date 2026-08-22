from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from latency_meta_mdp.latency_parity import run_direct_zero_latency_parity


def _module():
    return importlib.import_module("latency_meta_mdp.latency_parity_artifact")


def test_write_latency_parity_artifact_is_exact_pickle_free_and_no_overwrite(
    tmp_path: Path,
) -> None:
    module = _module()
    result = run_direct_zero_latency_parity(
        project_root=Path.cwd(),
        seed=10,
        camera_width=8,
        camera_height=8,
    )
    output_dir = tmp_path / "parity"

    manifest_path = module.write_latency_parity_artifact(
        result=result,
        project_root=Path.cwd(),
        output_dir=output_dir,
        seed=10,
    )

    assert manifest_path == output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["format_id"] == "logical_latency_parity_v1"
    assert len(manifest["implementation_revision"]) == 40
    assert len(manifest["implementation_source_sha256"]) == 64
    assert set(manifest["config_sha256"]) == {
        "runtime",
        "task",
        "motion",
        "control",
        "expert",
    }
    assert all(len(value) == 64 for value in manifest["config_sha256"].values())
    assert type(manifest["implementation_dirty"]) is bool
    assert manifest["eligible"] is (not manifest["implementation_dirty"])
    assert manifest["blockers"] == (
        [] if manifest["eligible"] else ["implementation_dirty"]
    )
    assert set(manifest["artifacts"]) == {
        "direct_trace.npz",
        "zero_delay_trace.npz",
        "events.json",
        "report.json",
    }
    assert all(len(value) == 64 for value in manifest["artifacts"].values())

    with np.load(output_dir / "direct_trace.npz", allow_pickle=False) as direct:
        assert set(direct.files) == set(result.direct.array_field_names())
        np.testing.assert_array_equal(
            direct["boundary_time_us"],
            result.direct.boundary_time_us,
        )
        assert all(direct[name].dtype != object for name in direct.files)
    with np.load(output_dir / "zero_delay_trace.npz", allow_pickle=False) as harness:
        np.testing.assert_array_equal(
            harness["executed_action"],
            result.harness.executed_action,
        )
        assert all(harness[name].dtype != object for name in harness.files)

    events = json.loads((output_dir / "events.json").read_text(encoding="utf-8"))
    assert len(events["harness_events"]) == len(result.harness_events)
    assert events["direct_physical_events"] == events["zero_delay_physical_events"]
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report["exact"] is True
    assert report["seed"] == 10
    assert report["mismatches"] == []

    with pytest.raises(FileExistsError, match="already exists"):
        module.write_latency_parity_artifact(
            result=result,
            project_root=Path.cwd(),
            output_dir=output_dir,
            seed=10,
        )


def test_latency_parity_preflight_rejects_existing_output() -> None:
    module = _module()

    with pytest.raises(FileExistsError, match="already exists"):
        module.preflight_latency_parity_output(Path.cwd())
