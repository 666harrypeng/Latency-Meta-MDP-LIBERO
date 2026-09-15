from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from latency_meta_mdp.runtime.action_chunk_parity import run_action_chunk_zero_delay_parity


def _module():
    return importlib.import_module("latency_meta_mdp.runtime.action_chunk_parity_artifact")


def test_action_chunk_parity_artifact_is_exact_versioned_and_no_overwrite(
    tmp_path: Path,
) -> None:
    module = _module()
    result = run_action_chunk_zero_delay_parity(
        project_root=Path.cwd(),
        seed=10,
        camera_width=8,
        camera_height=8,
    )
    output_dir = tmp_path / "chunk-parity"

    manifest_path = module.write_action_chunk_parity_artifact(
        result=result,
        project_root=Path.cwd(),
        output_dir=output_dir,
        seed=10,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["format_id"] == "sharp_return_time_chunk_parity_v2"
    assert manifest["eligible"] is (not manifest["implementation_dirty"])
    assert manifest["blockers"] == ([] if manifest["eligible"] else ["implementation_dirty"])
    assert set(manifest["config_sha256"]) == {
        "runtime",
        "task",
        "motion",
        "control",
        "expert",
        "client",
        "temporal",
    }
    assert all(len(value) == 64 for value in manifest["config_sha256"].values())
    assert set(manifest["artifacts"]) == {
        "direct_trace.npz",
        "chunk_trace.npz",
        "events.json",
        "report.json",
    }

    with np.load(output_dir / "direct_trace.npz", allow_pickle=False) as direct:
        assert all(direct[name].dtype != object for name in direct.files)
    with np.load(output_dir / "chunk_trace.npz", allow_pickle=False) as chunk:
        np.testing.assert_array_equal(
            chunk["executed_action"],
            result.chunk.executed_action,
        )
        assert all(chunk[name].dtype != object for name in chunk.files)
    events = json.loads((output_dir / "events.json").read_text(encoding="utf-8"))
    assert len(events["chunk_events"]) == len(result.chunk_events)
    assert len(events["harness_events"]) == len(result.harness_events)
    assert events["direct_physical_events"] == events["chunk_physical_events"]
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report["exact"] is True
    assert report["generator_kind"] == "reference_action_replay_oracle"
    assert report["prediction_horizon"] == 50
    assert report["launch_trigger_horizon"] == 25
    assert report["maximum_delay_ticks"] == 20
    assert report["chunk_launch_ticks"][:3] == [25, 50, 75]
    assert report["starvation_ticks"] == []
    assert report["bootstrap_simulation_time_before_us"] == 0
    assert report["bootstrap_simulation_time_after_us"] == 0

    with pytest.raises(FileExistsError, match="already exists"):
        module.write_action_chunk_parity_artifact(
            result=result,
            project_root=Path.cwd(),
            output_dir=output_dir,
            seed=10,
        )


def test_action_chunk_parity_preflight_rejects_existing_output() -> None:
    module = _module()

    with pytest.raises(FileExistsError, match="already exists"):
        module.preflight_action_chunk_parity_output(Path.cwd())
