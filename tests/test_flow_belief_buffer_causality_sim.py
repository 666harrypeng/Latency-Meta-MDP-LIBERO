import json
from types import SimpleNamespace

import numpy as np
import pytest

from latency_meta_mdp.legacy.belief.flow.buffer_causality import BufferCausalityCandidate
from latency_meta_mdp.legacy.belief.flow.buffer_causality_sim import (
    BufferCounterfactualSimulation,
    phase_candidates_from_episode_arrays,
    recorded_return_state,
    replay_source_max_abs,
    snapshot_return_state,
    write_buffer_counterfactual_simulation,
)


def _snapshot() -> SimpleNamespace:
    return SimpleNamespace(
        robot_qpos=np.arange(7, dtype=np.float64),
        robot_qvel=np.arange(7, dtype=np.float64) + 10,
        robot_gripper_qpos=np.array([0.03, -0.03]),
        robot_gripper_qvel=np.array([0.2, -0.2]),
        object_qpos=np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]),
        object_qvel=np.array([4.0, 5.0, 6.0, 0.1, 0.2, 0.3]),
        eef_pos=np.array([0.4, 0.5, 0.6]),
        eef_xmat=np.eye(3),
    )


def _recorded_arrays(count: int = 80) -> dict[str, np.ndarray]:
    snapshot = _snapshot()
    return {
        "robot_qpos": np.repeat(snapshot.robot_qpos[None], count, axis=0),
        "robot_qvel": np.repeat(snapshot.robot_qvel[None], count, axis=0),
        "gripper_qpos": np.repeat(snapshot.robot_gripper_qpos[None], count, axis=0),
        "gripper_qvel": np.repeat(snapshot.robot_gripper_qvel[None], count, axis=0),
        "object_pose": np.repeat(snapshot.object_qpos[None], count, axis=0),
        "object_velocity": np.repeat(snapshot.object_qvel[None], count, axis=0),
        "eef_position_world": np.repeat(snapshot.eef_pos[None], count, axis=0),
        "eef_orientation_matrix_world": np.repeat(snapshot.eef_xmat[None], count, axis=0),
    }


def test_return_state_packing_matches_recorded_contract() -> None:
    snapshot = _snapshot()
    arrays = _recorded_arrays()

    expected = np.concatenate(
        (
            snapshot.robot_qpos,
            snapshot.robot_qvel,
            [0.06, 0.4],
            snapshot.object_qpos[:3],
            snapshot.object_qvel[:3],
        )
    ).astype(np.float32)

    np.testing.assert_array_equal(snapshot_return_state(snapshot), expected)
    np.testing.assert_array_equal(recorded_return_state(arrays, 3), expected)


def test_replay_source_parity_checks_every_physical_field() -> None:
    snapshot = _snapshot()
    arrays = _recorded_arrays()

    assert replay_source_max_abs(snapshot=snapshot, arrays=arrays, source_tick=5) == 0.0
    arrays["eef_position_world"][5, 2] += 0.25
    assert replay_source_max_abs(snapshot=snapshot, arrays=arrays, source_tick=5) == 0.25


def test_phase_candidates_use_real_midpoints_with_complete_buffers() -> None:
    phases = np.asarray(["pregrasp"] * 40 + ["approach"] * 30 + ["close"] * 20 + ["lift"] * 50)

    rows = phase_candidates_from_episode_arrays(
        episode_id="l2-seed-001180-attempt-000",
        level=2,
        scene_seed=1180,
        expert_phase=phases,
        minimum_source_tick=25,
        required_real_action_count=25,
    )

    assert tuple(row.phase for row in rows) == ("pregrasp", "approach", "close", "lift")
    assert [row.source_tick for row in rows] == [32, 54, 79, 102]
    assert all(row.source_tick >= 25 for row in rows)
    assert all(row.source_tick + 25 <= len(phases) for row in rows)


def test_simulation_artifact_is_atomic_typed_and_no_overwrite(tmp_path) -> None:
    contexts = (
        BufferCausalityCandidate(
            episode_id="l1-seed-001180-attempt-000",
            level=1,
            scene_seed=1180,
            source_tick=25,
            phase="pregrasp",
            real_transition_count=100,
        ),
    )
    simulation = BufferCounterfactualSimulation(
        contexts=contexts,
        branch_ids=("expert", "hold", "half_speed", "delayed_prefix_5"),
        branch_actions=np.zeros((1, 4, 25, 7), dtype=np.float32),
        target_states=np.zeros((1, 4, 20, 22), dtype=np.float32),
        handoff_state=np.full((1, 4, 20), "driven"),
        replay_max_abs=np.zeros(1, dtype=np.float64),
        expert_reference_max_abs=np.full(1, 1e-8, dtype=np.float64),
    )
    output = tmp_path / "artifact"

    manifest_path = write_buffer_counterfactual_simulation(
        simulation=simulation,
        output_dir=output,
        manifest_fields={"implementation_revision": "a" * 40},
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["format_id"] == "flow_belief_buffer_counterfactual_sim_v1"
    assert manifest["context_count"] == 1
    assert manifest["branch_ids"] == list(simulation.branch_ids)
    assert manifest["maximum_replay_max_abs"] == 0.0
    assert manifest["maximum_expert_reference_max_abs"] == 1e-8
    assert set(manifest["artifacts"]) == {"contexts.json", "counterfactuals.npz"}
    with np.load(output / "counterfactuals.npz", allow_pickle=False) as arrays:
        assert arrays["branch_actions"].shape == (1, 4, 25, 7)
        assert arrays["target_states"].shape == (1, 4, 20, 22)
        assert arrays["handoff_state"].shape == (1, 4, 20)
        np.testing.assert_array_equal(arrays["expert_reference_max_abs"], [1e-8])
    with pytest.raises(FileExistsError):
        write_buffer_counterfactual_simulation(
            simulation=simulation,
            output_dir=output,
            manifest_fields={},
        )

    dirty_manifest = write_buffer_counterfactual_simulation(
        simulation=simulation,
        output_dir=tmp_path / "dirty",
        manifest_fields={"implementation_dirty": True},
    )
    dirty = json.loads(dirty_manifest.read_text())
    assert dirty["eligible"] is False
    assert dirty["blockers"] == ["implementation_dirty"]
