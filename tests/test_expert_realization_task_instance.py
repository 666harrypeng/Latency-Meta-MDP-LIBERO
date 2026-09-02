from __future__ import annotations

import dataclasses
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

INITIAL_STATE_MEMBERS = (
    "mujoco_qpos",
    "mujoco_qvel",
    "mujoco_act",
    "mujoco_ctrl",
    "sim_time_seconds",
    "robot_qpos",
    "robot_qvel",
    "gripper_qpos",
    "gripper_qvel",
    "object_qpos",
    "object_qvel",
    "eef_position_world",
    "eef_orientation_matrix_world",
    "osc_origin_position_world",
    "osc_origin_orientation_world",
    "osc_goal_position_base",
    "osc_goal_orientation_matrix_base",
    "clock_physics_step",
    "clock_formal_tick",
    "clock_time_us",
    "clock_physics_dt_us",
    "clock_formal_tick_us",
    "handoff_state_json_utf8",
    "outcome_state_json_utf8",
    "agentview_rgb_sha256",
    "robot0_eye_in_hand_rgb_sha256",
    "motion_time_origin_us",
)


def _initial_state_values() -> dict[str, np.ndarray]:
    return {
        "mujoco_qpos": np.arange(16, dtype=np.float64),
        "mujoco_qvel": np.arange(15, dtype=np.float64),
        "mujoco_act": np.empty((0,), dtype=np.float64),
        "mujoco_ctrl": np.arange(9, dtype=np.float64),
        "sim_time_seconds": np.array(0.0, dtype=np.float64),
        "robot_qpos": np.arange(7, dtype=np.float64),
        "robot_qvel": np.arange(7, dtype=np.float64),
        "gripper_qpos": np.array([0.02, -0.02], dtype=np.float64),
        "gripper_qvel": np.array([0.0, 0.0], dtype=np.float64),
        "object_qpos": np.array([0.0, -0.12, 0.833, 1.0, 0.0, 0.0, 0.0]),
        "object_qvel": np.arange(6, dtype=np.float64),
        "eef_position_world": np.arange(3, dtype=np.float64),
        "eef_orientation_matrix_world": np.eye(3, dtype=np.float64),
        "osc_origin_position_world": np.arange(3, dtype=np.float64),
        "osc_origin_orientation_world": np.eye(3, dtype=np.float64),
        "osc_goal_position_base": np.arange(3, dtype=np.float64),
        "osc_goal_orientation_matrix_base": np.eye(3, dtype=np.float64),
        "clock_physics_step": np.array(0, dtype=np.int64),
        "clock_formal_tick": np.array(0, dtype=np.int64),
        "clock_time_us": np.array(0, dtype=np.int64),
        "clock_physics_dt_us": np.array(2_000, dtype=np.int64),
        "clock_formal_tick_us": np.array(20_000, dtype=np.int64),
        "handoff_state_json_utf8": np.frombuffer(b'{"state":"driven"}\n', dtype=np.uint8),
        "outcome_state_json_utf8": np.frombuffer(b'{"status":"running"}\n', dtype=np.uint8),
        "agentview_rgb_sha256": np.arange(32, dtype=np.uint8),
        "robot0_eye_in_hand_rgb_sha256": np.arange(32, dtype=np.uint8)[::-1],
        "motion_time_origin_us": np.array(0, dtype=np.int64),
    }


def _state():
    from latency_meta_mdp.expert_realization.task_instance import InitialStateSnapshot

    return InitialStateSnapshot(**_initial_state_values())


def test_canonical_json_encoder_has_locked_sorted_pretty_bytes() -> None:
    """Break caught: task identity JSON changes formatting or permits noncanonical key order."""
    from latency_meta_mdp.expert_realization.task_instance import _canonical_json_bytes

    assert _canonical_json_bytes({"z": 1, "a": [2, 3]}) == (
        b'{\n  "a": [\n    2,\n    3\n  ],\n  "z": 1\n}\n'
    )
    with pytest.raises(ValueError):
        _canonical_json_bytes({"bad": float("nan")})


def test_initial_state_npz_is_repeatable_sorted_and_object_free() -> None:
    """Break caught: ZIP metadata, member order, or object arrays destabilize task IDs."""
    from latency_meta_mdp.expert_realization.task_instance import (
        decode_initial_state_npz,
        encode_initial_state_npz,
    )

    state = _state()
    first = encode_initial_state_npz(state)
    second = encode_initial_state_npz(state)

    assert first == second
    with np.load(io.BytesIO(first), allow_pickle=False) as archive:
        assert tuple(archive.files) == tuple(sorted(INITIAL_STATE_MEMBERS))
        assert all(archive[name].dtype != np.dtype(object) for name in archive.files)
    decoded = decode_initial_state_npz(first)
    for name in INITIAL_STATE_MEMBERS:
        assert np.array_equal(getattr(decoded, name), getattr(state, name))
        assert not getattr(decoded, name).flags.writeable


def test_initial_state_contract_has_every_locked_member_and_detaches_arrays() -> None:
    """Break caught: boundary-0 identity omits a causal state category or leaks mutable views."""
    state = _state()

    assert tuple(field.name for field in dataclasses.fields(state)) == INITIAL_STATE_MEMBERS
    for name in INITIAL_STATE_MEMBERS:
        assert isinstance(getattr(state, name), np.ndarray)
        assert not getattr(state, name).flags.writeable


@pytest.mark.parametrize(
    ("field", "bad"),
    (
        ("mujoco_ctrl", np.zeros(8, dtype=np.float64)),
        ("sim_time_seconds", np.zeros(1, dtype=np.float64)),
        ("robot_qpos", np.zeros(7, dtype=np.float32)),
        ("gripper_qvel", np.zeros(3, dtype=np.float64)),
        ("object_qvel", np.zeros(5, dtype=np.float64)),
        ("eef_orientation_matrix_world", np.zeros((9,), dtype=np.float64)),
        ("clock_formal_tick", np.array(0, dtype=np.int32)),
        ("agentview_rgb_sha256", np.zeros(31, dtype=np.uint8)),
    ),
)
def test_initial_state_rejects_wrong_shape_or_dtype(field: str, bad: np.ndarray) -> None:
    """Break caught: malformed state can be hashed as if it satisfied the replay contract."""
    from latency_meta_mdp.expert_realization.task_instance import InitialStateSnapshot

    values = _initial_state_values()
    values[field] = bad
    with pytest.raises((TypeError, ValueError), match=field):
        InitialStateSnapshot(**values)


def test_same_seed_has_exact_shared_endpoint_bytes_across_l1_l2_l3() -> None:
    """Break caught: level-specific path construction changes the committed shared endpoints."""
    from latency_meta_mdp.expert_realization.task_instance import _materialize_motion_profile

    rows = [
        _materialize_motion_profile(project_root=Path.cwd(), level=level, task_instance_seed=4000)
        for level in (1, 2, 3)
    ]

    assert rows[0].shared_endpoint_bytes == rows[1].shared_endpoint_bytes
    assert rows[1].shared_endpoint_bytes == rows[2].shared_endpoint_bytes
    assert all(row.shared_endpoints_xy.shape == (2, 2) for row in rows)
    assert all(row.shared_endpoints_xy.dtype == np.dtype(np.float64) for row in rows)
    assert all(not row.shared_endpoints_xy.flags.writeable for row in rows)
    assert len({row.motion_profile_bytes for row in rows}) == 3


def test_materialized_task_instance_has_no_realization_or_planning_identity() -> None:
    """Break caught: a realization seed/index contaminates one shared task instance."""
    from latency_meta_mdp.expert_realization.task_instance import MaterializedTaskInstance

    names = {field.name for field in dataclasses.fields(MaterializedTaskInstance)}
    forbidden_fragments = ("realization", "strategy", "planner", "attempt")
    assert not any(fragment in name for fragment in forbidden_fragments for name in names)


def test_initial_state_mismatch_reports_first_named_field() -> None:
    """Break caught: exact replay is weakened or a mismatch cannot block on the first field."""
    from latency_meta_mdp.expert_realization.task_instance import (
        TaskInstanceReplayMismatch,
        _compare_initial_states,
    )

    expected = _state()
    changed = np.array(expected.robot_qvel, copy=True)
    changed[2] += 1.0
    actual = dataclasses.replace(expected, robot_qvel=changed)

    with pytest.raises(TaskInstanceReplayMismatch) as error:
        _compare_initial_states(expected, actual)
    assert error.value.field == "robot_qvel"


def test_task4_import_graph_excludes_historical_high_level_modules() -> None:
    """Break caught: Task 4 silently depends on old expert, collection, or recording stacks."""
    code = """
import sys
import latency_meta_mdp.expert_realization.task_instance
import latency_meta_mdp.expert_realization.shared_prefix
forbidden = {
    'latency_meta_mdp.expert',
    'latency_meta_mdp.expert_collection',
    'latency_meta_mdp.recording',
    'latency_meta_mdp.episode_artifacts',
    'latency_meta_mdp.bulk_collection',
    'latency_meta_mdp.pilot_collection',
}
loaded = sorted(forbidden.intersection(sys.modules))
assert not loaded, loaded
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path.cwd(),
        env={"PYTHONPATH": str(Path.cwd() / "src")},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
