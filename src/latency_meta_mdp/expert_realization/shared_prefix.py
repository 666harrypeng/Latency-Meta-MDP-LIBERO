"""Exact settle-open-hold prefix and K6 decision anchor for Task 4."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from latency_meta_mdp.control import ActionContract
from latency_meta_mdp.expert_realization.task_instance import (
    MaterializedTaskInstance,
    TaskInstanceReplayMismatch,
    _build_task_instance_runtime,
    _canonical_json_bytes,
    _compare_initial_states,
    _framed_arrays_sha256,
    _initial_state_from_runtime,
    _readonly_exact,
    _sha256_bytes,
    _validate_camera_sample_clock,
)
from latency_meta_mdp.snapshots import BoundarySnapshot

_ARRAY_CONTRACT: dict[str, tuple[np.dtype[Any], tuple[int | None, ...]]] = {
    "boundary_formal_ticks": (np.dtype(np.int64), (6,)),
    "boundary_physics_steps": (np.dtype(np.int64), (6,)),
    "boundary_time_us": (np.dtype(np.int64), (6,)),
    "shared_actions": (np.dtype(np.float64), (5, 7)),
    "k6_agentview_rgb": (np.dtype(np.uint8), (6, 256, 256, 3)),
    "k6_robot0_eye_in_hand_rgb": (np.dtype(np.uint8), (6, 256, 256, 3)),
    "k6_robot_proprio": (np.dtype(np.float32), (6, 16)),
    "anchor_mujoco_qpos": (np.dtype(np.float64), (None,)),
    "anchor_mujoco_qvel": (np.dtype(np.float64), (None,)),
    "anchor_mujoco_act": (np.dtype(np.float64), (None,)),
    "anchor_mujoco_ctrl": (np.dtype(np.float64), (9,)),
    "anchor_robot_qpos": (np.dtype(np.float64), (7,)),
    "anchor_robot_qvel": (np.dtype(np.float64), (7,)),
    "anchor_gripper_qpos": (np.dtype(np.float64), (2,)),
    "anchor_gripper_qvel": (np.dtype(np.float64), (2,)),
    "anchor_object_qpos": (np.dtype(np.float64), (7,)),
    "anchor_object_qvel": (np.dtype(np.float64), (6,)),
    "anchor_eef_position_world": (np.dtype(np.float64), (3,)),
    "anchor_eef_orientation_matrix_world": (np.dtype(np.float64), (3, 3)),
    "osc_origin_position_world": (np.dtype(np.float64), (3,)),
    "osc_origin_orientation_world": (np.dtype(np.float64), (3, 3)),
    "osc_goal_position_base": (np.dtype(np.float64), (3,)),
    "osc_goal_orientation_matrix_base": (np.dtype(np.float64), (3, 3)),
}


@dataclass(frozen=True)
class SharedPrefixAnchor:
    decision_source_tick: int
    boundary_formal_ticks: np.ndarray
    boundary_physics_steps: np.ndarray
    boundary_time_us: np.ndarray
    shared_actions: np.ndarray
    k6_agentview_rgb: np.ndarray
    k6_robot0_eye_in_hand_rgb: np.ndarray
    k6_robot_proprio: np.ndarray
    anchor_mujoco_qpos: np.ndarray
    anchor_mujoco_qvel: np.ndarray
    anchor_mujoco_act: np.ndarray
    anchor_mujoco_ctrl: np.ndarray
    anchor_robot_qpos: np.ndarray
    anchor_robot_qvel: np.ndarray
    anchor_gripper_qpos: np.ndarray
    anchor_gripper_qvel: np.ndarray
    anchor_object_qpos: np.ndarray
    anchor_object_qvel: np.ndarray
    anchor_eef_position_world: np.ndarray
    anchor_eef_orientation_matrix_world: np.ndarray
    osc_origin_position_world: np.ndarray
    osc_origin_orientation_world: np.ndarray
    osc_goal_position_base: np.ndarray
    osc_goal_orientation_matrix_base: np.ndarray
    handoff_state_json_utf8: bytes
    outcome_state_json_utf8: bytes
    clock_physics_step: int
    clock_formal_tick: int
    clock_time_us: int
    shared_actions_sha256: str
    k6_camera_sha256: str
    k6_proprio_sha256: str
    planning_start_sha256: str

    def __post_init__(self) -> None:
        if self.decision_source_tick != 5:
            raise ValueError("decision_source_tick must equal 5")
        for name, (dtype, shape) in _ARRAY_CONTRACT.items():
            object.__setattr__(
                self,
                name,
                _readonly_exact(getattr(self, name), dtype=dtype, shape=shape, name=name),
            )
        if not np.array_equal(self.boundary_formal_ticks, np.arange(6, dtype=np.int64)):
            raise ValueError("K6 formal ticks must be 0 through 5")
        if not np.array_equal(self.boundary_physics_steps, 10 * self.boundary_formal_ticks):
            raise ValueError("K6 physics clocks are invalid")
        if not np.array_equal(self.boundary_time_us, 20_000 * self.boundary_formal_ticks):
            raise ValueError("K6 time clocks are invalid")
        if (self.clock_physics_step, self.clock_formal_tick, self.clock_time_us) != (
            50,
            5,
            100_000,
        ):
            raise ValueError("anchor clock must be boundary 5")
        for name in (
            "handoff_state_json_utf8",
            "outcome_state_json_utf8",
        ):
            value = getattr(self, name)
            if type(value) is not bytes:
                raise TypeError(f"{name} must be bytes")
            json.loads(value)
        for name in (
            "shared_actions_sha256",
            "k6_camera_sha256",
            "k6_proprio_sha256",
            "planning_start_sha256",
        ):
            value = getattr(self, name)
            if type(value) is not str or len(value) != 64 or value != value.lower():
                raise ValueError(f"{name} must be a lowercase SHA-256")


def _prefix_actions(contract: ActionContract, *, decision_source_tick: int) -> np.ndarray:
    if decision_source_tick != 5:
        raise ValueError("decision_source_tick must equal 5")
    action = contract.compose_action(
        arm_reference=np.zeros(6, dtype=np.float64),
        gripper_command=contract.gripper_open_command,
    )
    expected = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float64)
    if action.dtype != np.float64 or not np.array_equal(action, expected):
        raise ValueError("settle_open_hold_v1 action contract drifted")
    actions = np.repeat(action[None, :], decision_source_tick, axis=0)
    actions.setflags(write=False)
    return actions


def _proprio(snapshot: BoundarySnapshot) -> np.ndarray:
    row = np.concatenate(
        [
            np.asarray(snapshot.robot_qpos, dtype=np.float64),
            np.asarray(snapshot.robot_qvel, dtype=np.float64),
            [snapshot.robot_gripper_qpos[0] - snapshot.robot_gripper_qpos[1]],
            [snapshot.robot_gripper_qvel[0] - snapshot.robot_gripper_qvel[1]],
        ]
    ).astype(np.float32)
    if row.shape != (16,):
        raise ValueError("model-visible proprio must be 16D")
    return row


def _build_shared_prefix_anchor(
    *,
    runtime: Any,
    boundaries: tuple[BoundarySnapshot, ...],
    shared_actions: np.ndarray,
    motion_profile_sha256: str,
    shared_endpoint_sha256: str,
) -> SharedPrefixAnchor:
    if len(boundaries) != 6:
        raise ValueError("K6 requires exactly boundaries 0 through 5")
    formal_ticks = np.array([row.formal_tick_index for row in boundaries], dtype=np.int64)
    physics_steps = np.array([row.physics_step_index for row in boundaries], dtype=np.int64)
    time_us = np.array([row.time_us for row in boundaries], dtype=np.int64)
    for index, boundary in enumerate(boundaries):
        for camera_name in ("agentview", "robot0_eye_in_hand"):
            _validate_camera_sample_clock(boundary, camera_name, context=f"k6[{index}]")
    agentview = np.stack([row.cameras["agentview"].rgb for row in boundaries]).astype(
        np.uint8, copy=False
    )
    wrist = np.stack([row.cameras["robot0_eye_in_hand"].rgb for row in boundaries]).astype(
        np.uint8, copy=False
    )
    proprio = np.stack([_proprio(row) for row in boundaries]).astype(np.float32, copy=False)
    anchor = boundaries[-1]
    arm = runtime.env.robots[0].part_controllers["right"]
    handoff_payload = runtime.handoff.fingerprint_payload()
    handoff_payload.pop("outcome", None)
    handoff_bytes = _canonical_json_bytes(handoff_payload)
    outcome_bytes = _canonical_json_bytes(runtime.tracker.fingerprint_payload())

    shared_hash = _framed_arrays_sha256(
        ("source_formal_ticks", formal_ticks[:-1]),
        ("shared_actions", shared_actions),
    )
    camera_hash = _framed_arrays_sha256(
        ("k6_agentview_rgb", agentview),
        ("k6_robot0_eye_in_hand_rgb", wrist),
        (
            "k6_agentview_source_physics_step",
            np.array(
                [row.cameras["agentview"].source_physics_step for row in boundaries],
                dtype=np.int64,
            ),
        ),
        (
            "k6_agentview_source_formal_tick",
            np.array(
                [row.cameras["agentview"].source_formal_tick for row in boundaries],
                dtype=np.int64,
            ),
        ),
        (
            "k6_agentview_source_time_us",
            np.array(
                [row.cameras["agentview"].source_time_us for row in boundaries], dtype=np.int64
            ),
        ),
        (
            "k6_robot0_eye_in_hand_source_physics_step",
            np.array(
                [
                    row.cameras["robot0_eye_in_hand"].source_physics_step
                    for row in boundaries
                ],
                dtype=np.int64,
            ),
        ),
        (
            "k6_robot0_eye_in_hand_source_formal_tick",
            np.array(
                [row.cameras["robot0_eye_in_hand"].source_formal_tick for row in boundaries],
                dtype=np.int64,
            ),
        ),
        (
            "k6_robot0_eye_in_hand_source_time_us",
            np.array(
                [row.cameras["robot0_eye_in_hand"].source_time_us for row in boundaries],
                dtype=np.int64,
            ),
        ),
        ("boundary_formal_ticks", formal_ticks),
        ("boundary_physics_steps", physics_steps),
        ("boundary_time_us", time_us),
    )
    proprio_hash = _framed_arrays_sha256(
        ("k6_robot_proprio", proprio),
        ("boundary_formal_ticks", formal_ticks),
        ("boundary_physics_steps", physics_steps),
        ("boundary_time_us", time_us),
    )
    planning_arrays = (
        ("anchor_mujoco_qpos", np.asarray(anchor.qpos, dtype=np.float64)),
        ("anchor_mujoco_qvel", np.asarray(anchor.qvel, dtype=np.float64)),
        ("anchor_mujoco_act", np.asarray(anchor.act, dtype=np.float64)),
        ("anchor_mujoco_ctrl", np.asarray(anchor.actuator_ctrl, dtype=np.float64)),
        ("anchor_robot_qpos", np.asarray(anchor.robot_qpos, dtype=np.float64)),
        ("anchor_robot_qvel", np.asarray(anchor.robot_qvel, dtype=np.float64)),
        ("anchor_gripper_qpos", np.asarray(anchor.robot_gripper_qpos, dtype=np.float64)),
        ("anchor_gripper_qvel", np.asarray(anchor.robot_gripper_qvel, dtype=np.float64)),
        ("anchor_object_qpos", np.asarray(anchor.object_qpos, dtype=np.float64)),
        ("anchor_object_qvel", np.asarray(anchor.object_qvel, dtype=np.float64)),
        ("anchor_eef_position_world", np.asarray(anchor.eef_pos, dtype=np.float64)),
        ("anchor_eef_orientation_matrix_world", np.asarray(anchor.eef_xmat, dtype=np.float64)),
        ("osc_origin_position_world", np.asarray(arm.origin_pos, dtype=np.float64)),
        ("osc_origin_orientation_world", np.asarray(arm.origin_ori, dtype=np.float64)),
        ("osc_goal_position_base", np.asarray(arm.goal_pos, dtype=np.float64)),
        ("osc_goal_orientation_matrix_base", np.asarray(arm.goal_ori, dtype=np.float64)),
        (
            "clock_physics_step",
            np.array(runtime.executor.ledger.physics_step_index, dtype=np.int64),
        ),
        ("clock_formal_tick", np.array(runtime.executor.ledger.formal_tick_index, dtype=np.int64)),
        ("clock_time_us", np.array(runtime.executor.ledger.time_us, dtype=np.int64)),
    )
    planning_hash = _framed_arrays_sha256(
        *planning_arrays,
        json_values=(
            handoff_payload,
            runtime.tracker.fingerprint_payload(),
            {
                "shared_actions_sha256": shared_hash,
                "k6_camera_sha256": camera_hash,
                "k6_proprio_sha256": proprio_hash,
                "motion_profile_sha256": motion_profile_sha256,
                "shared_endpoint_sha256": shared_endpoint_sha256,
            },
        ),
    )
    return SharedPrefixAnchor(
        decision_source_tick=5,
        boundary_formal_ticks=formal_ticks,
        boundary_physics_steps=physics_steps,
        boundary_time_us=time_us,
        shared_actions=np.asarray(shared_actions, dtype=np.float64),
        k6_agentview_rgb=agentview,
        k6_robot0_eye_in_hand_rgb=wrist,
        k6_robot_proprio=proprio,
        anchor_mujoco_qpos=planning_arrays[0][1],
        anchor_mujoco_qvel=planning_arrays[1][1],
        anchor_mujoco_act=planning_arrays[2][1],
        anchor_mujoco_ctrl=planning_arrays[3][1],
        anchor_robot_qpos=planning_arrays[4][1],
        anchor_robot_qvel=planning_arrays[5][1],
        anchor_gripper_qpos=planning_arrays[6][1],
        anchor_gripper_qvel=planning_arrays[7][1],
        anchor_object_qpos=planning_arrays[8][1],
        anchor_object_qvel=planning_arrays[9][1],
        anchor_eef_position_world=planning_arrays[10][1],
        anchor_eef_orientation_matrix_world=planning_arrays[11][1],
        osc_origin_position_world=planning_arrays[12][1],
        osc_origin_orientation_world=planning_arrays[13][1],
        osc_goal_position_base=planning_arrays[14][1],
        osc_goal_orientation_matrix_base=planning_arrays[15][1],
        handoff_state_json_utf8=handoff_bytes,
        outcome_state_json_utf8=outcome_bytes,
        clock_physics_step=runtime.executor.ledger.physics_step_index,
        clock_formal_tick=runtime.executor.ledger.formal_tick_index,
        clock_time_us=runtime.executor.ledger.time_us,
        shared_actions_sha256=shared_hash,
        k6_camera_sha256=camera_hash,
        k6_proprio_sha256=proprio_hash,
        planning_start_sha256=planning_hash,
    )


def _execute_shared_prefix(
    *,
    runtime: Any,
    boundary_zero: BoundarySnapshot,
    decision_source_tick: int,
    motion_profile_sha256: str,
    shared_endpoint_sha256: str,
) -> SharedPrefixAnchor:
    actions = _prefix_actions(runtime.action_contract, decision_source_tick=decision_source_tick)
    boundaries = [boundary_zero]
    for action in actions:
        boundaries.append(runtime.executor.step_formal(action))
    return _build_shared_prefix_anchor(
        runtime=runtime,
        boundaries=tuple(boundaries),
        shared_actions=actions,
        motion_profile_sha256=motion_profile_sha256,
        shared_endpoint_sha256=shared_endpoint_sha256,
    )


def _array_mismatch_field(name: str, expected: np.ndarray, actual: np.ndarray) -> str:
    if expected.ndim and expected.shape == actual.shape and expected.shape[0] in (5, 6):
        for index in range(expected.shape[0]):
            if not np.array_equal(expected[index], actual[index]):
                return f"{name}[{index}]"
    aliases = {
        "anchor_mujoco_qpos": "anchor.mujoco_qpos",
        "anchor_mujoco_qvel": "anchor.mujoco_qvel",
        "anchor_mujoco_act": "anchor.mujoco_act",
        "anchor_mujoco_ctrl": "anchor.mujoco_ctrl",
        "anchor_robot_qpos": "anchor.robot_qpos",
        "anchor_robot_qvel": "anchor.robot_qvel",
        "anchor_gripper_qpos": "anchor.gripper_qpos",
        "anchor_gripper_qvel": "anchor.gripper_qvel",
        "anchor_object_qpos": "anchor.object_qpos",
        "anchor_object_qvel": "anchor.object_qvel",
        "anchor_eef_position_world": "anchor.eef_position_world",
        "anchor_eef_orientation_matrix_world": "anchor.eef_orientation_matrix_world",
        "osc_origin_position_world": "anchor.osc_origin_position_world",
        "osc_origin_orientation_world": "anchor.osc_origin_orientation_world",
        "osc_goal_position_base": "anchor.osc_goal_position_base",
        "osc_goal_orientation_matrix_base": "anchor.osc_goal_orientation_matrix_base",
    }
    return aliases.get(name, name)


def _first_json_mismatch_path(prefix: str, expected: Any, actual: Any) -> str | None:
    if type(expected) is not type(actual):
        return prefix
    if isinstance(expected, dict):
        keys = sorted(set(expected) | set(actual))
        for key in keys:
            path = f"{prefix}.{key}"
            if key not in expected or key not in actual:
                return path
            mismatch = _first_json_mismatch_path(path, expected[key], actual[key])
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return f"{prefix}.length"
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            mismatch = _first_json_mismatch_path(f"{prefix}[{index}]", left, right)
            if mismatch is not None:
                return mismatch
        return None
    return None if expected == actual else prefix


def _json_bytes_mismatch_path(prefix: str, expected: bytes, actual: bytes) -> str:
    try:
        expected_value = json.loads(expected)
        actual_value = json.loads(actual)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return prefix
    return _first_json_mismatch_path(prefix, expected_value, actual_value) or prefix


def _compare_shared_prefix_anchors(
    expected: SharedPrefixAnchor, actual: SharedPrefixAnchor
) -> None:
    for field in fields(SharedPrefixAnchor):
        name = field.name
        left = getattr(expected, name)
        right = getattr(actual, name)
        if isinstance(left, np.ndarray):
            if not isinstance(right, np.ndarray) or not np.array_equal(left, right):
                raise TaskInstanceReplayMismatch(
                    field=_array_mismatch_field(name, left, np.asarray(right))
                )
        elif left != right:
            if name == "handoff_state_json_utf8":
                mismatch = _json_bytes_mismatch_path("handoff", left, right)
            elif name == "outcome_state_json_utf8":
                mismatch = _json_bytes_mismatch_path("outcome", left, right)
            else:
                mismatch = {
                    "clock_physics_step": "anchor.clock_physics_step",
                    "clock_formal_tick": "anchor.clock_formal_tick",
                    "clock_time_us": "anchor.clock_time_us",
                }.get(name, name)
            raise TaskInstanceReplayMismatch(field=mismatch)


def replay_shared_prefix(
    task_instance: MaterializedTaskInstance, *, decision_source_tick: int
) -> SharedPrefixAnchor:
    if decision_source_tick != 5:
        raise ValueError("decision_source_tick must equal 5")
    if not isinstance(task_instance, MaterializedTaskInstance):
        raise TypeError("task_instance must be a MaterializedTaskInstance")
    task_instance.validate_publication_consistency()
    runtime = _build_task_instance_runtime(task_instance)
    try:
        boundary_zero = runtime.executor.initialize()
        actual_initial = _initial_state_from_runtime(runtime, boundary_zero)
        _compare_initial_states(task_instance.initial_state, actual_initial)
        anchor = _execute_shared_prefix(
            runtime=runtime,
            boundary_zero=boundary_zero,
            decision_source_tick=decision_source_tick,
            motion_profile_sha256=task_instance.task_instance_id.motion_profile_sha256,
            shared_endpoint_sha256=_sha256_bytes(task_instance.shared_endpoint_bytes),
        )
        _compare_shared_prefix_anchors(task_instance.expected_anchor, anchor)
        for name, expected in task_instance.expected_anchor_fingerprints.items():
            if getattr(anchor, name) != expected:
                raise TaskInstanceReplayMismatch(field=name)
        return anchor
    finally:
        runtime.close()
