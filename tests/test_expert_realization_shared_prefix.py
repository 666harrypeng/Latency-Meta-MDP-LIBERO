from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from latency_meta_mdp.envs.snapshots import BoundarySnapshot, CameraSample
from latency_meta_mdp.runtime.timing import ClockLedger


def _camera(name: str, tick: int) -> CameraSample:
    rgb = np.full((256, 256, 3), tick + (17 if name == "agentview" else 71), dtype=np.uint8)
    rgb.setflags(write=False)
    return CameraSample(
        name=name,
        rgb=rgb,
        segmentation=None,
        source_physics_step=10 * tick,
        source_formal_tick=tick,
        source_time_us=20_000 * tick,
    )


def _boundary(tick: int) -> BoundarySnapshot:
    f64 = np.float64
    return BoundarySnapshot(
        physics_step_index=10 * tick,
        formal_tick_index=tick,
        time_us=20_000 * tick,
        sim_time_seconds=0.02 * tick,
        qpos=np.arange(16, dtype=f64) + tick,
        qvel=np.arange(15, dtype=f64) + tick,
        act=np.empty((0,), dtype=f64),
        actuator_ctrl=np.arange(9, dtype=f64) + tick,
        robot_qpos=np.arange(7, dtype=f64) + tick,
        robot_qvel=np.arange(7, dtype=f64) + tick,
        robot_gripper_qpos=np.array([0.02, -0.02], dtype=f64),
        robot_gripper_qvel=np.array([0.001, -0.001], dtype=f64),
        eef_pos=np.arange(3, dtype=f64) + tick,
        eef_xmat=np.eye(3, dtype=f64),
        object_qpos=np.array([tick, -0.12, 0.833, 1.0, 0.0, 0.0, 0.0], dtype=f64),
        object_qvel=np.arange(6, dtype=f64) + tick,
        object_body_pos=np.zeros(3, dtype=f64),
        object_body_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=f64),
        commanded_world={},
        cameras={
            "agentview": _camera("agentview", tick),
            "robot0_eye_in_hand": _camera("robot0_eye_in_hand", tick),
        },
    )


def _boundary_with_camera_clock_change(
    tick: int, *, camera_name: str, field: str, value: object
) -> BoundarySnapshot:
    boundary = _boundary(tick)
    cameras = dict(boundary.cameras)
    cameras[camera_name] = dataclasses.replace(cameras[camera_name], **{field: value})
    return dataclasses.replace(boundary, cameras=cameras)


class _Payload:
    def __init__(self, value: dict[str, object]) -> None:
        self.value = value

    def fingerprint_payload(self) -> dict[str, object]:
        return dict(self.value)


def _runtime() -> SimpleNamespace:
    arm = SimpleNamespace(
        origin_pos=np.array([0.1, 0.2, 0.3], dtype=np.float64),
        origin_ori=np.eye(3, dtype=np.float64),
        goal_pos=np.array([0.4, 0.5, 0.6], dtype=np.float64),
        goal_ori=np.eye(3, dtype=np.float64),
    )
    ledger = ClockLedger(physics_dt_us=2_000, formal_tick_us=20_000)
    for step in range(50):
        ledger.record_successful_step(sim_time_seconds=(step + 1) * 0.002)
    outcome = _Payload({"status": "running", "last_boundary_time_us": 100_000})
    handoff = _Payload(
        {
            "state": "driven",
            "last_contact": {"time_us": 100_000},
            "outcome": outcome.fingerprint_payload(),
        }
    )
    return SimpleNamespace(
        env=SimpleNamespace(
            robots=[SimpleNamespace(part_controllers={"right": arm})],
        ),
        tracker=outcome,
        handoff=handoff,
        executor=SimpleNamespace(ledger=ledger),
    )


def _anchor():
    from latency_meta_mdp.data.collection.shared_prefix import (
        _build_shared_prefix_anchor,
        _prefix_actions,
    )
    from latency_meta_mdp.envs.control import load_action_contract

    contract = load_action_contract(
        Path.cwd() / "configs/runtime/control/panda_osc_pose_delta_v1.yaml"
    )
    actions = _prefix_actions(contract, decision_source_tick=5)
    return _build_shared_prefix_anchor(
        runtime=_runtime(),
        boundaries=tuple(_boundary(tick) for tick in range(6)),
        shared_actions=actions,
        motion_profile_sha256="a" * 64,
        shared_endpoint_sha256="b" * 64,
    )


def test_prefix_is_exactly_five_open_gripper_zero_delta_actions() -> None:
    """Break caught: zero gripper, six actions, or a family action leaks before boundary 5."""
    from latency_meta_mdp.data.collection.shared_prefix import _prefix_actions
    from latency_meta_mdp.envs.control import load_action_contract

    contract = load_action_contract(
        Path.cwd() / "configs/runtime/control/panda_osc_pose_delta_v1.yaml"
    )
    actions = _prefix_actions(contract, decision_source_tick=5)

    assert actions.shape == (5, 7)
    assert actions.dtype == np.dtype(np.float64)
    assert np.array_equal(actions, np.tile([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], (5, 1)))
    assert not actions.flags.writeable


def test_k6_anchor_uses_boundaries_zero_through_five_and_exact_contracts() -> None:
    """Break caught: K6 ends at boundary 4/6 or changes publication dtypes and shapes."""
    anchor = _anchor()

    assert np.array_equal(anchor.boundary_formal_ticks, np.arange(6, dtype=np.int64))
    assert np.array_equal(anchor.boundary_physics_steps, 10 * np.arange(6, dtype=np.int64))
    assert np.array_equal(anchor.boundary_time_us, 20_000 * np.arange(6, dtype=np.int64))
    assert anchor.shared_actions.shape == (5, 7)
    assert anchor.k6_agentview_rgb.shape == (6, 256, 256, 3)
    assert anchor.k6_robot0_eye_in_hand_rgb.shape == (6, 256, 256, 3)
    assert anchor.k6_agentview_rgb.dtype == np.dtype(np.uint8)
    assert anchor.k6_robot0_eye_in_hand_rgb.dtype == np.dtype(np.uint8)
    assert anchor.k6_robot_proprio.shape == (6, 16)
    assert anchor.k6_robot_proprio.dtype == np.dtype(np.float32)
    assert anchor.clock_physics_step == 50
    assert anchor.clock_formal_tick == 5
    assert anchor.clock_time_us == 100_000
    assert anchor.decision_source_tick == 5
    assert anchor.k6_robot_proprio[0, -2] == np.float32(0.04)
    assert anchor.k6_robot_proprio[0, -1] == np.float32(0.002)
    for field in dataclasses.fields(anchor):
        value = getattr(anchor, field.name)
        if isinstance(value, np.ndarray):
            assert not value.flags.writeable, field.name
    for name in (
        "shared_actions_sha256",
        "k6_camera_sha256",
        "k6_proprio_sha256",
        "planning_start_sha256",
    ):
        value = getattr(anchor, name)
        assert len(value) == 64
        assert value == value.lower()


@pytest.mark.parametrize("camera_name", ("agentview", "robot0_eye_in_hand"))
@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("name", "wrong_view"),
        ("source_physics_step", 999),
        ("source_formal_tick", 999),
        ("source_time_us", 999_999),
    ),
)
def test_k6_rejects_each_camera_source_clock_mismatch(
    camera_name: str, field: str, bad_value: object
) -> None:
    """Break caught: stale image bytes cannot masquerade as a synchronized K6 camera sample."""
    from latency_meta_mdp.data.collection.shared_prefix import (
        _build_shared_prefix_anchor,
        _prefix_actions,
    )
    from latency_meta_mdp.data.collection.task_instance import TaskInstanceReplayMismatch
    from latency_meta_mdp.envs.control import load_action_contract

    contract = load_action_contract(
        Path.cwd() / "configs/runtime/control/panda_osc_pose_delta_v1.yaml"
    )
    boundaries = tuple(
        _boundary_with_camera_clock_change(
            tick,
            camera_name=camera_name,
            field=field,
            value=bad_value,
        )
        if tick == 3
        else _boundary(tick)
        for tick in range(6)
    )

    with pytest.raises(TaskInstanceReplayMismatch) as error:
        _build_shared_prefix_anchor(
            runtime=_runtime(),
            boundaries=boundaries,
            shared_actions=_prefix_actions(contract, decision_source_tick=5),
            motion_profile_sha256="a" * 64,
            shared_endpoint_sha256="b" * 64,
        )

    assert error.value.field == f"k6[3].{camera_name}.{field}"


def test_boundary_zero_rejects_camera_source_clock_mismatch() -> None:
    """Break caught: the TaskInstanceId cannot bind a mislabeled boundary-zero image."""
    from latency_meta_mdp.data.collection.task_instance import (
        TaskInstanceReplayMismatch,
        _initial_state_from_runtime,
    )

    boundary = _boundary_with_camera_clock_change(
        0,
        camera_name="agentview",
        field="source_time_us",
        value=20_000,
    )

    runtime = _runtime()
    runtime.require_open = lambda: None
    runtime.executor = SimpleNamespace(
        ledger=ClockLedger(physics_dt_us=2_000, formal_tick_us=20_000)
    )
    with pytest.raises(TaskInstanceReplayMismatch) as error:
        _initial_state_from_runtime(runtime, boundary)

    assert error.value.field == "boundary0.agentview.source_time_us"


def test_anchor_mismatch_names_first_camera_tick_and_blocks_return() -> None:
    """Break caught: bitwise K6 corruption is reduced to a tolerance or generic hash error."""
    from latency_meta_mdp.data.collection.shared_prefix import _compare_shared_prefix_anchors
    from latency_meta_mdp.data.collection.task_instance import TaskInstanceReplayMismatch

    expected = _anchor()
    changed = np.array(expected.k6_agentview_rgb, copy=True)
    changed[3, 10, 20, 1] ^= np.uint8(1)
    actual = dataclasses.replace(expected, k6_agentview_rgb=changed)

    with pytest.raises(TaskInstanceReplayMismatch) as error:
        _compare_shared_prefix_anchors(expected, actual)
    assert error.value.field == "k6_agentview_rgb[3]"


@pytest.mark.parametrize(
    ("field", "replacement", "expected_name"),
    (
        ("shared_actions", np.zeros((5, 7), dtype=np.float64), "shared_actions[0]"),
        ("k6_robot_proprio", np.zeros((6, 16), dtype=np.float32), "k6_robot_proprio[0]"),
        ("anchor_mujoco_ctrl", np.zeros(9, dtype=np.float64), "anchor.mujoco_ctrl"),
    ),
)
def test_anchor_mismatch_reports_each_top_level_category(
    field: str, replacement: object, expected_name: str
) -> None:
    """Break caught: a top-level replay category can change without a precise blocking field."""
    from latency_meta_mdp.data.collection.shared_prefix import _compare_shared_prefix_anchors
    from latency_meta_mdp.data.collection.task_instance import TaskInstanceReplayMismatch

    expected = _anchor()
    actual = dataclasses.replace(expected, **{field: replacement})
    with pytest.raises(TaskInstanceReplayMismatch) as error:
        _compare_shared_prefix_anchors(expected, actual)
    assert error.value.field == expected_name


def test_handoff_mismatch_reports_exact_json_path() -> None:
    """Break caught: a handoff-state divergence is localized to its causal JSON field."""
    from latency_meta_mdp.data.collection.shared_prefix import _compare_shared_prefix_anchors
    from latency_meta_mdp.data.collection.task_instance import TaskInstanceReplayMismatch

    expected = _anchor()
    payload = json.loads(expected.handoff_state_json_utf8)
    payload["state"] = "physical"
    actual = dataclasses.replace(
        expected,
        handoff_state_json_utf8=(json.dumps(payload, sort_keys=True) + "\n").encode(),
    )

    with pytest.raises(TaskInstanceReplayMismatch) as error:
        _compare_shared_prefix_anchors(expected, actual)

    assert error.value.field == "handoff.state"


def test_nested_outcome_mismatch_reports_exact_json_path() -> None:
    """Break caught: a nested causal-state divergence is not hidden behind a generic label."""
    from latency_meta_mdp.data.collection.shared_prefix import _compare_shared_prefix_anchors
    from latency_meta_mdp.data.collection.task_instance import TaskInstanceReplayMismatch

    expected = _anchor()
    payload = json.loads(expected.outcome_state_json_utf8)
    payload["last_boundary_time_us"] += 20_000
    actual = dataclasses.replace(
        expected,
        outcome_state_json_utf8=(json.dumps(payload, sort_keys=True) + "\n").encode(),
    )

    with pytest.raises(TaskInstanceReplayMismatch) as error:
        _compare_shared_prefix_anchors(expected, actual)

    assert error.value.field == "outcome.last_boundary_time_us"


def test_runtime_close_is_idempotent_and_closes_environment_once() -> None:
    """Break caught: success/failure cleanup double-closes or leaks a fresh renderer runtime."""
    from latency_meta_mdp.data.collection.task_instance import _TaskInstanceRuntime

    class Env:
        def __init__(self) -> None:
            self.close_count = 0

        def close(self) -> None:
            self.close_count += 1

    env = Env()
    runtime = _TaskInstanceRuntime(
        env=env,
        action_contract=object(),
        tracker=object(),
        handoff=object(),
        world=object(),
        executor=object(),
    )

    runtime.close()
    runtime.close()

    assert env.close_count == 1
    assert runtime.closed is True
    with pytest.raises(RuntimeError, match="closed"):
        runtime.require_open()


@pytest.mark.parametrize("tick", (0, 4, 6))
def test_invalid_decision_tick_rejected_before_runtime_construction(
    monkeypatch: pytest.MonkeyPatch, tick: int
) -> None:
    """Break caught: an invalid decision tick creates or advances a renderer before rejection."""
    from latency_meta_mdp.data.collection import shared_prefix

    called = False

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("runtime must not be constructed")

    monkeypatch.setattr(shared_prefix, "_build_task_instance_runtime", forbidden)
    with pytest.raises(ValueError, match="decision_source_tick"):
        shared_prefix.replay_shared_prefix(object(), decision_source_tick=tick)
    assert called is False
