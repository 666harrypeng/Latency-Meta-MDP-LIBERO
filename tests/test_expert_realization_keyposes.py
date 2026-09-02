from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest


def _task_instance(level: int = 3, seed: int = 4000):
    from latency_meta_mdp.expert_realization.contracts import TaskInstanceId
    from latency_meta_mdp.expert_realization.task_instance import (
        MaterializedTaskInstance,
        _materialize_motion_profile,
    )

    motion = _materialize_motion_profile(
        project_root=Path.cwd(),
        level=level,
        task_instance_seed=seed,
    )
    instance = object.__new__(MaterializedTaskInstance)
    object.__setattr__(instance, "project_root", Path.cwd())
    object.__setattr__(
        instance,
        "task_instance_id",
        TaskInstanceId(
            level,
            seed,
            hashlib.sha256(motion.motion_profile_bytes).hexdigest(),
            "b" * 64,
        ),
    )
    object.__setattr__(instance, "motion_profile_mapping", motion.motion_profile_mapping)
    object.__setattr__(instance, "decision_source_tick", 5)
    object.__setattr__(instance, "expected_anchor", _anchor())
    return instance


def _anchor():
    from latency_meta_mdp.expert_realization.shared_prefix import SharedPrefixAnchor

    f64 = np.float64
    return SharedPrefixAnchor(
        decision_source_tick=5,
        boundary_formal_ticks=np.arange(6, dtype=np.int64),
        boundary_physics_steps=10 * np.arange(6, dtype=np.int64),
        boundary_time_us=20_000 * np.arange(6, dtype=np.int64),
        shared_actions=np.tile([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], (5, 1)).astype(f64),
        k6_agentview_rgb=np.zeros((6, 256, 256, 3), dtype=np.uint8),
        k6_robot0_eye_in_hand_rgb=np.zeros((6, 256, 256, 3), dtype=np.uint8),
        k6_robot_proprio=np.zeros((6, 16), dtype=np.float32),
        anchor_mujoco_qpos=np.zeros(16, dtype=f64),
        anchor_mujoco_qvel=np.zeros(15, dtype=f64),
        anchor_mujoco_act=np.empty(0, dtype=f64),
        anchor_mujoco_ctrl=np.zeros(9, dtype=f64),
        anchor_robot_qpos=np.zeros(7, dtype=f64),
        anchor_robot_qvel=np.zeros(7, dtype=f64),
        anchor_gripper_qpos=np.zeros(2, dtype=f64),
        anchor_gripper_qvel=np.zeros(2, dtype=f64),
        anchor_object_qpos=np.array([0.0, -0.12, 0.833, 1.0, 0.0, 0.0, 0.0], dtype=f64),
        anchor_object_qvel=np.zeros(6, dtype=f64),
        anchor_eef_position_world=np.array([0.0, -0.20, 1.0], dtype=f64),
        anchor_eef_orientation_matrix_world=np.eye(3, dtype=f64),
        osc_origin_position_world=np.zeros(3, dtype=f64),
        osc_origin_orientation_world=np.eye(3, dtype=f64),
        osc_goal_position_base=np.zeros(3, dtype=f64),
        osc_goal_orientation_matrix_base=np.eye(3, dtype=f64),
        handoff_state_json_utf8=b'{"state":"driven"}\n',
        outcome_state_json_utf8=b'{"status":"running"}\n',
        clock_physics_step=50,
        clock_formal_tick=5,
        clock_time_us=100_000,
        shared_actions_sha256="a" * 64,
        k6_camera_sha256="b" * 64,
        k6_proprio_sha256="c" * 64,
        planning_start_sha256="d" * 64,
    )


def _config_and_keys(instance: object):
    from latency_meta_mdp.expert_realization.contracts import ExpertRealizationKey
    from latency_meta_mdp.expert_realization.strategy import StructuredStrategyConfig

    config = StructuredStrategyConfig.from_path(
        Path.cwd() / "configs/expert_realization/panda_ball_structured.yaml"
    )
    keys = tuple(
        ExpertRealizationKey(instance.task_instance_id, index, config.source_sha256)
        for index in range(8)
    )
    return config, keys


@pytest.fixture(autouse=True)
def _validated_task_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    from latency_meta_mdp.expert_realization.task_instance import MaterializedTaskInstance

    monkeypatch.setattr(
        MaterializedTaskInstance,
        "validate_publication_consistency",
        lambda self: None,
    )


def test_all_family_keyposes_are_typed_bounded_and_keep_fixed_orientation() -> None:
    """Break caught: structured diversity generates unsafe geometry or orientation noise."""
    from latency_meta_mdp.expert_realization.keyposes import build_interception_keyposes
    from latency_meta_mdp.expert_realization.strategy import sample_strategy

    instance = _task_instance()
    anchor = _anchor()
    config, keys = _config_and_keys(instance)
    plans = [
        build_interception_keyposes(instance, anchor, sample_strategy(instance, key, config))
        for key in keys
    ]

    assert [plan.family.value for plan in plans] == [
        "canonical_direct",
        "canonical_direct",
        "early_high_arc",
        "early_high_arc",
        "lateral_arc",
        "lateral_arc",
        "time_shifted_smooth",
        "time_shifted_smooth",
    ]
    for plan in plans:
        assert plan.interception_time_us == plan.interception_tick * 20_000
        assert plan.pregrasp_arrival_tick > 5
        assert plan.pregrasp_arrival_tick < plan.interception_tick
        assert plan.interception_tick + plan.close_dwell_ticks + 20 <= 150
        assert np.array_equal(
            plan.fixed_orientation_world,
            anchor.anchor_eef_orientation_matrix_world,
        )
        assert plan.requires_physical_handoff is True
        assert plan.rotation_action_variation is False
        assert plan.iid_per_tick_action_noise is False
        for point in (
            plan.initial_eef_position_world,
            *plan.guide_positions_world,
            plan.pregrasp_position_world,
            plan.grasp_affordance_position_world,
            plan.lift_target_world,
        ):
            assert abs(point[0]) <= 0.37
            assert abs(point[1]) <= 0.37
            assert 0.8 <= point[2] <= 1.4
        assert not plan.guide_positions_world.flags.writeable
        assert not plan.fixed_orientation_world.flags.writeable

    assert plans[0].guide_positions_world.shape == (0, 3)
    assert plans[2].guide_positions_world.shape == (1, 3)
    assert plans[4].guide_positions_world.shape == (1, 3)
    assert plans[6].guide_positions_world.shape == (0, 3)
    assert plans[6].time_scaling_profile == "minimum_jerk_slow"

    for plan in plans[4:6]:
        midpoint = 0.5 * (anchor.anchor_eef_position_world + plan.pregrasp_position_world)
        signed_offset = np.dot(
            plan.guide_positions_world[0, :2] - midpoint[:2],
            plan.lateral_direction_xy,
        )
        assert np.isclose(
            signed_offset,
            plan.strategy.lateral_direction_sign * plan.strategy.lateral_offset_m,
        )
    for plan in plans:
        lift_offset = plan.lift_target_world[:2] - plan.grasp_affordance_position_world[:2]
        assert np.isclose(
            np.dot(lift_offset, plan.lateral_direction_xy),
            plan.strategy.lift_lateral_direction_sign
            * plan.strategy.lift_lateral_offset_m,
        )


def test_keyposes_reject_anchor_from_another_task_instance() -> None:
    """Break caught: planning starts from a valid but different task-instance anchor."""
    from latency_meta_mdp.expert_realization.keyposes import build_interception_keyposes
    from latency_meta_mdp.expert_realization.strategy import sample_strategy
    from latency_meta_mdp.expert_realization.task_instance import TaskInstanceReplayMismatch

    instance = _task_instance()
    config, keys = _config_and_keys(instance)
    wrong_anchor = __import__("dataclasses").replace(
        instance.expected_anchor,
        anchor_robot_qpos=np.ones(7, dtype=np.float64),
    )

    with pytest.raises(TaskInstanceReplayMismatch, match="anchor.robot_qpos"):
        build_interception_keyposes(
            instance,
            wrong_anchor,
            sample_strategy(instance, keys[0], config),
        )


def test_l3_formal_tick_sampling_switches_velocity_at_exact_segment_boundary() -> None:
    """Break caught: L3 keyposes query the old segment one formal tick after direction change."""
    from latency_meta_mdp.expert_realization.keyposes import _sample_profile_at_tick

    instance = _task_instance(level=3)
    before = _sample_profile_at_tick(instance, 48)
    boundary = _sample_profile_at_tick(instance, 49)
    after = _sample_profile_at_tick(instance, 50)
    segments = instance.motion_profile_mapping["segments"]

    assert before.segment_index == 0
    assert boundary.segment_index == 1
    assert after.segment_index == 1
    np.testing.assert_allclose(before.velocity[:2], segments[0]["end_velocity_xy"], atol=1e-15)
    np.testing.assert_allclose(boundary.position[:2], segments[0]["end_xy"], atol=1e-15)
    np.testing.assert_allclose(
        boundary.velocity[:2], segments[1]["start_velocity_xy"], atol=1e-15
    )
    np.testing.assert_allclose(after.velocity[:2], segments[1]["start_velocity_xy"], atol=1e-15)


def test_close_dwell_is_only_a_minimum_and_never_claims_handoff() -> None:
    """Break caught: a timer is treated as equivalent to physical robot-object handoff."""
    from latency_meta_mdp.expert_realization.keyposes import build_interception_keyposes
    from latency_meta_mdp.expert_realization.strategy import sample_strategy

    instance = _task_instance()
    config, keys = _config_and_keys(instance)
    plan = build_interception_keyposes(
        instance,
        _anchor(),
        sample_strategy(instance, keys[-1], config),
    )

    assert plan.close_dwell_ticks >= 0
    assert plan.requires_physical_handoff is True
    assert "action" not in {field.name for field in __import__("dataclasses").fields(plan)}
