from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def make_shared_prefix_anchor():
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


def make_detached_task_instance(level: int = 3, seed: int = 4000):
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
    object.__setattr__(instance, "expected_anchor", make_shared_prefix_anchor())
    return instance


def strategy_config_and_keys(instance: object):
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
