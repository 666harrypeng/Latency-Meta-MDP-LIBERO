from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def make_shared_prefix_anchor():
    from latency_meta_mdp.data.collection.shared_prefix import SharedPrefixAnchor

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
    from latency_meta_mdp.data.collection.contracts import TaskInstanceId
    from latency_meta_mdp.data.collection.task_instance import (
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
    from latency_meta_mdp.data.collection.contracts import ExpertRealizationKey
    from latency_meta_mdp.data.collection.strategy import StructuredStrategyConfig

    config = StructuredStrategyConfig.from_path(
        Path.cwd() / "configs/data/expert_realization/panda_ball_structured.yaml"
    )
    keys = tuple(
        ExpertRealizationKey(instance.task_instance_id, index, config.source_sha256)
        for index in range(8)
    )
    return config, keys


def make_formal_source_metadata(
    *,
    camera_height: int = 2,
    camera_width: int = 3,
    schema_version: int = 2,
    accepted_slot: int = 0,
    realization_draw_index: int = 0,
):
    from latency_meta_mdp.data.collection.contracts import (
        ExpertRealizationId,
        ExpertRealizationKey,
        StrategyFamily,
        TaskInstanceId,
    )
    from latency_meta_mdp.data.collection.recording_contracts import ImplementationIdentity
    from latency_meta_mdp.data.source.contracts import (
        FormalSourceEpisodeMetadata,
    )

    task = TaskInstanceId(1, 4000, "a" * 64, "b" * 64)
    realization = ExpertRealizationId(
        ExpertRealizationKey(task, realization_draw_index, "c" * 64),
        "d" * 64,
    )
    kwargs = {}
    if schema_version == 3:
        kwargs = {
            "accepted_slot": accepted_slot,
            "realization_draw_index": realization_draw_index,
        }
    return FormalSourceEpisodeMetadata(
        schema_version=schema_version,
        record_profile="formal_source",
        episode_id="source-l1-task000-r000",
        corpus_id="panda-ball-structured-source-pilot",
        logical_master_task_index=0,
        task_instance_id=task,
        expert_realization_id=realization,
        task_id="dynamic_grasp_lift",
        instruction="Grasp the moving ball and lift it.",
        physics_dt_us=2_000,
        formal_tick_us=20_000,
        camera_height=camera_height,
        camera_width=camera_width,
        action_contract_id="panda_osc_pose_delta_v1",
        action_dim=7,
        actuator_dim=9,
        expert_id="panda_ball_smooth_approach_canonical_grasp_v3",
        strategy_family=StrategyFamily.CANONICAL_DIRECT,
        formal_corpus_config_sha256="e" * 64,
        source_corpus_config_sha256="f" * 64,
        task_config_sha256="1" * 64,
        motion_config_sha256="2" * 64,
        runtime_config_sha256="3" * 64,
        controller_config_sha256="4" * 64,
        structured_expert_config_sha256="f" * 64,
        curobo_planner_config_sha256="5" * 64,
        task_instance_manifest_sha256="6" * 64,
        frozen_plan_set_manifest_sha256="d" * 64,
        realization_universe_sha256="7" * 64,
        strategy_sha256="8" * 64,
        planner_candidates_sha256="9" * 64,
        selected_reference_sha256="a" * 64,
        implementation=ImplementationIdentity("1" * 40, "b" * 64, False),
        **kwargs,
    )


def make_formal_source_records(metadata, *, boundary_count: int = 2):
    from latency_meta_mdp.data.collection.recording_contracts import (
        StructuredBoundaryRecord,
        StructuredDeploymentRecord,
        StructuredExpertAuditRecord,
        StructuredPhysicalEventRecord,
        StructuredQualificationRecord,
        StructuredTransitionRecord,
    )

    if type(boundary_count) is not int or boundary_count < 2:
        raise ValueError("boundary_count must be at least two")
    terminal_tick = boundary_count - 1
    first_contact_tick = max(0, terminal_tick - 4)
    stable_grasp_tick = max(first_contact_tick, terminal_tick - 3)
    handoff_tick = stable_grasp_tick
    lift_tick = max(handoff_tick, terminal_tick - 1)
    boundaries = []
    for tick in range(boundary_count):
        has_previous = tick > 0
        terminal = tick == terminal_tick
        contact = tick >= first_contact_tick
        physical = tick >= handoff_tick
        boundaries.append(
            StructuredBoundaryRecord(
                formal_tick_index=tick,
                physics_step_index=tick * 10,
                time_us=tick * 20_000,
                deployment=StructuredDeploymentRecord(
                    source_physics_step=tick * 10,
                    source_formal_tick=tick,
                    source_time_us=tick * 20_000,
                    agentview_rgb=np.full(
                        (metadata.camera_height, metadata.camera_width, 3),
                        tick,
                        dtype=np.uint8,
                    ),
                    robot0_eye_in_hand_rgb=np.full(
                        (metadata.camera_height, metadata.camera_width, 3),
                        tick + 1,
                        dtype=np.uint8,
                    ),
                    robot_qpos=np.arange(7, dtype=np.float64) + tick * 0.01,
                    robot_qvel=np.full(7, 0.5, dtype=np.float64),
                    gripper_qpos=np.zeros(2, dtype=np.float64),
                    gripper_qvel=np.zeros(2, dtype=np.float64),
                    eef_position_world=np.array([0.4 + tick * 0.001, 0.0, 0.3], dtype=np.float64),
                    eef_orientation_matrix_world=np.eye(3, dtype=np.float64),
                ),
                qualification=StructuredQualificationRecord(
                    object_pose=np.array(
                        [0.5 + tick * 0.002, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0],
                        dtype=np.float64,
                    ),
                    object_velocity=np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
                    commanded_motion_position=np.array(
                        [0.5 + tick * 0.002, 0.0, 0.2], dtype=np.float64
                    ),
                    commanded_motion_velocity=np.array([0.1, 0.0, 0.0], dtype=np.float64),
                    commanded_motion_acceleration=np.zeros(3, dtype=np.float64),
                    commanded_motion_segment_index=0,
                    left_pad_contact=contact,
                    right_pad_contact=contact,
                    handoff_state="physical" if physical else "driven",
                    relative_geometry=np.zeros(3, dtype=np.float64),
                    actuator_ctrl=np.zeros(9, dtype=np.float64),
                    applied_reference=(
                        np.array(
                            [(tick - 1) * 0.01, 0, 0, 0, 0, 0, -1],
                            dtype=np.float64,
                        )
                        if has_previous
                        else None
                    ),
                    applied_reference_source_tick=tick - 1 if has_previous else None,
                    nullspace_joint_position_error=(
                        np.zeros(7, dtype=np.float64) if has_previous else None
                    ),
                    eef_position_error=(np.zeros(3, dtype=np.float64) if has_previous else None),
                    eef_orientation_error_rotvec=(
                        np.zeros(3, dtype=np.float64) if has_previous else None
                    ),
                ),
                outcome_status="success" if terminal else "running",
            )
        )
    transitions = tuple(
        StructuredTransitionRecord(
            source_formal_tick=tick,
            target_formal_tick=tick + 1,
            expert_action=np.array([tick * 0.01, 0, 0, 0, 0, 0, -1], dtype=np.float64),
            action_mask=np.ones(7, dtype=np.bool_),
            expert_audit=StructuredExpertAuditRecord(
                expert_realization_id=metadata.expert_realization_id,
                source_physics_step=tick * 10,
                source_formal_tick=tick,
                source_time_us=tick * 20_000,
                phase_id="shared_prefix" if tick < 5 else "smooth_approach",
                reference_kind="shared_prefix" if tick < 5 else "selected_reference",
                selected_reference_index=None if tick < 5 else tick - 4,
                target_eef_position_world=np.array(
                    [0.4 + (tick + 1) * 0.001, 0.0, 0.3], dtype=np.float64
                ),
                target_eef_orientation_matrix_world=np.eye(3, dtype=np.float64),
                estimated_object_velocity_world=np.array([0.1, 0.0, 0.0], dtype=np.float64),
            ),
        )
        for tick in range(terminal_tick)
    )
    events = tuple(
        StructuredPhysicalEventRecord(
            kind=kind,
            physics_step_index=time_us // 2_000,
            time_us=time_us,
            payload={"lift_height_m": 0.1},
            terminal_reason="lift_succeeded" if kind == "success" else None,
        )
        for kind, time_us in (
            ("first_contact", first_contact_tick * 20_000),
            ("stable_grasp", stable_grasp_tick * 20_000),
            ("handoff", handoff_tick * 20_000),
            ("lift_threshold", lift_tick * 20_000),
            ("success", terminal_tick * 20_000),
        )
    )
    return tuple(boundaries), transitions, events


def make_formal_source_episode(
    *,
    camera_height: int = 2,
    camera_width: int = 3,
    boundary_count: int = 2,
    schema_version: int = 2,
    accepted_slot: int = 0,
    realization_draw_index: int = 0,
):
    from latency_meta_mdp.data.source.contracts import (
        FormalSourceSynchronizedEpisode,
    )

    metadata = make_formal_source_metadata(
        camera_height=camera_height,
        camera_width=camera_width,
        schema_version=schema_version,
        accepted_slot=accepted_slot,
        realization_draw_index=realization_draw_index,
    )
    boundaries, transitions, events = make_formal_source_records(
        metadata,
        boundary_count=boundary_count,
    )
    return FormalSourceSynchronizedEpisode(
        metadata=metadata,
        boundaries=boundaries,
        transitions=transitions,
        physical_events=events,
        terminal_reason="lift_succeeded",
    )
