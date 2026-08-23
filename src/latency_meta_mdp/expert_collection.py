"""Synchronized zero-latency expert episode collection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from latency_meta_mdp.artifacts import sha256_file
from latency_meta_mdp.backend import FormalStepExecutor, PreparedPhysicsPoint, RoboSuitePlant
from latency_meta_mdp.config import RuntimeConfig, load_runtime_config
from latency_meta_mdp.control import ActionContract, load_action_contract
from latency_meta_mdp.expert import (
    ExpertConfig,
    ExpertDecision,
    ExpertObservation,
    ScriptedBallExpert,
    is_qualified_expert_episode,
    load_expert_config,
)
from latency_meta_mdp.handoff import (
    HandoffAwareBallWorld,
    OneWayHandoff,
    PadContactSample,
    PandaBallContactDetector,
)
from latency_meta_mdp.motion import DrivenBallWorld, build_motion_profile, load_motion_config
from latency_meta_mdp.outcomes import EpisodeOutcomeTracker, OutcomeCriteria, OutcomeStatus
from latency_meta_mdp.recording import (
    BoundaryRecord,
    CameraRecord,
    CommandedMotionRecord,
    ControlDebugRecord,
    DeploymentRecord,
    EpisodeMetadata,
    ExpertAuditRecord,
    PadContactRecord,
    PhysicalEventKind,
    PhysicalEventRecord,
    PrivilegedRecord,
    RecordProfile,
    SynchronizedEpisode,
    TransitionRecord,
)
from latency_meta_mdp.snapshots import BoundarySnapshot, BoundarySnapshotter
from latency_meta_mdp.task import load_task_spec, make_dynamic_grasp_lift_environment
from latency_meta_mdp.timing import ClockLedger

_SAFE_EPISODE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class ExpertEpisodeSpec:
    episode_id: str
    level: int
    scene_seed: int
    motion_seed: int
    expert_seed: int
    record_profile: RecordProfile
    camera_width: int
    camera_height: int

    def __post_init__(self) -> None:
        if _SAFE_EPISODE_ID.fullmatch(self.episode_id) is None:
            raise ValueError("episode_id must be one safe path component")
        if self.level not in range(4):
            raise ValueError("level must be one of 0, 1, 2, 3")
        for name in ("scene_seed", "motion_seed", "expert_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.record_profile, RecordProfile):
            raise TypeError("record_profile must be a RecordProfile")
        for name in ("camera_width", "camera_height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


class _PhysicalEventCapture:
    def __init__(self, *, env: Any, handoff: OneWayHandoff) -> None:
        self._env = env
        self._handoff = handoff
        self._event_count = 0
        self.records: list[PhysicalEventRecord] = []

    def on_physics_point(self, point: PreparedPhysicsPoint) -> dict[str, np.ndarray]:
        updates = self._handoff.on_physics_point(point)
        events = self._handoff.outcome_tracker.events
        for event in events[self._event_count :]:
            contact = self._handoff.last_contact
            if contact is None or contact.time_us != event.time_us:
                raise RuntimeError("outcome event is missing its same-time contact sample")
            payload: dict[str, Any] = {
                "left_pad_contact": contact.left_pad_contact,
                "right_pad_contact": contact.right_pad_contact,
                "left_contact_count": contact.left_contact_count,
                "right_contact_count": contact.right_contact_count,
                "lift_height_m": self._env.goal_measurements().lift_height_m,
                "handoff_state": self._handoff.state.value,
            }
            if event.kind == PhysicalEventKind.HANDOFF.value:
                if self._handoff.release_qpos is None or self._handoff.release_qvel is None:
                    raise RuntimeError("handoff event is missing the released object state")
                payload["release_object_qpos"] = self._handoff.release_qpos
                payload["release_object_qvel"] = self._handoff.release_qvel
            self.records.append(
                PhysicalEventRecord(
                    kind=PhysicalEventKind(event.kind),
                    physics_step_index=event.time_us // 2_000,
                    time_us=event.time_us,
                    payload=payload,
                    terminal_reason=event.terminal_reason,
                )
            )
        self._event_count = len(events)
        return updates


def _controller_debug(
    *,
    env: Any,
    snapshot: BoundarySnapshot,
    previous_action: np.ndarray | None,
) -> ControlDebugRecord:
    from robosuite.utils.transform_utils import mat2quat, quat2axisangle

    if previous_action is None:
        return ControlDebugRecord(
            actuator_ctrl=snapshot.actuator_ctrl,
            applied_reference=None,
            applied_reference_source_formal_tick=None,
            nullspace_joint_position_error=None,
            eef_position_error=None,
            eef_orientation_error_rotvec=None,
        )
    arm = env.robots[0].part_controllers["right"]
    desired_position_world = arm.origin_pos + arm.origin_ori @ arm.goal_pos
    desired_orientation_world = arm.origin_ori @ arm.goal_ori
    orientation_delta = desired_orientation_world @ snapshot.eef_xmat.T
    orientation_error = quat2axisangle(np.array(mat2quat(orientation_delta), copy=True))
    source_tick = None if previous_action is None else snapshot.formal_tick_index - 1
    return ControlDebugRecord(
        actuator_ctrl=snapshot.actuator_ctrl,
        applied_reference=previous_action,
        applied_reference_source_formal_tick=source_tick,
        nullspace_joint_position_error=np.asarray(arm.initial_joint) - snapshot.robot_qpos,
        eef_position_error=desired_position_world - snapshot.eef_pos,
        eef_orientation_error_rotvec=orientation_error,
    )


def _same_time_contact(
    *,
    snapshot: BoundarySnapshot,
    contact: PadContactSample | None,
) -> PadContactSample:
    if contact is None or (
        contact.physics_step_index != snapshot.physics_step_index
        or contact.formal_tick_index != snapshot.formal_tick_index
        or contact.time_us != snapshot.time_us
    ):
        raise RuntimeError("boundary is missing its same-time contact sample")
    return contact


def _boundary_record(
    *,
    env: Any,
    snapshot: BoundarySnapshot,
    handoff: OneWayHandoff,
    record_profile: RecordProfile,
    previous_action: np.ndarray | None,
) -> BoundaryRecord:
    cameras = {
        name: CameraRecord(
            name=name,
            source_physics_step=sample.source_physics_step,
            source_formal_tick=sample.source_formal_tick,
            source_time_us=sample.source_time_us,
            rgb=sample.rgb,
        )
        for name, sample in snapshot.cameras.items()
    }
    deployment = DeploymentRecord(
        images=cameras,
        robot_qpos=snapshot.robot_qpos,
        robot_qvel=snapshot.robot_qvel,
        gripper_qpos=snapshot.robot_gripper_qpos,
        gripper_qvel=snapshot.robot_gripper_qvel,
        eef_position_world=snapshot.eef_pos,
        eef_orientation_matrix_world=snapshot.eef_xmat,
    )
    privileged = None
    if record_profile.includes_privileged:
        contact = _same_time_contact(snapshot=snapshot, contact=handoff.last_contact)
        world = snapshot.commanded_world
        privileged = PrivilegedRecord(
            object_pose=snapshot.object_qpos,
            object_velocity=snapshot.object_qvel,
            commanded_motion=CommandedMotionRecord(
                position=world["target_position"],
                velocity=world["target_velocity"],
                acceleration=world["target_acceleration"],
                segment_index=int(np.asarray(world["segment_index"]).item()),
            ),
            contact=PadContactRecord(
                left=contact.left_pad_contact,
                right=contact.right_pad_contact,
            ),
            handoff_state=handoff.state,
            relative_geometry=snapshot.object_body_pos - snapshot.eef_pos,
        )
    control_debug = None
    if record_profile.includes_control_debug:
        control_debug = _controller_debug(
            env=env,
            snapshot=snapshot,
            previous_action=previous_action,
        )
    return BoundaryRecord(
        formal_tick_index=snapshot.formal_tick_index,
        physics_step_index=snapshot.physics_step_index,
        time_us=snapshot.time_us,
        deployment=deployment,
        privileged=privileged,
        control_debug=control_debug,
        outcome_status=handoff.outcome_tracker.status,
    )


def _transition_record(decision: ExpertDecision) -> TransitionRecord:
    return TransitionRecord(
        source_formal_tick=decision.source_formal_tick,
        target_formal_tick=decision.source_formal_tick + 1,
        expert_action=decision.action,
        action_mask=np.ones(decision.action.shape, dtype=bool),
        expert_audit=ExpertAuditRecord(
            expert_id="panda_ball_feedback_v1",
            source_physics_step=decision.source_physics_step,
            source_formal_tick=decision.source_formal_tick,
            source_time_us=decision.source_time_us,
            phase=decision.phase,
            history_start_time_us=decision.history_start_time_us,
            history_sample_count=decision.history_sample_count,
            target_eef_position_world=decision.target_eef_position,
            estimated_object_velocity_world=decision.estimated_object_velocity,
        ),
    )


@dataclass(frozen=True)
class ExpertEpisodeRuntime:
    runtime_config: RuntimeConfig
    contract: ActionContract
    expert_config: ExpertConfig
    env: Any
    tracker: EpisodeOutcomeTracker
    handoff: OneWayHandoff
    event_capture: _PhysicalEventCapture
    executor: FormalStepExecutor
    expert: ScriptedBallExpert
    metadata: EpisodeMetadata

    def close(self) -> None:
        self.env.close()


def build_expert_episode_runtime(
    *,
    project_root: Path,
    spec: ExpertEpisodeSpec,
) -> ExpertEpisodeRuntime:
    """Construct one fresh episode-local runtime without advancing it."""
    root = project_root.resolve()
    paths = {
        "runtime": root / "configs/runtime/robosuite_v1.yaml",
        "task": root / "configs/task/dynamic_grasp_lift_l0.yaml",
        "motion": root / f"configs/motion/dynamic_grasp_lift_l{spec.level}.yaml",
        "control": root / "configs/control/panda_osc_pose_delta_v1.yaml",
        "expert": root / "configs/expert/panda_ball_feedback_v1.yaml",
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("expert collection configuration is incomplete")
    runtime_config = load_runtime_config(paths["runtime"])
    contract: ActionContract = load_action_contract(paths["control"])
    expert_config = load_expert_config(paths["expert"])
    motion_config = load_motion_config(paths["motion"])
    task_spec = load_task_spec(paths["task"])
    if (
        contract.physics_dt_us != runtime_config.physics_dt_us
        or contract.formal_tick_us != runtime_config.formal_tick_us
        or runtime_config.camera_stride_ticks != 1
    ):
        raise ValueError("collection configuration does not share one formal clock")

    env = make_dynamic_grasp_lift_environment(
        spec=task_spec,
        seed=spec.scene_seed,
        offscreen=True,
        controller_config=contract.to_robosuite_config(),
    )
    criteria = OutcomeCriteria(
        physics_dt_us=runtime_config.physics_dt_us,
        formal_tick_us=runtime_config.formal_tick_us,
        stable_grasp_dwell_us=40_000,
        lift_height_m=task_spec.lift_success_height_m,
        lift_dwell_us=100_000,
        grasp_deadline_us=None if spec.level == 0 else motion_config.anchor_time_us,
        lift_timeout_us=10_000_000,
    )
    tracker = EpisodeOutcomeTracker(criteria)
    handoff = OneWayHandoff(
        env=env,
        action_contract=contract,
        contact_detector=PandaBallContactDetector(env),
        outcome_tracker=tracker,
    )
    profile = build_motion_profile(
        config=motion_config,
        seed=spec.motion_seed,
        workspace_z=task_spec.ball_initial_position[2],
    )
    event_capture = _PhysicalEventCapture(env=env, handoff=handoff)
    executor = FormalStepExecutor(
        plant=RoboSuitePlant(
            env=env,
            snapshotter=BoundarySnapshotter(
                camera_names=tuple(task_spec.policy_camera_names),
                width=spec.camera_width,
                height=spec.camera_height,
            ),
            world_writer=HandoffAwareBallWorld(
                driver=DrivenBallWorld(profile=profile, motion_level=spec.level),
                handoff=handoff,
            ),
            physics_point_observer=event_capture.on_physics_point,
            control_observer=handoff.on_control_applied,
        ),
        ledger=ClockLedger(
            physics_dt_us=runtime_config.physics_dt_us,
            formal_tick_us=runtime_config.formal_tick_us,
        ),
    )
    expert = ScriptedBallExpert(action_contract=contract, config=expert_config)
    metadata = EpisodeMetadata(
        schema_version=1,
        episode_id=spec.episode_id,
        task_id=task_spec.task_id,
        instruction=(
            task_spec.instruction
            if spec.level == 0
            else "Grasp the moving ball and lift it."
        ),
        level=spec.level,
        scene_seed=spec.scene_seed,
        motion_seed=spec.motion_seed,
        expert_seed=spec.expert_seed,
        physics_dt_us=runtime_config.physics_dt_us,
        formal_tick_us=runtime_config.formal_tick_us,
        action_contract_id=contract.contract_id,
        action_dim=contract.action_dim,
        actuator_dim=contract.actuator_dim,
        expert_id=expert_config.expert_id,
        record_profile=spec.record_profile,
        config_sha256={name: sha256_file(path) for name, path in paths.items()},
        motion_profile=profile.to_mapping(),
    )
    return ExpertEpisodeRuntime(
        runtime_config=runtime_config,
        contract=contract,
        expert_config=expert_config,
        env=env,
        tracker=tracker,
        handoff=handoff,
        event_capture=event_capture,
        executor=executor,
        expert=expert,
        metadata=metadata,
    )


def run_expert_episode_attempt(
    *,
    project_root: Path,
    spec: ExpertEpisodeSpec,
) -> SynchronizedEpisode:
    """Run one terminal synchronized attempt without hiding ordinary task failure."""

    episode_runtime = build_expert_episode_runtime(project_root=project_root, spec=spec)
    runtime = episode_runtime.runtime_config
    env = episode_runtime.env
    tracker = episode_runtime.tracker
    handoff = episode_runtime.handoff
    event_capture = episode_runtime.event_capture
    executor = episode_runtime.executor
    expert = episode_runtime.expert
    expert_config = episode_runtime.expert_config
    metadata = episode_runtime.metadata
    boundaries: list[BoundaryRecord] = []
    transitions: list[TransitionRecord] = []
    try:
        snapshot = executor.initialize()
        boundaries.append(
            _boundary_record(
                env=env,
                snapshot=snapshot,
                handoff=handoff,
                record_profile=spec.record_profile,
                previous_action=None,
            )
        )
        max_steps = expert_config.collection_max_duration_us // runtime.formal_tick_us
        for _ in range(max_steps):
            arm = env.robots[0].part_controllers["right"]
            decision = expert.next_action(
                observation=ExpertObservation.from_snapshot(
                    snapshot,
                    world_to_base_rotation=arm.origin_ori.T,
                ),
                handoff_state=handoff.state,
            )
            transitions.append(_transition_record(decision))
            snapshot = executor.step_formal(decision.action)
            boundaries.append(
                _boundary_record(
                    env=env,
                    snapshot=snapshot,
                    handoff=handoff,
                    record_profile=spec.record_profile,
                    previous_action=decision.action,
                )
            )
            if tracker.status is not OutcomeStatus.RUNNING:
                break
        if tracker.status is OutcomeStatus.RUNNING:
            raise RuntimeError(
                "expert attempt exhausted its collection budget without terminal state"
            )
        if tracker.terminal_reason is None or tracker.terminal_time_us is None:
            raise RuntimeError("terminal expert attempt is missing its reason")
        episode = SynchronizedEpisode(
            metadata=metadata,
            boundaries=tuple(boundaries),
            transitions=tuple(transitions),
            physical_events=tuple(event_capture.records),
            terminal_status=tracker.status,
            terminal_reason=tracker.terminal_reason,
        )
        episode.validate_complete()
        return episode
    finally:
        episode_runtime.close()


def collect_expert_episode(
    *,
    project_root: Path,
    spec: ExpertEpisodeSpec,
) -> SynchronizedEpisode:
    """Run one attempt and admit only a qualified expert success."""

    episode = run_expert_episode_attempt(project_root=project_root, spec=spec)
    expert_config = load_expert_config(
        project_root.resolve() / "configs/expert/panda_ball_feedback_v1.yaml"
    )
    terminal_time_us = episode.boundaries[-1].time_us
    if not is_qualified_expert_episode(
        status=episode.terminal_status,
        terminal_time_us=terminal_time_us,
        max_duration_us=expert_config.collection_max_duration_us,
    ):
        raise RuntimeError("expert attempt did not satisfy collection qualification")
    return episode
