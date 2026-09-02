"""Full typed recording helpers for successful smooth-expert source episodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from latency_meta_mdp.expert_realization.executor import StructuredExpertDecision
from latency_meta_mdp.expert_realization.recording_contracts import (
    StructuredBoundaryRecord,
    StructuredDeploymentRecord,
    StructuredExpertAuditRecord,
    StructuredPhysicalEventRecord,
    StructuredQualificationRecord,
    StructuredTransitionRecord,
)
from latency_meta_mdp.expert_realization.selector import SelectedReference
from latency_meta_mdp.expert_realization.source_corpus.contracts import (
    FormalSourceEpisodeMetadata,
    FormalSourceSynchronizedEpisode,
)
from latency_meta_mdp.expert_realization.task_instance import MaterializedTaskInstance
from latency_meta_mdp.expert_realization.trajectory_intent import PlannedMotionIntent
from latency_meta_mdp.outcomes import OutcomeEvent
from latency_meta_mdp.snapshots import BoundarySnapshot

if TYPE_CHECKING:
    from latency_meta_mdp.expert_realization.config import PilotGateConfig
    from latency_meta_mdp.expert_realization.qualification import QualificationDecision
    from latency_meta_mdp.expert_realization.robot_bridge import PandaPlanningBridge
    from latency_meta_mdp.expert_realization.safety import ActualRolloutSafetyReport


class SourceRecordingFailure(RuntimeError):
    """A formal source execution ended without an admissible success episode."""

    def __init__(self, *, terminal_status: str, terminal_reason: str) -> None:
        self.terminal_status = terminal_status
        self.terminal_reason = terminal_reason
        super().__init__(f"source recording failed: {terminal_status}: {terminal_reason}")


class SourceQualificationFailure(RuntimeError):
    """A task-success trace failed one or more immutable physical admission gates."""

    def __init__(
        self,
        *,
        report: ActualRolloutSafetyReport,
        decision: QualificationDecision,
    ) -> None:
        self.report = report
        self.decision = decision
        super().__init__(f"source recording failed physical qualification: {decision.failures}")


@dataclass(frozen=True)
class QualifiedSourceRecording:
    episode: FormalSourceSynchronizedEpisode
    safety_report: ActualRolloutSafetyReport
    qualification: QualificationDecision

    def __post_init__(self) -> None:
        from latency_meta_mdp.expert_realization.qualification import QualificationDecision
        from latency_meta_mdp.expert_realization.safety import ActualRolloutSafetyReport

        if not isinstance(self.episode, FormalSourceSynchronizedEpisode):
            raise TypeError("episode must be FormalSourceSynchronizedEpisode")
        if not isinstance(self.safety_report, ActualRolloutSafetyReport):
            raise TypeError("safety_report must be ActualRolloutSafetyReport")
        if not isinstance(self.qualification, QualificationDecision):
            raise TypeError("qualification must be QualificationDecision")
        if not self.qualification.eligible:
            raise ValueError("QualifiedSourceRecording requires an eligible qualification")

    def qualification_mapping(self) -> dict[str, Any]:
        return {
            **self.qualification.to_mapping(),
            "actual_rollout_safety": self.safety_report.to_mapping(),
        }


def _commanded(snapshot: BoundarySnapshot, name: str, *, shape: tuple[int, ...]) -> np.ndarray:
    if name not in snapshot.commanded_world:
        raise ValueError(f"boundary commanded_world is missing {name}")
    value = np.asarray(snapshot.commanded_world[name])
    if value.shape != shape:
        raise ValueError(f"boundary commanded_world {name} has invalid shape")
    return value


def _commanded_scalar(snapshot: BoundarySnapshot, name: str) -> Any:
    return _commanded(snapshot, name, shape=()).item()


def _controller_errors(
    *, runtime: Any, snapshot: BoundarySnapshot, previous_action: np.ndarray | None
) -> dict[str, Any]:
    if previous_action is None:
        return {
            "applied_reference": None,
            "applied_reference_source_tick": None,
            "nullspace_joint_position_error": None,
            "eef_position_error": None,
            "eef_orientation_error_rotvec": None,
        }
    from robosuite.utils.transform_utils import mat2quat, quat2axisangle

    arm = runtime.env.robots[0].part_controllers["right"]
    desired_position_world = arm.origin_pos + arm.origin_ori @ arm.goal_pos
    desired_orientation_world = arm.origin_ori @ arm.goal_ori
    orientation_delta = desired_orientation_world @ snapshot.eef_xmat.T
    orientation_error = quat2axisangle(np.array(mat2quat(orientation_delta), copy=True))
    return {
        "applied_reference": np.asarray(previous_action, dtype=np.float64),
        "applied_reference_source_tick": snapshot.formal_tick_index - 1,
        "nullspace_joint_position_error": (
            np.asarray(arm.initial_joint, dtype=np.float64)
            - np.asarray(snapshot.robot_qpos, dtype=np.float64)
        ),
        "eef_position_error": (
            np.asarray(desired_position_world, dtype=np.float64)
            - np.asarray(snapshot.eef_pos, dtype=np.float64)
        ),
        "eef_orientation_error_rotvec": np.asarray(orientation_error, dtype=np.float64),
    }


class SourceEpisodeRecorder:
    """Accumulate one exact successful source episode without owning simulator execution."""

    def __init__(self, metadata: FormalSourceEpisodeMetadata) -> None:
        if not isinstance(metadata, FormalSourceEpisodeMetadata):
            raise TypeError("metadata must be FormalSourceEpisodeMetadata")
        self.metadata = metadata
        self._boundaries: list[StructuredBoundaryRecord] = []
        self._transitions: list[StructuredTransitionRecord] = []

    @property
    def transitions(self) -> tuple[StructuredTransitionRecord, ...]:
        return tuple(self._transitions)

    def append_boundary(
        self,
        snapshot: BoundarySnapshot,
        *,
        runtime: Any,
        previous_action: np.ndarray | None,
        outcome_status: str,
    ) -> None:
        if not isinstance(snapshot, BoundarySnapshot):
            raise TypeError("snapshot must be a BoundarySnapshot")
        expected_tick = len(self._boundaries)
        if snapshot.formal_tick_index != expected_tick:
            raise ValueError(f"next boundary must have formal tick {expected_tick}")
        if expected_tick == 0 and previous_action is not None:
            raise ValueError("boundary zero cannot have a previous action")
        if expected_tick > 0 and previous_action is None:
            raise ValueError("nonzero boundary requires its previous action")
        if len(self._transitions) != expected_tick:
            raise ValueError("boundary append requires one preceding transition")
        controller = _controller_errors(
            runtime=runtime,
            snapshot=snapshot,
            previous_action=previous_action,
        )
        deployment = StructuredDeploymentRecord(
            source_physics_step=snapshot.physics_step_index,
            source_formal_tick=snapshot.formal_tick_index,
            source_time_us=snapshot.time_us,
            agentview_rgb=snapshot.cameras["agentview"].rgb,
            robot0_eye_in_hand_rgb=snapshot.cameras["robot0_eye_in_hand"].rgb,
            robot_qpos=np.asarray(snapshot.robot_qpos, dtype=np.float64),
            robot_qvel=np.asarray(snapshot.robot_qvel, dtype=np.float64),
            gripper_qpos=np.asarray(snapshot.robot_gripper_qpos, dtype=np.float64),
            gripper_qvel=np.asarray(snapshot.robot_gripper_qvel, dtype=np.float64),
            eef_position_world=np.asarray(snapshot.eef_pos, dtype=np.float64),
            eef_orientation_matrix_world=np.asarray(snapshot.eef_xmat, dtype=np.float64),
        )
        qualification = StructuredQualificationRecord(
            object_pose=np.concatenate(
                [snapshot.object_body_pos, snapshot.object_body_quat_wxyz]
            ).astype(np.float64),
            object_velocity=np.asarray(snapshot.object_qvel, dtype=np.float64),
            commanded_motion_position=np.asarray(
                _commanded(snapshot, "target_position", shape=(3,)), dtype=np.float64
            ),
            commanded_motion_velocity=np.asarray(
                _commanded(snapshot, "target_velocity", shape=(3,)), dtype=np.float64
            ),
            commanded_motion_acceleration=np.asarray(
                _commanded(snapshot, "target_acceleration", shape=(3,)), dtype=np.float64
            ),
            commanded_motion_segment_index=int(
                _commanded_scalar(snapshot, "segment_index")
            ),
            left_pad_contact=bool(_commanded_scalar(snapshot, "contact_left")),
            right_pad_contact=bool(_commanded_scalar(snapshot, "contact_right")),
            handoff_state=str(_commanded_scalar(snapshot, "handoff_state")),
            relative_geometry=(
                np.asarray(snapshot.eef_pos, dtype=np.float64)
                - np.asarray(snapshot.object_body_pos, dtype=np.float64)
            ),
            actuator_ctrl=np.asarray(snapshot.actuator_ctrl, dtype=np.float64),
            **controller,
        )
        self._boundaries.append(
            StructuredBoundaryRecord(
                formal_tick_index=snapshot.formal_tick_index,
                physics_step_index=snapshot.physics_step_index,
                time_us=snapshot.time_us,
                deployment=deployment,
                qualification=qualification,
                outcome_status=outcome_status,
            )
        )

    def append_shared_transition(
        self, snapshot: BoundarySnapshot, action: np.ndarray
    ) -> None:
        if not isinstance(snapshot, BoundarySnapshot):
            raise TypeError("snapshot must be a BoundarySnapshot")
        if snapshot.formal_tick_index != len(self._transitions) or len(self._boundaries) != (
            len(self._transitions) + 1
        ):
            raise ValueError("shared transition source is not the current boundary")
        audit = StructuredExpertAuditRecord(
            expert_realization_id=self.metadata.expert_realization_id,
            source_physics_step=snapshot.physics_step_index,
            source_formal_tick=snapshot.formal_tick_index,
            source_time_us=snapshot.time_us,
            phase_id="shared_prefix",
            reference_kind="shared_prefix",
            selected_reference_index=None,
            target_eef_position_world=np.asarray(snapshot.eef_pos, dtype=np.float64),
            target_eef_orientation_matrix_world=np.asarray(snapshot.eef_xmat, dtype=np.float64),
            estimated_object_velocity_world=np.asarray(snapshot.object_qvel[:3], dtype=np.float64),
        )
        self._transitions.append(
            StructuredTransitionRecord(
                source_formal_tick=snapshot.formal_tick_index,
                target_formal_tick=snapshot.formal_tick_index + 1,
                expert_action=np.asarray(action, dtype=np.float64),
                action_mask=np.ones(7, dtype=np.bool_),
                expert_audit=audit,
            )
        )

    def append_decision_transition(self, decision: StructuredExpertDecision) -> None:
        if not isinstance(decision, StructuredExpertDecision):
            raise TypeError("decision must be StructuredExpertDecision")
        if decision.source_formal_tick != len(self._transitions) or len(self._boundaries) != (
            len(self._transitions) + 1
        ):
            raise ValueError("decision transition source is not the current boundary")
        audit = StructuredExpertAuditRecord(
            expert_realization_id=self.metadata.expert_realization_id,
            source_physics_step=decision.source_physics_step,
            source_formal_tick=decision.source_formal_tick,
            source_time_us=decision.source_time_us,
            phase_id=decision.phase.value,
            reference_kind=decision.reference_kind,
            selected_reference_index=decision.selected_reference_index,
            target_eef_position_world=decision.target_eef_position_world,
            target_eef_orientation_matrix_world=decision.target_eef_orientation_world,
            estimated_object_velocity_world=decision.estimated_object_velocity_world,
        )
        self._transitions.append(
            StructuredTransitionRecord(
                source_formal_tick=decision.source_formal_tick,
                target_formal_tick=decision.source_formal_tick + 1,
                expert_action=decision.action,
                action_mask=np.ones(7, dtype=np.bool_),
                expert_audit=audit,
            )
        )

    def build_success(
        self,
        *,
        terminal_reason: str,
        outcome_events: tuple[OutcomeEvent, ...],
        handoff_release_qpos: np.ndarray | None,
        handoff_release_qvel: np.ndarray | None,
    ) -> FormalSourceSynchronizedEpisode:
        if (
            not self._boundaries
            or len(self._boundaries) != len(self._transitions) + 1
            or self._boundaries[-1].outcome_status != "success"
        ):
            raise ValueError("source publication requires a complete successful trace")
        records = []
        for event in outcome_events:
            reason = event.terminal_reason
            payload: dict[str, Any] = {}
            if event.kind == "handoff":
                if handoff_release_qpos is None or handoff_release_qvel is None:
                    raise ValueError("handoff event requires released object state")
                payload = {
                    "release_object_qpos": np.asarray(handoff_release_qpos).tolist(),
                    "release_object_qvel": np.asarray(handoff_release_qvel).tolist(),
                }
            records.append(
                StructuredPhysicalEventRecord(
                    kind=event.kind,
                    physics_step_index=event.time_us // 2_000,
                    time_us=event.time_us,
                    payload=payload,
                    terminal_reason=None if reason is None else reason.value,
                )
            )
        return FormalSourceSynchronizedEpisode(
            metadata=self.metadata,
            boundaries=tuple(self._boundaries),
            transitions=tuple(self._transitions),
            physical_events=tuple(records),
            terminal_reason=terminal_reason,
        )


def execute_structured_source_recording(
    *,
    task_instance: MaterializedTaskInstance,
    intent: PlannedMotionIntent,
    reference: SelectedReference,
    metadata: FormalSourceEpisodeMetadata,
    maximum_formal_ticks: int,
    planning_bridge: PandaPlanningBridge,
    gate: PilotGateConfig,
) -> QualifiedSourceRecording:
    """Execute, physically qualify, and return one complete successful source recording."""
    from latency_meta_mdp.expert_realization.config import PilotGateConfig
    from latency_meta_mdp.expert_realization.qualification import qualify_actual_rollout
    from latency_meta_mdp.expert_realization.robot_bridge import PandaPlanningBridge
    from latency_meta_mdp.expert_realization.rollout import _execute_structured_realization
    from latency_meta_mdp.expert_realization.safety import ActualRolloutSafetyReport

    if not isinstance(planning_bridge, PandaPlanningBridge):
        raise TypeError("planning_bridge must be a PandaPlanningBridge")
    if not isinstance(gate, PilotGateConfig):
        raise TypeError("gate must be a PilotGateConfig")
    rollout, source_episode, safety_report = _execute_structured_realization(
        task_instance=task_instance,
        intent=intent,
        reference=reference,
        maximum_formal_ticks=maximum_formal_ticks,
        source_metadata=metadata,
        safety_bridge=planning_bridge,
    )
    if source_episode is None:
        raise SourceRecordingFailure(
            terminal_status=rollout.terminal_status,
            terminal_reason=rollout.terminal_reason,
        )
    if not isinstance(source_episode, FormalSourceSynchronizedEpisode):
        raise TypeError("source execution returned an invalid episode")
    if not isinstance(safety_report, ActualRolloutSafetyReport):
        raise TypeError("source execution returned no actual-physics safety report")
    decision = qualify_actual_rollout(safety_report, gate=gate)
    if not decision.eligible:
        raise SourceQualificationFailure(report=safety_report, decision=decision)
    return QualifiedSourceRecording(
        episode=source_episode,
        safety_report=safety_report,
        qualification=decision,
    )
