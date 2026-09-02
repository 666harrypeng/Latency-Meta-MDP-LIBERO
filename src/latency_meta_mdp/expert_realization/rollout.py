"""Actual RoboSuite/OSC execution of one frozen smooth expert realization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from latency_meta_mdp.expert_realization.executor import (
    SemanticExecutionFailure,
    StructuredExpertDecision,
    StructuredExpertExecutor,
)
from latency_meta_mdp.expert_realization.selector import SelectedReference
from latency_meta_mdp.expert_realization.shared_prefix import (
    _build_shared_prefix_anchor,
    _compare_shared_prefix_anchors,
)
from latency_meta_mdp.expert_realization.task_instance import (
    MaterializedTaskInstance,
    _build_task_instance_runtime,
    _sha256_bytes,
)
from latency_meta_mdp.expert_realization.trajectory_intent import PlannedMotionIntent
from latency_meta_mdp.outcomes import OutcomeStatus


@dataclass(frozen=True)
class StructuredRealizationRollout:
    terminal_status: str
    terminal_reason: str
    terminal_tick: int | None
    physical_handoff_tick: int | None
    phase_sequence: tuple[str, ...]
    decisions: tuple[StructuredExpertDecision, ...]
    actions: np.ndarray
    eef_positions_world: np.ndarray
    object_positions_world: np.ndarray
    agentview_rgb: np.ndarray
    wrist_rgb: np.ndarray

    def __post_init__(self) -> None:
        if self.terminal_status not in {
            "success",
            "failure",
            "semantic_failure",
            "budget_exhausted",
        }:
            raise ValueError("unknown structured rollout terminal status")
        if type(self.terminal_reason) is not str or not self.terminal_reason:
            raise ValueError("terminal_reason must be a non-empty string")
        for name in ("terminal_tick", "physical_handoff_tick"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or None")
        if type(self.phase_sequence) is not tuple or any(
            type(value) is not str or not value for value in self.phase_sequence
        ):
            raise TypeError("phase_sequence must contain strings")
        if type(self.decisions) is not tuple or any(
            not isinstance(value, StructuredExpertDecision) for value in self.decisions
        ):
            raise TypeError("decisions must contain StructuredExpertDecision values")
        contracts = {
            "actions": (np.dtype(np.float64), (None, 7)),
            "eef_positions_world": (np.dtype(np.float64), (None, 3)),
            "object_positions_world": (np.dtype(np.float64), (None, 3)),
            "agentview_rgb": (np.dtype(np.uint8), (None, 256, 256, 3)),
            "wrist_rgb": (np.dtype(np.uint8), (None, 256, 256, 3)),
        }
        for name, (dtype, shape) in contracts.items():
            value = np.asarray(getattr(self, name))
            if (
                value.dtype != dtype
                or value.ndim != len(shape)
                or any(
                    expected is not None and actual != expected
                    for actual, expected in zip(value.shape, shape, strict=True)
                )
            ):
                raise ValueError(f"{name} violates its rollout array contract")
            copied = np.array(value, copy=True)
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)
        boundary_count = len(self.eef_positions_world)
        if (
            len(self.object_positions_world) != boundary_count
            or len(self.agentview_rgb) != boundary_count
            or len(self.wrist_rgb) != boundary_count
            or len(self.actions) != boundary_count - 1
        ):
            raise ValueError("structured rollout arrays are not time aligned")


def _append_boundary(
    snapshot: object,
    *,
    eef_positions: list[np.ndarray],
    object_positions: list[np.ndarray],
    agentview_frames: list[np.ndarray],
    wrist_frames: list[np.ndarray],
) -> None:
    eef_positions.append(np.asarray(snapshot.eef_pos, dtype=np.float64))
    object_positions.append(np.asarray(snapshot.object_body_pos, dtype=np.float64))
    agentview_frames.append(np.asarray(snapshot.cameras["agentview"].rgb, dtype=np.uint8))
    wrist_frames.append(np.asarray(snapshot.cameras["robot0_eye_in_hand"].rgb, dtype=np.uint8))


def _execute_structured_realization(
    *,
    task_instance: MaterializedTaskInstance,
    intent: PlannedMotionIntent,
    reference: SelectedReference,
    maximum_formal_ticks: int,
    source_metadata: object | None = None,
) -> tuple[StructuredRealizationRollout, object | None]:
    """Replay the exact K6 prefix, then execute the frozen realization in real physics."""
    if not isinstance(task_instance, MaterializedTaskInstance):
        raise TypeError("task_instance must be a MaterializedTaskInstance")
    if not isinstance(intent, PlannedMotionIntent):
        raise TypeError("intent must be a PlannedMotionIntent")
    if not isinstance(reference, SelectedReference):
        raise TypeError("reference must be a SelectedReference")
    if intent.task_instance_id != task_instance.task_instance_id or (
        reference.expert_realization_key.task_instance_id != task_instance.task_instance_id
    ):
        raise ValueError("task, intent, and reference identities do not match")
    if type(maximum_formal_ticks) is not int or maximum_formal_ticks <= 5:
        raise ValueError("maximum_formal_ticks must exceed the K6 decision tick")

    runtime = _build_task_instance_runtime(task_instance)
    actions: list[np.ndarray] = []
    decisions: list[StructuredExpertDecision] = []
    eef_positions: list[np.ndarray] = []
    object_positions: list[np.ndarray] = []
    agentview_frames: list[np.ndarray] = []
    wrist_frames: list[np.ndarray] = []
    terminal_status = "budget_exhausted"
    terminal_reason = "maximum_formal_ticks_exhausted"
    snapshot = None
    source_recorder = None
    if source_metadata is not None:
        from latency_meta_mdp.expert_realization.source_corpus.recording import (
            SourceEpisodeRecorder,
        )

        source_recorder = SourceEpisodeRecorder(source_metadata)
    try:
        snapshot = runtime.executor.initialize()
        boundaries = [snapshot]
        if source_recorder is not None:
            source_recorder.append_boundary(
                snapshot,
                runtime=runtime,
                previous_action=None,
                outcome_status=runtime.tracker.status.value,
            )
        _append_boundary(
            snapshot,
            eef_positions=eef_positions,
            object_positions=object_positions,
            agentview_frames=agentview_frames,
            wrist_frames=wrist_frames,
        )
        for action in task_instance.expected_anchor.shared_actions:
            if source_recorder is not None:
                source_recorder.append_shared_transition(snapshot, action)
            actions.append(np.asarray(action, dtype=np.float64))
            snapshot = runtime.executor.step_formal(action)
            if source_recorder is not None:
                source_recorder.append_boundary(
                    snapshot,
                    runtime=runtime,
                    previous_action=action,
                    outcome_status=runtime.tracker.status.value,
                )
            boundaries.append(snapshot)
            _append_boundary(
                snapshot,
                eef_positions=eef_positions,
                object_positions=object_positions,
                agentview_frames=agentview_frames,
                wrist_frames=wrist_frames,
            )
        actual_anchor = _build_shared_prefix_anchor(
            runtime=runtime,
            boundaries=tuple(boundaries),
            shared_actions=task_instance.expected_anchor.shared_actions,
            motion_profile_sha256=task_instance.task_instance_id.motion_profile_sha256,
            shared_endpoint_sha256=_sha256_bytes(task_instance.shared_endpoint_bytes),
        )
        _compare_shared_prefix_anchors(task_instance.expected_anchor, actual_anchor)
        arm = runtime.env.robots[0].part_controllers["right"]
        expert = StructuredExpertExecutor(
            action_contract=runtime.action_contract,
            intent=intent,
            reference=reference,
            decision_source_tick=task_instance.decision_source_tick,
            world_to_base_rotation=np.asarray(arm.origin_ori.T, dtype=np.float64),
        )
        try:
            while (
                runtime.tracker.status is OutcomeStatus.RUNNING
                and snapshot.formal_tick_index < maximum_formal_ticks
            ):
                contact = runtime.handoff.last_contact
                decision = expert.next_action(
                    snapshot=snapshot,
                    handoff_state=runtime.handoff.state,
                    left_pad_contact=False if contact is None else contact.left_pad_contact,
                    right_pad_contact=False if contact is None else contact.right_pad_contact,
                )
                decisions.append(decision)
                if source_recorder is not None:
                    source_recorder.append_decision_transition(decision)
                actions.append(np.asarray(decision.action, dtype=np.float64))
                snapshot = runtime.executor.step_formal(decision.action)
                if source_recorder is not None:
                    source_recorder.append_boundary(
                        snapshot,
                        runtime=runtime,
                        previous_action=decision.action,
                        outcome_status=runtime.tracker.status.value,
                    )
                _append_boundary(
                    snapshot,
                    eef_positions=eef_positions,
                    object_positions=object_positions,
                    agentview_frames=agentview_frames,
                    wrist_frames=wrist_frames,
                )
        except SemanticExecutionFailure as error:
            terminal_status = "semantic_failure"
            terminal_reason = str(error)
        else:
            if runtime.tracker.status is OutcomeStatus.SUCCESS:
                terminal_status = "success"
                terminal_reason = runtime.tracker.terminal_reason.value
            elif runtime.tracker.status is OutcomeStatus.FAILURE:
                terminal_status = "failure"
                terminal_reason = runtime.tracker.terminal_reason.value
        rollout = StructuredRealizationRollout(
            terminal_status=terminal_status,
            terminal_reason=terminal_reason,
            terminal_tick=snapshot.formal_tick_index,
            physical_handoff_tick=(
                None if runtime.tracker.handoff_us is None else runtime.tracker.handoff_us // 20_000
            ),
            phase_sequence=tuple(decision.phase.value for decision in decisions),
            decisions=tuple(decisions),
            actions=np.asarray(actions, dtype=np.float64),
            eef_positions_world=np.asarray(eef_positions, dtype=np.float64),
            object_positions_world=np.asarray(object_positions, dtype=np.float64),
            agentview_rgb=np.asarray(agentview_frames, dtype=np.uint8),
            wrist_rgb=np.asarray(wrist_frames, dtype=np.uint8),
        )
        source_episode = None
        if source_recorder is not None and terminal_status == "success":
            source_episode = source_recorder.build_success(
                terminal_reason=terminal_reason,
                outcome_events=runtime.tracker.events,
                handoff_release_qpos=runtime.handoff.release_qpos,
                handoff_release_qvel=runtime.handoff.release_qvel,
            )
        return rollout, source_episode
    finally:
        runtime.close()


def execute_structured_realization(
    *,
    task_instance: MaterializedTaskInstance,
    intent: PlannedMotionIntent,
    reference: SelectedReference,
    maximum_formal_ticks: int,
) -> StructuredRealizationRollout:
    """Replay one frozen realization for bounded visual review."""
    rollout, _source_episode = _execute_structured_realization(
        task_instance=task_instance,
        intent=intent,
        reference=reference,
        maximum_formal_ticks=maximum_formal_ticks,
    )
    return rollout
