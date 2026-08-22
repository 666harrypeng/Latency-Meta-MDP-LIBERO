"""Typed, synchronized contracts for expert episode capture."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

import numpy as np

from latency_meta_mdp.outcomes import OutcomeStatus, TerminalReason


class RecordProfile(str, Enum):
    SFT = "sft"
    BELIEF = "belief"
    PILOT_DEBUG = "pilot_debug"

    @property
    def includes_privileged(self) -> bool:
        return self in (RecordProfile.BELIEF, RecordProfile.PILOT_DEBUG)

    @property
    def includes_control_debug(self) -> bool:
        return self is RecordProfile.PILOT_DEBUG


class HandoffState(str, Enum):
    DRIVEN = "driven"
    CONTACT_PENDING = "contact_pending"
    PHYSICAL = "physical"
    FAILURE = "failure"


class PhysicalEventKind(str, Enum):
    FIRST_CONTACT = "first_contact"
    STABLE_GRASP = "stable_grasp"
    HANDOFF = "handoff"
    LIFT_THRESHOLD = "lift_threshold"
    SUCCESS = "success"
    FAILURE = "failure"


_CONFIG_NAMES = frozenset({"runtime", "task", "motion", "control"})
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CAMERA_NAMES = frozenset({"agentview", "robot0_eye_in_hand"})


def _readonly_array(value: Any, *, name: str) -> np.ndarray:
    array = np.array(value, copy=True)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    array.setflags(write=False)
    return array


def _readonly_vector(value: Any, *, name: str, length: int) -> np.ndarray:
    vector = _readonly_array(value, name=name)
    if vector.shape != (length,) or not np.issubdtype(vector.dtype, np.number):
        raise ValueError(f"{name} must be a numeric vector of length {length}")
    return vector


def _freeze(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _readonly_array(value, name="record payload")
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class CameraRecord:
    name: str
    source_physics_step: int
    source_formal_tick: int
    source_time_us: int
    rgb: np.ndarray

    def __post_init__(self) -> None:
        if self.name not in _CAMERA_NAMES:
            raise ValueError("camera name is not part of the policy view")
        if (
            self.source_physics_step < 0
            or self.source_formal_tick < 0
            or self.source_physics_step != self.source_formal_tick * 10
            or self.source_time_us != self.source_physics_step * 2_000
        ):
            raise ValueError("camera source indices do not match the 2 ms / 20 ms clock")
        rgb = np.array(self.rgb, copy=True)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("camera must contain RGB uint8 HxWx3 data")
        rgb.setflags(write=False)
        object.__setattr__(self, "rgb", rgb)


@dataclass(frozen=True)
class DeploymentRecord:
    images: Mapping[str, CameraRecord]
    robot_qpos: np.ndarray
    robot_qvel: np.ndarray
    gripper_state: np.ndarray

    def __post_init__(self) -> None:
        if set(self.images) != _CAMERA_NAMES:
            raise ValueError("deployment images must contain the two policy cameras")
        cameras: dict[str, CameraRecord] = {}
        for name, camera in self.images.items():
            if not isinstance(camera, CameraRecord) or camera.name != name:
                raise ValueError("deployment image keys must match typed camera records")
            cameras[name] = camera
        object.__setattr__(self, "images", MappingProxyType(cameras))
        object.__setattr__(
            self,
            "robot_qpos",
            _readonly_vector(self.robot_qpos, name="robot_qpos", length=7),
        )
        object.__setattr__(
            self,
            "robot_qvel",
            _readonly_vector(self.robot_qvel, name="robot_qvel", length=7),
        )
        object.__setattr__(
            self,
            "gripper_state",
            _readonly_vector(self.gripper_state, name="gripper_state", length=2),
        )


@dataclass(frozen=True)
class CommandedMotionRecord:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    segment_index: int

    def __post_init__(self) -> None:
        for name in ("position", "velocity", "acceleration"):
            object.__setattr__(
                self,
                name,
                _readonly_vector(getattr(self, name), name=f"commanded_motion.{name}", length=3),
            )
        if (
            isinstance(self.segment_index, bool)
            or not isinstance(self.segment_index, int)
            or self.segment_index < 0
        ):
            raise ValueError("segment_index must be a non-negative integer")


@dataclass(frozen=True)
class PadContactRecord:
    left: bool
    right: bool

    def __post_init__(self) -> None:
        if type(self.left) is not bool or type(self.right) is not bool:
            raise TypeError("pad contact values must be booleans")


@dataclass(frozen=True)
class PrivilegedRecord:
    object_pose: np.ndarray
    object_velocity: np.ndarray
    commanded_motion: CommandedMotionRecord
    contact: PadContactRecord
    handoff_state: HandoffState
    relative_geometry: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "object_pose",
            _readonly_vector(self.object_pose, name="object_pose", length=7),
        )
        object.__setattr__(
            self,
            "object_velocity",
            _readonly_vector(self.object_velocity, name="object_velocity", length=6),
        )
        object.__setattr__(
            self,
            "relative_geometry",
            _readonly_vector(self.relative_geometry, name="relative_geometry", length=3),
        )
        if not isinstance(self.commanded_motion, CommandedMotionRecord):
            raise TypeError("commanded_motion must be a CommandedMotionRecord")
        if not isinstance(self.contact, PadContactRecord):
            raise TypeError("contact must be a PadContactRecord")
        if not isinstance(self.handoff_state, HandoffState):
            raise TypeError("handoff_state must be a HandoffState")


@dataclass(frozen=True)
class ControlDebugRecord:
    actuator_ctrl: np.ndarray
    applied_reference: np.ndarray
    joint_position_error: np.ndarray
    eef_position_error: np.ndarray
    eef_orientation_error_rotvec: np.ndarray

    def __post_init__(self) -> None:
        actuator_ctrl = _readonly_array(self.actuator_ctrl, name="actuator_ctrl")
        if actuator_ctrl.ndim != 1:
            raise ValueError("actuator_ctrl must be a vector")
        applied_reference = _readonly_array(
            self.applied_reference,
            name="applied_reference",
        )
        if applied_reference.ndim != 1:
            raise ValueError("applied_reference must be a vector")
        object.__setattr__(
            self,
            "actuator_ctrl",
            actuator_ctrl,
        )
        object.__setattr__(
            self,
            "applied_reference",
            applied_reference,
        )
        object.__setattr__(
            self,
            "joint_position_error",
            _readonly_vector(
                self.joint_position_error,
                name="joint_position_error",
                length=7,
            ),
        )
        object.__setattr__(
            self,
            "eef_position_error",
            _readonly_vector(
                self.eef_position_error,
                name="eef_position_error",
                length=3,
            ),
        )
        object.__setattr__(
            self,
            "eef_orientation_error_rotvec",
            _readonly_vector(
                self.eef_orientation_error_rotvec,
                name="eef_orientation_error_rotvec",
                length=3,
            ),
        )


@dataclass(frozen=True)
class EpisodeMetadata:
    schema_version: int
    episode_id: str
    task_id: str
    instruction: str
    level: int
    scene_seed: int
    motion_seed: int
    expert_seed: int
    physics_dt_us: int
    formal_tick_us: int
    action_contract_id: str
    action_dim: int
    actuator_dim: int
    record_profile: RecordProfile
    config_sha256: Mapping[str, str]
    motion_profile: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("episode schema_version must be 1")
        if not self.episode_id or self.task_id != "dynamic_grasp_lift":
            raise ValueError("episode task identity is invalid")
        if not self.instruction.strip():
            raise ValueError("instruction must be non-empty")
        if self.level not in range(4):
            raise ValueError("level must be one of 0, 1, 2, 3")
        for name in ("scene_seed", "motion_seed", "expert_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.physics_dt_us != 2_000 or self.formal_tick_us != 20_000:
            raise ValueError("episode metadata must use the certified 2 ms / 20 ms clock")
        if not self.action_contract_id:
            raise ValueError("action_contract_id must be non-empty")
        for name in ("action_dim", "actuator_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            self.action_contract_id != "panda_osc_pose_delta_v1"
            or self.action_dim != 7
            or self.actuator_dim != 9
        ):
            raise ValueError("episode metadata does not match the selected Panda action contract")
        if not isinstance(self.record_profile, RecordProfile):
            raise TypeError("record_profile must be a RecordProfile")
        if set(self.config_sha256) != _CONFIG_NAMES:
            raise ValueError("config_sha256 must contain runtime, task, motion, and control")
        if any(_SHA256.fullmatch(value) is None for value in self.config_sha256.values()):
            raise ValueError("config_sha256 values must be lowercase SHA-256 strings")
        if not self.motion_profile:
            raise ValueError("motion_profile must be non-empty")
        object.__setattr__(self, "config_sha256", _freeze(self.config_sha256))
        object.__setattr__(self, "motion_profile", _freeze(self.motion_profile))

    @property
    def physics_steps_per_tick(self) -> int:
        return self.formal_tick_us // self.physics_dt_us


@dataclass(frozen=True)
class BoundaryRecord:
    formal_tick_index: int
    physics_step_index: int
    time_us: int
    deployment: DeploymentRecord
    privileged: PrivilegedRecord | None
    control_debug: ControlDebugRecord | None
    outcome_status: OutcomeStatus

    def validate(self, metadata: EpisodeMetadata) -> None:
        expected_time = self.formal_tick_index * metadata.formal_tick_us
        expected_step = self.formal_tick_index * metadata.physics_steps_per_tick
        if (
            self.formal_tick_index < 0
            or self.time_us != expected_time
            or self.physics_step_index != expected_step
        ):
            raise ValueError("boundary record does not match the formal clock")
        if not isinstance(self.deployment, DeploymentRecord):
            raise TypeError("deployment must be a DeploymentRecord")
        for camera in self.deployment.images.values():
            if (
                camera.source_physics_step != self.physics_step_index
                or camera.source_formal_tick != self.formal_tick_index
                or camera.source_time_us != self.time_us
            ):
                raise ValueError("camera source must match its formal boundary")
        if metadata.record_profile.includes_privileged:
            if not isinstance(self.privileged, PrivilegedRecord):
                raise ValueError("privileged record is required by this profile")
        elif self.privileged is not None:
            raise ValueError("privileged record is disabled by this profile")
        if metadata.record_profile.includes_control_debug:
            if not isinstance(self.control_debug, ControlDebugRecord):
                raise ValueError("control_debug record is required by this profile")
            if self.control_debug.actuator_ctrl.shape != (metadata.actuator_dim,):
                raise ValueError("actuator_ctrl does not match metadata actuator_dim")
            if self.control_debug.applied_reference.shape != (metadata.action_dim,):
                raise ValueError("applied_reference does not match metadata action_dim")
        elif self.control_debug is not None:
            raise ValueError("control_debug record is disabled by this profile")
        if not isinstance(self.outcome_status, OutcomeStatus):
            raise TypeError("outcome_status must be an OutcomeStatus")


@dataclass(frozen=True)
class TransitionRecord:
    source_formal_tick: int
    target_formal_tick: int
    expert_action: np.ndarray
    action_mask: np.ndarray

    def __post_init__(self) -> None:
        if self.source_formal_tick < 0 or self.target_formal_tick != self.source_formal_tick + 1:
            raise ValueError("transition must connect adjacent formal ticks")
        action = _readonly_array(self.expert_action, name="expert_action")
        if action.ndim != 1:
            raise ValueError("expert_action must be a vector")
        mask = np.array(self.action_mask, copy=True)
        if mask.shape != action.shape or mask.dtype != np.bool_:
            raise ValueError("expert_action and boolean action_mask must be aligned vectors")
        mask.setflags(write=False)
        object.__setattr__(self, "expert_action", action)
        object.__setattr__(self, "action_mask", mask)

    def validate(self, metadata: EpisodeMetadata) -> None:
        if self.expert_action.shape != (metadata.action_dim,):
            raise ValueError("expert_action does not match metadata action_dim")


@dataclass(frozen=True)
class PhysicalEventRecord:
    kind: PhysicalEventKind
    physics_step_index: int
    time_us: int
    payload: Mapping[str, Any]
    terminal_reason: TerminalReason | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, PhysicalEventKind):
            raise TypeError("physical event kind must be a PhysicalEventKind")
        if self.kind in {PhysicalEventKind.SUCCESS, PhysicalEventKind.FAILURE}:
            if not isinstance(self.terminal_reason, TerminalReason):
                raise TypeError("terminal physical events require a TerminalReason")
            if (
                self.kind is PhysicalEventKind.SUCCESS
                and self.terminal_reason is not TerminalReason.LIFT_SUCCEEDED
            ):
                raise ValueError("success event must use the lift success reason")
            if (
                self.kind is PhysicalEventKind.FAILURE
                and self.terminal_reason is TerminalReason.LIFT_SUCCEEDED
            ):
                raise ValueError("failure event cannot use the lift success reason")
        elif self.terminal_reason is not None:
            raise ValueError("non-terminal physical events cannot carry a terminal reason")
        object.__setattr__(self, "payload", _freeze(self.payload))

    def validate(self, metadata: EpisodeMetadata, *, final_time_us: int) -> None:
        if (
            self.physics_step_index < 0
            or self.time_us != self.physics_step_index * metadata.physics_dt_us
            or self.time_us > final_time_us
        ):
            raise ValueError("physical event does not match the physics clock or episode bounds")
        if (
            self.kind
            in {
                PhysicalEventKind.HANDOFF,
                PhysicalEventKind.LIFT_THRESHOLD,
                PhysicalEventKind.SUCCESS,
                PhysicalEventKind.FAILURE,
            }
            and self.time_us % metadata.formal_tick_us
        ):
            raise ValueError("formal decision event does not match the formal clock")


@dataclass(frozen=True)
class DeploymentBoundaryView:
    formal_tick_index: int
    time_us: int
    deployment: DeploymentRecord


@dataclass(frozen=True)
class DeploymentEpisodeView:
    schema_version: int
    task_id: str
    instruction: str
    formal_tick_us: int
    action_contract_id: str
    action_dim: int
    boundaries: tuple[DeploymentBoundaryView, ...]
    transitions: tuple[TransitionRecord, ...]


@dataclass(frozen=True)
class SynchronizedEpisode:
    metadata: EpisodeMetadata
    boundaries: tuple[BoundaryRecord, ...]
    transitions: tuple[TransitionRecord, ...]
    physical_events: tuple[PhysicalEventRecord, ...]
    terminal_status: OutcomeStatus
    terminal_reason: TerminalReason

    def __post_init__(self) -> None:
        object.__setattr__(self, "boundaries", tuple(self.boundaries))
        object.__setattr__(self, "transitions", tuple(self.transitions))
        object.__setattr__(self, "physical_events", tuple(self.physical_events))

    def deployment_view(self) -> DeploymentEpisodeView:
        self.validate_complete()
        return DeploymentEpisodeView(
            schema_version=self.metadata.schema_version,
            task_id=self.metadata.task_id,
            instruction=self.metadata.instruction,
            formal_tick_us=self.metadata.formal_tick_us,
            action_contract_id=self.metadata.action_contract_id,
            action_dim=self.metadata.action_dim,
            boundaries=tuple(
                DeploymentBoundaryView(
                    formal_tick_index=boundary.formal_tick_index,
                    time_us=boundary.time_us,
                    deployment=boundary.deployment,
                )
                for boundary in self.boundaries
            ),
            transitions=tuple(self.transitions),
        )

    def validate_complete(self) -> None:
        if len(self.boundaries) < 2:
            raise ValueError("a complete episode requires at least two boundaries")
        if len(self.transitions) != len(self.boundaries) - 1:
            raise ValueError("a complete episode requires T transitions for T+1 boundaries")
        for expected_tick, boundary in enumerate(self.boundaries):
            boundary.validate(self.metadata)
            if boundary.formal_tick_index != expected_tick:
                raise ValueError("episode boundaries must be contiguous and start at tick zero")
            if (
                expected_tick < len(self.boundaries) - 1
                and boundary.outcome_status is not OutcomeStatus.RUNNING
            ):
                raise ValueError("non-final boundaries must have running outcome status")
        for expected_tick, transition in enumerate(self.transitions):
            transition.validate(self.metadata)
            if transition.source_formal_tick != expected_tick:
                raise ValueError("episode transitions must be contiguous and start at tick zero")

        final_time_us = self.boundaries[-1].time_us
        previous_event_time = -1
        event_times: dict[PhysicalEventKind, int] = {}
        events_by_kind: dict[PhysicalEventKind, PhysicalEventRecord] = {}
        for event in self.physical_events:
            event.validate(self.metadata, final_time_us=final_time_us)
            if event.time_us < previous_event_time:
                raise ValueError("physical events must be time ordered")
            if event.kind in event_times:
                raise ValueError("physical milestone events may not be duplicated")
            previous_event_time = event.time_us
            event_times[event.kind] = event.time_us
            events_by_kind[event.kind] = event

        if self.terminal_status not in {OutcomeStatus.SUCCESS, OutcomeStatus.FAILURE}:
            raise ValueError("complete episode terminal_status must be success or failure")
        if not isinstance(self.terminal_reason, TerminalReason):
            raise TypeError("complete episode terminal_reason must be a TerminalReason")
        if self.boundaries[-1].outcome_status is not self.terminal_status:
            raise ValueError("final boundary outcome must match episode terminal_status")
        opposite_terminal = (
            PhysicalEventKind.FAILURE
            if self.terminal_status is OutcomeStatus.SUCCESS
            else PhysicalEventKind.SUCCESS
        )
        if opposite_terminal in event_times:
            raise ValueError("episode contains an opposite terminal event")
        terminal_kind = (
            PhysicalEventKind.SUCCESS
            if self.terminal_status is OutcomeStatus.SUCCESS
            else PhysicalEventKind.FAILURE
        )
        if terminal_kind in events_by_kind and (
            events_by_kind[terminal_kind].terminal_reason is not self.terminal_reason
        ):
            raise ValueError("terminal event reason must match episode terminal reason")

        if self.terminal_status is OutcomeStatus.SUCCESS:
            required = (
                PhysicalEventKind.FIRST_CONTACT,
                PhysicalEventKind.STABLE_GRASP,
                PhysicalEventKind.HANDOFF,
                PhysicalEventKind.LIFT_THRESHOLD,
                PhysicalEventKind.SUCCESS,
            )
            if any(kind not in event_times for kind in required):
                raise ValueError("success milestones are incomplete")
            if [event_times[kind] for kind in required] != sorted(
                event_times[kind] for kind in required
            ):
                raise ValueError("success milestones are out of order")
            if (
                self.terminal_reason is not TerminalReason.LIFT_SUCCEEDED
                or event_times[PhysicalEventKind.SUCCESS] != final_time_us
            ):
                raise ValueError("success terminal reason or time is inconsistent")
        else:
            if self.terminal_reason is TerminalReason.LIFT_SUCCEEDED:
                raise ValueError("failure cannot use the success terminal reason")
            if event_times.get(PhysicalEventKind.FAILURE) != final_time_us:
                raise ValueError("failure event must match the final boundary")
