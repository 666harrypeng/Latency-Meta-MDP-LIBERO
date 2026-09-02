"""Independent immutable contracts for structured-expert episode recording."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any

import numpy as np

from latency_meta_mdp.expert_realization.contracts import (
    AttemptId,
    ExpertRealizationId,
    StrategyFamily,
    TaskInstanceId,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OUTCOME = {"running", "success", "failure"}
_HANDOFF = {"driven", "physical", "failure"}


def _strict_mapping(value: Any, expected: set[str], *, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be a mapping")
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown or missing:
        raise ValueError(f"{name} fields mismatch; unknown={unknown}, missing={missing}")
    return value


def _sha(value: Any, *, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _text(value: Any, *, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any] | type[Any],
) -> np.ndarray:
    source = np.asarray(value)
    expected = np.dtype(dtype)
    if source.dtype != expected or source.shape != shape:
        raise ValueError(f"{name} must have dtype {expected} and shape {shape}")
    if np.issubdtype(expected, np.floating) and not np.all(np.isfinite(source)):
        raise ValueError(f"{name} must be finite")
    result = np.array(source, dtype=expected, copy=True)
    result.setflags(write=False)
    return result


def _rotation(value: Any, *, name: str) -> np.ndarray:
    result = _array(value, name=name, shape=(3, 3), dtype=np.float64)
    if not np.allclose(result.T @ result, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(result), 1.0, atol=1e-6
    ):
        raise ValueError(f"{name} must be a proper rotation matrix")
    return result


def _freeze_json(value: Any, *, name: str = "JSON value") -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError(f"{name} mapping keys must be strings")
        return MappingProxyType({key: _freeze_json(item, name=name) for key, item in value.items()})
    if type(value) in (list, tuple):
        return tuple(_freeze_json(item, name=name) for item in value)
    raise TypeError(f"{name} contains a non-JSON value")


def json_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: json_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [json_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class ImplementationIdentity:
    revision: str
    source_sha256: str
    dirty: bool

    def __post_init__(self) -> None:
        _text(self.revision, name="revision")
        _sha(self.source_sha256, name="source_sha256")
        if type(self.dirty) is not bool:
            raise TypeError("dirty must be a boolean")

    def to_mapping(self) -> dict[str, Any]:
        return {"revision": self.revision, "source_sha256": self.source_sha256, "dirty": self.dirty}

    @classmethod
    def from_mapping(cls, mapping: Any) -> ImplementationIdentity:
        return cls(
            **_strict_mapping(mapping, {"revision", "source_sha256", "dirty"}, name=cls.__name__)
        )


@dataclass(frozen=True)
class StructuredEpisodeMetadata:
    schema_version: int
    episode_id: str
    task_instance_id: TaskInstanceId
    expert_realization_id: ExpertRealizationId
    attempt_id: AttemptId
    task_id: str
    instruction: str
    physics_dt_us: int
    formal_tick_us: int
    camera_height: int
    camera_width: int
    action_contract_id: str
    action_dim: int
    actuator_dim: int
    expert_id: str
    record_profile: str
    strategy_family: StrategyFamily
    task_config_sha256: str
    motion_config_sha256: str
    runtime_config_sha256: str
    controller_config_sha256: str
    structured_expert_config_sha256: str
    curobo_planner_config_sha256: str
    pilot_config_sha256: str
    pilot_gate_config_sha256: str
    task_instance_manifest_sha256: str
    frozen_plan_set_manifest_sha256: str
    strategy_sha256: str
    planner_candidates_sha256: str
    selected_reference_sha256: str
    implementation: ImplementationIdentity

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("episode schema_version must be 1")
        for name in (
            "episode_id",
            "instruction",
            "action_contract_id",
            "expert_id",
            "record_profile",
        ):
            _text(getattr(self, name), name=name)
        if self.task_id != "dynamic_grasp_lift":
            raise ValueError("unsupported task_id")
        if self.physics_dt_us != 2_000 or self.formal_tick_us != 20_000:
            raise ValueError("episode must use the 2 ms / 20 ms clock")
        if (
            self.action_contract_id != "panda_osc_pose_delta_v1"
            or self.action_dim != 7
            or self.actuator_dim != 9
        ):
            raise ValueError("unsupported Panda action contract")
        if self.expert_id != "panda_ball_structured_v1" or self.record_profile != "pilot_debug":
            raise ValueError("unsupported structured expert recording profile")
        _integer(self.camera_height, name="camera_height", minimum=1)
        _integer(self.camera_width, name="camera_width", minimum=1)
        if not isinstance(self.task_instance_id, TaskInstanceId):
            raise TypeError("task_instance_id must be a TaskInstanceId")
        if not isinstance(self.expert_realization_id, ExpertRealizationId):
            raise TypeError("expert_realization_id must be an ExpertRealizationId")
        if not isinstance(self.attempt_id, AttemptId):
            raise TypeError("attempt_id must be an AttemptId")
        if not isinstance(self.strategy_family, StrategyFamily):
            raise TypeError("strategy_family must be a StrategyFamily")
        if not isinstance(self.implementation, ImplementationIdentity):
            raise TypeError("implementation must be an ImplementationIdentity")
        for item in fields(self):
            if item.name.endswith("_sha256"):
                _sha(getattr(self, item.name), name=item.name)
        key = self.expert_realization_id.expert_realization_key
        if key.task_instance_id != self.task_instance_id:
            raise ValueError("realization task identity does not match episode task identity")
        if self.attempt_id.expert_realization_id != self.expert_realization_id:
            raise ValueError("attempt identity does not match episode realization identity")
        if (
            self.expert_realization_id.task_instance_plan_set_sha256
            != self.frozen_plan_set_manifest_sha256
        ):
            raise ValueError("realization plan-set hash does not match episode metadata")

    @property
    def physics_steps_per_tick(self) -> int:
        return self.formal_tick_us // self.physics_dt_us

    def to_mapping(self) -> dict[str, Any]:
        result = {item.name: getattr(self, item.name) for item in fields(self)}
        result["task_instance_id"] = self.task_instance_id.to_mapping()
        result["expert_realization_id"] = self.expert_realization_id.to_mapping()
        result["attempt_id"] = self.attempt_id.to_mapping()
        result["strategy_family"] = self.strategy_family.value
        result["implementation"] = self.implementation.to_mapping()
        return result

    @classmethod
    def from_mapping(cls, mapping: Any) -> StructuredEpisodeMetadata:
        raw = _strict_mapping(
            mapping, {item.name for item in fields(cls)}, name=cls.__name__
        ).copy()
        config_sha = raw.get("structured_expert_config_sha256")
        _sha(config_sha, name="structured_expert_config_sha256")
        raw["task_instance_id"] = TaskInstanceId.from_mapping(raw["task_instance_id"])
        raw["expert_realization_id"] = ExpertRealizationId.from_mapping(
            raw["expert_realization_id"], structured_expert_config_sha256=config_sha
        )
        raw["attempt_id"] = AttemptId.from_mapping(
            raw["attempt_id"], structured_expert_config_sha256=config_sha
        )
        raw["strategy_family"] = StrategyFamily(raw["strategy_family"])
        raw["implementation"] = ImplementationIdentity.from_mapping(raw["implementation"])
        return cls(**raw)


@dataclass(frozen=True)
class StructuredDeploymentRecord:
    source_physics_step: int
    source_formal_tick: int
    source_time_us: int
    agentview_rgb: np.ndarray
    robot0_eye_in_hand_rgb: np.ndarray
    robot_qpos: np.ndarray
    robot_qvel: np.ndarray
    gripper_qpos: np.ndarray
    gripper_qvel: np.ndarray
    eef_position_world: np.ndarray
    eef_orientation_matrix_world: np.ndarray

    def __post_init__(self) -> None:
        _integer(self.source_formal_tick, name="source_formal_tick")
        if (
            self.source_physics_step != self.source_formal_tick * 10
            or self.source_time_us != self.source_formal_tick * 20_000
        ):
            raise ValueError("deployment source clock is inconsistent")
        for name in ("agentview_rgb", "robot0_eye_in_hand_rgb"):
            source = np.asarray(getattr(self, name))
            if source.dtype != np.uint8 or source.ndim != 3 or source.shape[-1] != 3:
                raise ValueError(f"{name} must be uint8 HxWx3")
            result = np.array(source, copy=True)
            result.setflags(write=False)
            object.__setattr__(self, name, result)
        for name, length in (
            ("robot_qpos", 7),
            ("robot_qvel", 7),
            ("gripper_qpos", 2),
            ("gripper_qvel", 2),
            ("eef_position_world", 3),
        ):
            object.__setattr__(
                self,
                name,
                _array(getattr(self, name), name=name, shape=(length,), dtype=np.float64),
            )
        object.__setattr__(
            self,
            "eef_orientation_matrix_world",
            _rotation(self.eef_orientation_matrix_world, name="eef_orientation_matrix_world"),
        )


@dataclass(frozen=True)
class StructuredQualificationRecord:
    object_pose: np.ndarray
    object_velocity: np.ndarray
    commanded_motion_position: np.ndarray
    commanded_motion_velocity: np.ndarray
    commanded_motion_acceleration: np.ndarray
    commanded_motion_segment_index: int
    left_pad_contact: bool
    right_pad_contact: bool
    handoff_state: str
    relative_geometry: np.ndarray
    actuator_ctrl: np.ndarray
    applied_reference: np.ndarray | None
    applied_reference_source_tick: int | None
    nullspace_joint_position_error: np.ndarray | None
    eef_position_error: np.ndarray | None
    eef_orientation_error_rotvec: np.ndarray | None

    def __post_init__(self) -> None:
        for name, length in (
            ("object_pose", 7),
            ("object_velocity", 6),
            ("commanded_motion_position", 3),
            ("commanded_motion_velocity", 3),
            ("commanded_motion_acceleration", 3),
            ("relative_geometry", 3),
            ("actuator_ctrl", 9),
        ):
            object.__setattr__(
                self,
                name,
                _array(getattr(self, name), name=name, shape=(length,), dtype=np.float64),
            )
        _integer(self.commanded_motion_segment_index, name="commanded_motion_segment_index")
        if type(self.left_pad_contact) is not bool or type(self.right_pad_contact) is not bool:
            raise TypeError("pad contacts must be booleans")
        if self.handoff_state not in _HANDOFF:
            raise ValueError("handoff_state is invalid")
        optional = (
            self.applied_reference,
            self.applied_reference_source_tick,
            self.nullspace_joint_position_error,
            self.eef_position_error,
            self.eef_orientation_error_rotvec,
        )
        if all(item is None for item in optional):
            return
        if any(item is None for item in optional):
            raise ValueError("control reference and all errors must be jointly present or absent")
        _integer(self.applied_reference_source_tick, name="applied_reference_source_tick")
        for name, length in (
            ("applied_reference", 7),
            ("nullspace_joint_position_error", 7),
            ("eef_position_error", 3),
            ("eef_orientation_error_rotvec", 3),
        ):
            object.__setattr__(
                self,
                name,
                _array(getattr(self, name), name=name, shape=(length,), dtype=np.float64),
            )


@dataclass(frozen=True)
class StructuredExpertAuditRecord:
    expert_realization_id: ExpertRealizationId
    source_physics_step: int
    source_formal_tick: int
    source_time_us: int
    phase_id: str
    reference_kind: str
    selected_reference_index: int | None
    target_eef_position_world: np.ndarray
    target_eef_orientation_matrix_world: np.ndarray
    estimated_object_velocity_world: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.expert_realization_id, ExpertRealizationId):
            raise TypeError("expert_realization_id must be an ExpertRealizationId")
        _integer(self.source_formal_tick, name="source_formal_tick")
        if (
            self.source_physics_step != self.source_formal_tick * 10
            or self.source_time_us != self.source_formal_tick * 20_000
        ):
            raise ValueError("audit source clock is inconsistent")
        _text(self.phase_id, name="phase_id")
        if self.reference_kind not in {"shared_prefix", "selected_reference"}:
            raise ValueError("reference_kind is invalid")
        if self.reference_kind == "shared_prefix":
            if self.selected_reference_index is not None:
                raise ValueError("shared_prefix cannot select a reference index")
        else:
            _integer(self.selected_reference_index, name="selected_reference_index")
        object.__setattr__(
            self,
            "target_eef_position_world",
            _array(
                self.target_eef_position_world,
                name="target_eef_position_world",
                shape=(3,),
                dtype=np.float64,
            ),
        )
        object.__setattr__(
            self,
            "target_eef_orientation_matrix_world",
            _rotation(
                self.target_eef_orientation_matrix_world, name="target_eef_orientation_matrix_world"
            ),
        )
        object.__setattr__(
            self,
            "estimated_object_velocity_world",
            _array(
                self.estimated_object_velocity_world,
                name="estimated_object_velocity_world",
                shape=(3,),
                dtype=np.float64,
            ),
        )


@dataclass(frozen=True)
class StructuredTransitionRecord:
    source_formal_tick: int
    target_formal_tick: int
    expert_action: np.ndarray
    action_mask: np.ndarray
    expert_audit: StructuredExpertAuditRecord

    def __post_init__(self) -> None:
        _integer(self.source_formal_tick, name="source_formal_tick")
        if self.target_formal_tick != self.source_formal_tick + 1:
            raise ValueError("transition target tick must equal source tick + 1")
        object.__setattr__(
            self,
            "expert_action",
            _array(self.expert_action, name="expert_action", shape=(7,), dtype=np.float64),
        )
        object.__setattr__(
            self,
            "action_mask",
            _array(self.action_mask, name="action_mask", shape=(7,), dtype=np.bool_),
        )
        if not isinstance(self.expert_audit, StructuredExpertAuditRecord):
            raise TypeError("expert_audit must be a StructuredExpertAuditRecord")
        if self.expert_audit.source_formal_tick != self.source_formal_tick:
            raise ValueError("audit source tick must equal transition source tick")


@dataclass(frozen=True)
class StructuredBoundaryRecord:
    formal_tick_index: int
    physics_step_index: int
    time_us: int
    deployment: StructuredDeploymentRecord
    qualification: StructuredQualificationRecord
    outcome_status: str

    def __post_init__(self) -> None:
        _integer(self.formal_tick_index, name="formal_tick_index")
        if (
            self.physics_step_index != self.formal_tick_index * 10
            or self.time_us != self.formal_tick_index * 20_000
        ):
            raise ValueError("boundary clock is inconsistent")
        if not isinstance(self.deployment, StructuredDeploymentRecord) or not isinstance(
            self.qualification, StructuredQualificationRecord
        ):
            raise TypeError("boundary payloads have incorrect types")
        if (
            self.deployment.source_formal_tick != self.formal_tick_index
            or self.deployment.source_physics_step != self.physics_step_index
            or self.deployment.source_time_us != self.time_us
        ):
            raise ValueError("deployment source clock does not match boundary clock")
        if self.outcome_status not in _OUTCOME:
            raise ValueError("outcome_status is invalid")
        if self.formal_tick_index == 0 and self.qualification.applied_reference is not None:
            raise ValueError("boundary zero cannot have an applied reference")
        if (
            self.formal_tick_index > 0
            and self.qualification.applied_reference_source_tick != self.formal_tick_index - 1
        ):
            raise ValueError("applied reference must come from the previous formal tick")


@dataclass(frozen=True)
class StructuredPhysicalEventRecord:
    kind: str
    physics_step_index: int
    time_us: int
    payload: Mapping[str, Any]
    terminal_reason: str | None

    def __post_init__(self) -> None:
        _text(self.kind, name="kind")
        _integer(self.physics_step_index, name="physics_step_index")
        if self.time_us != self.physics_step_index * 2_000:
            raise ValueError("physical event clock is inconsistent")
        if type(self.payload) is not dict:
            raise TypeError("event payload must be a mapping")
        object.__setattr__(self, "payload", _freeze_json(self.payload, name="event payload"))
        if self.terminal_reason is not None:
            _text(self.terminal_reason, name="terminal_reason")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "physics_step_index": self.physics_step_index,
            "time_us": self.time_us,
            "payload": json_thaw(self.payload),
            "terminal_reason": self.terminal_reason,
        }

    @classmethod
    def from_mapping(cls, mapping: Any) -> StructuredPhysicalEventRecord:
        return cls(
            **_strict_mapping(mapping, {item.name for item in fields(cls)}, name=cls.__name__)
        )


@dataclass(frozen=True)
class StructuredSynchronizedEpisode:
    metadata: StructuredEpisodeMetadata
    boundaries: tuple[StructuredBoundaryRecord, ...]
    transitions: tuple[StructuredTransitionRecord, ...]
    physical_events: tuple[StructuredPhysicalEventRecord, ...]
    terminal_status: str
    terminal_reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, StructuredEpisodeMetadata):
            raise TypeError("metadata must be StructuredEpisodeMetadata")
        for name, item_type in (
            ("boundaries", StructuredBoundaryRecord),
            ("transitions", StructuredTransitionRecord),
            ("physical_events", StructuredPhysicalEventRecord),
        ):
            values = getattr(self, name)
            if type(values) is not tuple or any(not isinstance(item, item_type) for item in values):
                raise TypeError(f"{name} must be a tuple of {item_type.__name__}")
        if not self.boundaries or len(self.boundaries) != len(self.transitions) + 1:
            raise ValueError("episode must have B = T + 1")
        if [item.formal_tick_index for item in self.boundaries] != list(
            range(len(self.boundaries))
        ):
            raise ValueError("boundary formal ticks must be contiguous from zero")
        for index, transition in enumerate(self.transitions):
            if transition.source_formal_tick != index or transition.target_formal_tick != index + 1:
                raise ValueError("transition clocks must be contiguous with boundaries")
            if transition.expert_audit.expert_realization_id != self.metadata.expert_realization_id:
                raise ValueError("audit realization identity does not match metadata")
        height = self.metadata.camera_height
        width = self.metadata.camera_width
        for boundary in self.boundaries:
            if boundary.deployment.agentview_rgb.shape[:2] != (
                height,
                width,
            ) or boundary.deployment.robot0_eye_in_hand_rgb.shape[:2] != (height, width):
                raise ValueError("camera dimensions do not match metadata")
        if self.terminal_status not in {"success", "failure"}:
            raise ValueError("terminal_status must be success or failure")
        _text(self.terminal_reason, name="terminal_reason")

    def deployment_view(self) -> Mapping[str, Any]:
        boundaries = tuple(
            MappingProxyType(
                {
                    "formal_tick_index": item.formal_tick_index,
                    "physics_step_index": item.physics_step_index,
                    "time_us": item.time_us,
                    "agentview_rgb": item.deployment.agentview_rgb,
                    "robot0_eye_in_hand_rgb": item.deployment.robot0_eye_in_hand_rgb,
                    "robot_qpos": item.deployment.robot_qpos,
                    "robot_qvel": item.deployment.robot_qvel,
                    "gripper_qpos": item.deployment.gripper_qpos,
                    "gripper_qvel": item.deployment.gripper_qvel,
                    "eef_position_world": item.deployment.eef_position_world,
                    "eef_orientation_matrix_world": item.deployment.eef_orientation_matrix_world,
                    "outcome_status": item.outcome_status,
                }
            )
            for item in self.boundaries
        )
        transitions = tuple(
            MappingProxyType(
                {
                    "source_formal_tick": item.source_formal_tick,
                    "target_formal_tick": item.target_formal_tick,
                    "expert_action": item.expert_action,
                    "action_mask": item.action_mask,
                }
            )
            for item in self.transitions
        )
        return MappingProxyType(
            {
                "task_id": self.metadata.task_id,
                "instruction": self.metadata.instruction,
                "physics_dt_us": self.metadata.physics_dt_us,
                "formal_tick_us": self.metadata.formal_tick_us,
                "camera_height": self.metadata.camera_height,
                "camera_width": self.metadata.camera_width,
                "action_contract_id": self.metadata.action_contract_id,
                "action_dim": self.metadata.action_dim,
                "boundaries": boundaries,
                "transitions": transitions,
            }
        )
