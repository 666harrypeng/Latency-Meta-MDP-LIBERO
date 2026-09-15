"""Formal source identities layered over verified structured physical records."""

from __future__ import annotations

import re
from dataclasses import dataclass, fields
from typing import Any

from latency_meta_mdp.data.collection.contracts import (
    ExpertRealizationId,
    StrategyFamily,
    TaskInstanceId,
)
from latency_meta_mdp.data.collection.recording_contracts import (
    ImplementationIdentity,
    StructuredBoundaryRecord,
    StructuredPhysicalEventRecord,
    StructuredTransitionRecord,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _strict(value: Any, expected: set[str], *, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be a mapping")
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown or missing:
        raise ValueError(f"{name} fields mismatch; unknown={unknown}, missing={missing}")
    return value


def _text(value: Any, *, name: str, safe_id: bool = False) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty normalized string")
    if safe_id and _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{name} must be one safe path component")
    return value


def _sha(value: Any, *, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class FormalSourceEpisodeMetadata:
    schema_version: int
    record_profile: str
    episode_id: str
    corpus_id: str
    logical_master_task_index: int
    task_instance_id: TaskInstanceId
    expert_realization_id: ExpertRealizationId
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
    strategy_family: StrategyFamily
    formal_corpus_config_sha256: str
    source_corpus_config_sha256: str
    task_config_sha256: str
    motion_config_sha256: str
    runtime_config_sha256: str
    controller_config_sha256: str
    structured_expert_config_sha256: str
    curobo_planner_config_sha256: str
    task_instance_manifest_sha256: str
    frozen_plan_set_manifest_sha256: str
    realization_universe_sha256: str
    strategy_sha256: str
    planner_candidates_sha256: str
    selected_reference_sha256: str
    implementation: ImplementationIdentity
    accepted_slot: int | None = None
    realization_draw_index: int | None = None

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version not in (2, 3):
            raise ValueError("schema_version must equal 2 or 3")
        if self.record_profile != "formal_source":
            raise ValueError("record_profile must equal formal_source")
        _text(self.episode_id, name="episode_id", safe_id=True)
        _text(self.corpus_id, name="corpus_id")
        if type(self.logical_master_task_index) is not int or self.logical_master_task_index < 0:
            raise ValueError("logical_master_task_index must be non-negative")
        if not isinstance(self.task_instance_id, TaskInstanceId):
            raise TypeError("task_instance_id must be a TaskInstanceId")
        if not isinstance(self.expert_realization_id, ExpertRealizationId):
            raise TypeError("expert_realization_id must be an ExpertRealizationId")
        key = self.expert_realization_id.expert_realization_key
        if key.task_instance_id != self.task_instance_id:
            raise ValueError("realization task identity does not match source metadata")
        if self.expert_realization_id.task_instance_plan_set_sha256 != (
            self.frozen_plan_set_manifest_sha256
        ):
            raise ValueError("realization plan-set identity does not match source metadata")
        if self.schema_version == 2:
            if self.accepted_slot is not None or self.realization_draw_index is not None:
                raise ValueError("schema 2 source metadata cannot carry quota identities")
        else:
            if type(self.accepted_slot) is not int or self.accepted_slot < 0:
                raise ValueError("accepted_slot must be a non-negative integer")
            if (
                type(self.realization_draw_index) is not int
                or self.realization_draw_index < 0
                or self.realization_draw_index != key.realization_index
            ):
                raise ValueError("realization_draw_index must match the expert realization key")
        if self.task_id != "dynamic_grasp_lift":
            raise ValueError("unsupported task_id")
        _text(self.instruction, name="instruction")
        if self.physics_dt_us != 2_000 or self.formal_tick_us != 20_000:
            raise ValueError("source clock must use 2 ms physics and 20 ms formal ticks")
        for name in ("camera_height", "camera_width"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            self.action_contract_id != "panda_osc_pose_delta_v1"
            or self.action_dim != 7
            or self.actuator_dim != 9
        ):
            raise ValueError("source action contract is invalid")
        if self.expert_id != "panda_ball_smooth_approach_canonical_grasp_v3":
            raise ValueError("source expert_id is invalid")
        if not isinstance(self.strategy_family, StrategyFamily):
            raise TypeError("strategy_family must be a StrategyFamily")
        for item in fields(self):
            if item.name.endswith("_sha256"):
                _sha(getattr(self, item.name), name=item.name)
        if not isinstance(self.implementation, ImplementationIdentity):
            raise TypeError("implementation must be an ImplementationIdentity")

    def to_mapping(self) -> dict[str, Any]:
        result = {item.name: getattr(self, item.name) for item in fields(self)}
        if self.schema_version == 2:
            result.pop("accepted_slot")
            result.pop("realization_draw_index")
        result["task_instance_id"] = self.task_instance_id.to_mapping()
        result["expert_realization_id"] = self.expert_realization_id.to_mapping()
        result["strategy_family"] = self.strategy_family.value
        result["implementation"] = self.implementation.to_mapping()
        return result

    @classmethod
    def from_mapping(cls, mapping: Any) -> FormalSourceEpisodeMetadata:
        if type(mapping) is not dict:
            raise TypeError("FormalSourceEpisodeMetadata must be a mapping")
        schema_version = mapping.get("schema_version")
        optional = {"accepted_slot", "realization_draw_index"}
        expected = {item.name for item in fields(cls)}
        if schema_version == 2:
            expected -= optional
        raw = _strict(
            mapping,
            expected,
            name=cls.__name__,
        ).copy()
        if schema_version == 2:
            raw.update(accepted_slot=None, realization_draw_index=None)
        raw["task_instance_id"] = TaskInstanceId.from_mapping(raw["task_instance_id"])
        raw["expert_realization_id"] = ExpertRealizationId.from_mapping(
            raw["expert_realization_id"]
        )
        raw["strategy_family"] = StrategyFamily(raw["strategy_family"])
        raw["implementation"] = ImplementationIdentity.from_mapping(raw["implementation"])
        return cls(**raw)


@dataclass(frozen=True)
class FormalSourceSynchronizedEpisode:
    metadata: FormalSourceEpisodeMetadata
    boundaries: tuple[StructuredBoundaryRecord, ...]
    transitions: tuple[StructuredTransitionRecord, ...]
    physical_events: tuple[StructuredPhysicalEventRecord, ...]
    terminal_reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, FormalSourceEpisodeMetadata):
            raise TypeError("metadata must be FormalSourceEpisodeMetadata")
        for name, record_type in (
            ("boundaries", StructuredBoundaryRecord),
            ("transitions", StructuredTransitionRecord),
            ("physical_events", StructuredPhysicalEventRecord),
        ):
            values = getattr(self, name)
            if type(values) is not tuple or any(
                not isinstance(value, record_type) for value in values
            ):
                raise TypeError(f"{name} must be a tuple of {record_type.__name__}")
        if not self.boundaries or len(self.boundaries) != len(self.transitions) + 1:
            raise ValueError("source episode must have B = T + 1")
        if [item.formal_tick_index for item in self.boundaries] != list(
            range(len(self.boundaries))
        ):
            raise ValueError("source boundary clocks must be contiguous from zero")
        for index, transition in enumerate(self.transitions):
            if transition.source_formal_tick != index or transition.target_formal_tick != index + 1:
                raise ValueError("source transition clocks must be contiguous with boundaries")
            if transition.expert_audit.expert_realization_id != self.metadata.expert_realization_id:
                raise ValueError("source audit realization identity does not match metadata")
        if any(item.outcome_status != "running" for item in self.boundaries[:-1]):
            raise ValueError("only the terminal boundary may have a terminal outcome")
        if self.boundaries[-1].outcome_status != "success":
            raise ValueError("source terminal boundary must have outcome_status=success")
        event_times = [event.time_us for event in self.physical_events]
        if event_times != sorted(event_times):
            raise ValueError("source physical events must be time ordered")
        success_events = [event for event in self.physical_events if event.kind == "success"]
        if len(success_events) != 1:
            raise ValueError("source episode requires one terminal success event")
        success = success_events[0]
        if success.time_us != self.boundaries[-1].time_us:
            raise ValueError("source success event must match terminal boundary time")
        if success.terminal_reason != self.terminal_reason:
            raise ValueError("source success event reason must match terminal_reason")
        height = self.metadata.camera_height
        width = self.metadata.camera_width
        if any(
            boundary.deployment.agentview_rgb.shape[:2] != (height, width)
            or boundary.deployment.robot0_eye_in_hand_rgb.shape[:2] != (height, width)
            for boundary in self.boundaries
        ):
            raise ValueError("source camera dimensions do not match metadata")
        _text(self.terminal_reason, name="terminal_reason")

    @property
    def terminal_status(self) -> str:
        return "success"
