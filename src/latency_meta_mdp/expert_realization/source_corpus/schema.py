"""Canonical Arrow schemas and semantic field roles for source recordings."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import pyarrow as pa


class SourceFieldRole(str, Enum):
    DEPLOYMENT_INPUT = "deployment_input"
    SUPERVISION_CANDIDATE = "supervision_candidate"
    AUDIT_ONLY = "audit_only"
    IDENTITY = "identity"


@dataclass(frozen=True)
class SourceFieldSpec:
    name: str
    arrow_type: pa.DataType
    dtype: str
    shape: tuple[int, ...] | None
    unit: str | None
    nullable: bool
    roles: tuple[SourceFieldRole, ...]

    @property
    def arrow_field(self) -> pa.Field:
        return pa.field(self.name, self.arrow_type, nullable=self.nullable)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": None if self.shape is None else list(self.shape),
            "unit": self.unit,
            "nullable": self.nullable,
            "roles": [role.value for role in self.roles],
        }


_IDENTITY = (SourceFieldRole.IDENTITY,)
_DEPLOYMENT_TARGET = (
    SourceFieldRole.DEPLOYMENT_INPUT,
    SourceFieldRole.SUPERVISION_CANDIDATE,
)
_SUPERVISION = (SourceFieldRole.SUPERVISION_CANDIDATE,)
_AUDIT = (SourceFieldRole.AUDIT_ONLY,)
_IMAGE = pa.struct(
    [
        pa.field("bytes", pa.binary(), nullable=False),
        pa.field("path", pa.string(), nullable=False),
    ]
)


def _scalar(
    name: str,
    arrow_type: pa.DataType,
    *,
    dtype: str,
    unit: str | None,
    nullable: bool,
    roles: tuple[SourceFieldRole, ...],
) -> SourceFieldSpec:
    return SourceFieldSpec(name, arrow_type, dtype, None, unit, nullable, roles)


def _vector(
    name: str,
    length: int,
    *,
    shape: tuple[int, ...] | None = None,
    dtype: pa.DataType = pa.float64(),
    dtype_name: str = "float64",
    unit: str | None,
    nullable: bool = False,
    roles: tuple[SourceFieldRole, ...],
) -> SourceFieldSpec:
    return SourceFieldSpec(
        name=name,
        arrow_type=pa.list_(dtype) if nullable else pa.list_(dtype, length),
        dtype=dtype_name,
        shape=(length,) if shape is None else shape,
        unit=unit,
        nullable=nullable,
        roles=roles,
    )


SOURCE_FRAME_FIELDS = (
    _scalar("episode_id", pa.string(), dtype="string", unit=None, nullable=False, roles=_IDENTITY),
    _scalar("formal_tick", pa.int64(), dtype="int64", unit="tick", nullable=False, roles=_IDENTITY),
    _scalar(
        "physics_step",
        pa.int64(),
        dtype="int64",
        unit="physics_step",
        nullable=False,
        roles=_IDENTITY,
    ),
    _scalar("time_us", pa.int64(), dtype="int64", unit="us", nullable=False, roles=_IDENTITY),
    SourceFieldSpec(
        "agentview_rgb", _IMAGE, "image/png", (256, 256, 3), None, False, _DEPLOYMENT_TARGET
    ),
    SourceFieldSpec(
        "wrist_rgb", _IMAGE, "image/png", (256, 256, 3), None, False, _DEPLOYMENT_TARGET
    ),
    _vector("robot_qpos", 7, unit="rad", roles=_DEPLOYMENT_TARGET),
    _vector("robot_qvel", 7, unit="rad/s", roles=_DEPLOYMENT_TARGET),
    _vector("gripper_qpos", 2, unit="m", roles=_DEPLOYMENT_TARGET),
    _vector("gripper_qvel", 2, unit="m/s", roles=_DEPLOYMENT_TARGET),
    _vector("eef_position_world", 3, unit="m", roles=_DEPLOYMENT_TARGET),
    _vector(
        "eef_orientation_matrix_world",
        9,
        shape=(3, 3),
        unit="dimensionless",
        roles=_DEPLOYMENT_TARGET,
    ),
    _scalar("outcome_status", pa.string(), dtype="string", unit=None, nullable=False, roles=_AUDIT),
    _vector("object_pose", 7, unit="m+quaternion", roles=_SUPERVISION),
    _vector("object_velocity", 6, unit="m/s+rad/s", roles=_SUPERVISION),
    _vector("commanded_motion_position", 3, unit="m", roles=_AUDIT),
    _vector("commanded_motion_velocity", 3, unit="m/s", roles=_AUDIT),
    _vector("commanded_motion_acceleration", 3, unit="m/s^2", roles=_AUDIT),
    _scalar(
        "commanded_motion_segment_index",
        pa.int64(),
        dtype="int64",
        unit="index",
        nullable=False,
        roles=_AUDIT,
    ),
    _scalar("left_pad_contact", pa.bool_(), dtype="bool", unit=None, nullable=False, roles=_AUDIT),
    _scalar("right_pad_contact", pa.bool_(), dtype="bool", unit=None, nullable=False, roles=_AUDIT),
    _scalar("handoff_state", pa.string(), dtype="string", unit=None, nullable=False, roles=_AUDIT),
    _vector("relative_geometry", 3, unit="m", roles=_AUDIT),
    _vector("actuator_ctrl", 9, unit="actuator", roles=_AUDIT),
    _vector("applied_reference", 7, unit="mixed", nullable=True, roles=_AUDIT),
    _scalar(
        "applied_reference_source_tick",
        pa.int64(),
        dtype="int64",
        unit="tick",
        nullable=True,
        roles=_AUDIT,
    ),
    _vector("nullspace_joint_position_error", 7, unit="rad", nullable=True, roles=_AUDIT),
    _vector("eef_position_error", 3, unit="m", nullable=True, roles=_AUDIT),
    _vector("eef_orientation_error_rotvec", 3, unit="rad", nullable=True, roles=_AUDIT),
    _vector("expert_action", 7, unit="normalized_action", nullable=True, roles=_DEPLOYMENT_TARGET),
    _vector(
        "action_mask",
        7,
        dtype=pa.bool_(),
        dtype_name="bool",
        unit=None,
        nullable=True,
        roles=_DEPLOYMENT_TARGET,
    ),
    _scalar("phase_id", pa.string(), dtype="string", unit=None, nullable=True, roles=_AUDIT),
    _scalar("reference_kind", pa.string(), dtype="string", unit=None, nullable=True, roles=_AUDIT),
    _scalar(
        "selected_reference_index",
        pa.int64(),
        dtype="int64",
        unit="index",
        nullable=True,
        roles=_AUDIT,
    ),
    _vector("target_eef_position_world", 3, unit="m", nullable=True, roles=_AUDIT),
    _vector(
        "target_eef_orientation_matrix_world",
        9,
        shape=(3, 3),
        unit="dimensionless",
        nullable=True,
        roles=_AUDIT,
    ),
    _vector("estimated_object_velocity_world", 3, unit="m/s", nullable=True, roles=_AUDIT),
)

SOURCE_FRAME_SCHEMA = pa.schema([item.arrow_field for item in SOURCE_FRAME_FIELDS])

TASK_INSTANCE_SCHEMA_V1_SPLIT = pa.schema(
    [
        pa.field("task_instance_id", pa.string(), nullable=False),
        pa.field("corpus_id", pa.string(), nullable=False),
        pa.field("logical_master_task_index", pa.int64(), nullable=False),
        pa.field("task_instance_seed", pa.uint64(), nullable=False),
        pa.field("level", pa.int8(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("instruction", pa.string(), nullable=False),
        pa.field("motion_profile_sha256", pa.string(), nullable=False),
        pa.field("initial_state_sha256", pa.string(), nullable=False),
        pa.field("motion_profile_json", pa.string(), nullable=False),
        pa.field("initial_state_npz", pa.binary(), nullable=False),
        pa.field("admitted_realization_count", pa.int32(), nullable=False),
    ]
)

EPISODE_SCHEMA_V1_SPLIT = pa.schema(
    [
        pa.field("episode_id", pa.string(), nullable=False),
        pa.field("task_instance_id", pa.string(), nullable=False),
        pa.field("logical_master_task_index", pa.int64(), nullable=False),
        pa.field("level", pa.int8(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("realization_index", pa.int32(), nullable=False),
        pa.field("realization_seed", pa.uint64(), nullable=False),
        pa.field("strategy_family", pa.string(), nullable=False),
        pa.field("strategy_parameters_json", pa.string(), nullable=False),
        pa.field("frame_count", pa.int32(), nullable=False),
        pa.field("terminal_tick", pa.int64(), nullable=False),
        pa.field("first_contact_time_us", pa.int64(), nullable=False),
        pa.field("stable_grasp_time_us", pa.int64(), nullable=False),
        pa.field("handoff_time_us", pa.int64(), nullable=False),
        pa.field("lift_threshold_time_us", pa.int64(), nullable=False),
        pa.field("success_time_us", pa.int64(), nullable=False),
        pa.field("selected_planner_fingerprint", pa.string(), nullable=False),
        pa.field("qualification_json", pa.string(), nullable=False),
        pa.field("data_shard", pa.string(), nullable=False),
        pa.field("row_group_index", pa.int32(), nullable=False),
        pa.field("row_offset", pa.int64(), nullable=False),
        pa.field("row_count", pa.int32(), nullable=False),
    ]
)

TASK_INSTANCE_SCHEMA = pa.schema(
    [field for field in TASK_INSTANCE_SCHEMA_V1_SPLIT if field.name != "split"]
)

EPISODE_SCHEMA = pa.schema(
    [field for field in EPISODE_SCHEMA_V1_SPLIT if field.name != "split"]
)

EVENT_SCHEMA = pa.schema(
    [
        pa.field("episode_id", pa.string(), nullable=False),
        pa.field("event_index", pa.int32(), nullable=False),
        pa.field("kind", pa.string(), nullable=False),
        pa.field("physics_step", pa.int64(), nullable=False),
        pa.field("time_us", pa.int64(), nullable=False),
        pa.field("payload_json", pa.string(), nullable=False),
        pa.field("terminal_reason", pa.string(), nullable=True),
    ]
)


def fields_for_role(role: SourceFieldRole) -> tuple[str, ...]:
    if not isinstance(role, SourceFieldRole):
        raise TypeError("role must be a SourceFieldRole")
    return tuple(item.name for item in SOURCE_FRAME_FIELDS if role in item.roles)


def source_schema_document() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "format_id": "structured_expert_source_parquet_v2",
        "source_frame_fields": [item.to_mapping() for item in SOURCE_FRAME_FIELDS],
        "task_instance_fields": list(TASK_INSTANCE_SCHEMA.names),
        "episode_fields": list(EPISODE_SCHEMA.names),
        "event_fields": list(EVENT_SCHEMA.names),
    }


def legacy_split_source_schema_document() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "format_id": "structured_expert_source_parquet_v1",
        "source_frame_fields": [item.to_mapping() for item in SOURCE_FRAME_FIELDS],
        "task_instance_fields": list(TASK_INSTANCE_SCHEMA_V1_SPLIT.names),
        "episode_fields": list(EPISODE_SCHEMA_V1_SPLIT.names),
        "event_fields": list(EVENT_SCHEMA.names),
    }
