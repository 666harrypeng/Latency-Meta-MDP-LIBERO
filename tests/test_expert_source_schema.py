from __future__ import annotations

import pyarrow as pa
import pytest

EXPECTED_FRAME_FIELDS = (
    "episode_id",
    "formal_tick",
    "physics_step",
    "time_us",
    "agentview_rgb",
    "wrist_rgb",
    "robot_qpos",
    "robot_qvel",
    "gripper_qpos",
    "gripper_qvel",
    "eef_position_world",
    "eef_orientation_matrix_world",
    "outcome_status",
    "object_pose",
    "object_velocity",
    "commanded_motion_position",
    "commanded_motion_velocity",
    "commanded_motion_acceleration",
    "commanded_motion_segment_index",
    "left_pad_contact",
    "right_pad_contact",
    "handoff_state",
    "relative_geometry",
    "actuator_ctrl",
    "applied_reference",
    "applied_reference_source_tick",
    "nullspace_joint_position_error",
    "eef_position_error",
    "eef_orientation_error_rotvec",
    "expert_action",
    "action_mask",
    "phase_id",
    "reference_kind",
    "selected_reference_index",
    "target_eef_position_world",
    "target_eef_orientation_matrix_world",
    "estimated_object_velocity_world",
)


def test_source_frame_schema_preserves_the_complete_recording_contract() -> None:
    """Break caught: a source field is dropped or silently changes dtype/shape."""
    from latency_meta_mdp.data.source.schema import SOURCE_FRAME_SCHEMA

    assert tuple(SOURCE_FRAME_SCHEMA.names) == EXPECTED_FRAME_FIELDS
    assert SOURCE_FRAME_SCHEMA.field("robot_qpos").type == pa.list_(pa.float64(), 7)
    assert SOURCE_FRAME_SCHEMA.field("gripper_qpos").type == pa.list_(pa.float64(), 2)
    assert SOURCE_FRAME_SCHEMA.field("eef_orientation_matrix_world").type == pa.list_(
        pa.float64(), 9
    )
    assert SOURCE_FRAME_SCHEMA.field("expert_action").type == pa.list_(pa.float64())
    assert SOURCE_FRAME_SCHEMA.field("action_mask").type == pa.list_(pa.bool_())
    assert SOURCE_FRAME_SCHEMA.field("applied_reference").type == pa.list_(pa.float64())


def test_source_images_use_embedded_bytes_and_descriptive_paths() -> None:
    """Break caught: source images become external files or lossy video references."""
    from latency_meta_mdp.data.source.schema import SOURCE_FRAME_SCHEMA

    image_type = pa.struct(
        [
            pa.field("bytes", pa.binary(), nullable=False),
            pa.field("path", pa.string(), nullable=False),
        ]
    )
    assert SOURCE_FRAME_SCHEMA.field("agentview_rgb").type == image_type
    assert SOURCE_FRAME_SCHEMA.field("wrist_rgb").type == image_type
    assert SOURCE_FRAME_SCHEMA.field("agentview_rgb").nullable is False
    assert SOURCE_FRAME_SCHEMA.field("wrist_rgb").nullable is False


def test_only_terminal_or_conditionally_absent_fields_are_nullable() -> None:
    """Break caught: missing source state is silently accepted on ordinary boundaries."""
    from latency_meta_mdp.data.source.schema import SOURCE_FRAME_SCHEMA

    nullable = {field.name for field in SOURCE_FRAME_SCHEMA if field.nullable}
    assert nullable == {
        "applied_reference",
        "applied_reference_source_tick",
        "nullspace_joint_position_error",
        "eef_position_error",
        "eef_orientation_error_rotvec",
        "expert_action",
        "action_mask",
        "phase_id",
        "reference_kind",
        "selected_reference_index",
        "target_eef_position_world",
        "target_eef_orientation_matrix_world",
        "estimated_object_velocity_world",
    }


def test_field_roles_keep_privileged_and_audit_state_out_of_deployment_inputs() -> None:
    """Break caught: a generic source reader leaks simulator privilege into a learned model."""
    from latency_meta_mdp.data.source.schema import (
        SourceFieldRole,
        fields_for_role,
    )

    deployment = fields_for_role(SourceFieldRole.DEPLOYMENT_INPUT)
    supervision = fields_for_role(SourceFieldRole.SUPERVISION_CANDIDATE)
    audit = fields_for_role(SourceFieldRole.AUDIT_ONLY)

    assert "agentview_rgb" in deployment
    assert "robot_qpos" in deployment
    assert "expert_action" in deployment
    assert "agentview_rgb" in supervision
    assert "robot_qpos" in supervision
    assert "expert_action" in supervision
    assert "object_pose" in supervision
    assert "object_pose" not in deployment
    assert "commanded_motion_position" in audit
    assert "phase_id" in audit
    assert "actuator_ctrl" in audit
    assert not set(deployment) & {"strategy_family", "commanded_motion_position", "phase_id"}


def test_schema_document_exposes_shape_unit_nullability_and_all_roles() -> None:
    """Break caught: downstream builders cannot audit stored semantics without Python code."""
    from latency_meta_mdp.data.source.schema import (
        SourceFieldRole,
        source_schema_document,
    )

    document = source_schema_document()
    assert document["schema_version"] == 3
    assert document["format_id"] == "structured_expert_source_parquet_v3"
    fields = {row["name"]: row for row in document["source_frame_fields"]}
    assert fields["robot_qpos"] == {
        "name": "robot_qpos",
        "dtype": "float64",
        "shape": [7],
        "unit": "rad",
        "nullable": False,
        "roles": [
            SourceFieldRole.DEPLOYMENT_INPUT.value,
            SourceFieldRole.SUPERVISION_CANDIDATE.value,
        ],
    }
    assert fields["expert_action"]["nullable"] is True
    assert fields["expert_action"]["shape"] == [7]
    assert fields["agentview_rgb"]["dtype"] == "image/png"
    assert fields["agentview_rgb"]["shape"] == [256, 256, 3]


def test_central_metadata_schemas_are_relational_and_success_only() -> None:
    """Break caught: per-episode JSON or failed rows become necessary to locate source data."""
    from latency_meta_mdp.data.source.schema import (
        EPISODE_SCHEMA,
        EVENT_SCHEMA,
        TASK_INSTANCE_SCHEMA,
    )

    assert tuple(TASK_INSTANCE_SCHEMA.names) == (
        "task_instance_id",
        "corpus_id",
        "logical_master_task_index",
        "task_instance_seed",
        "level",
        "instruction",
        "motion_profile_sha256",
        "initial_state_sha256",
        "motion_profile_json",
        "initial_state_npz",
        "admitted_realization_count",
    )
    assert tuple(EPISODE_SCHEMA.names) == (
        "episode_id",
        "task_instance_id",
        "logical_master_task_index",
        "level",
        "accepted_slot",
        "realization_draw_index",
        "realization_seed",
        "strategy_family",
        "strategy_parameters_json",
        "frame_count",
        "terminal_tick",
        "first_contact_time_us",
        "stable_grasp_time_us",
        "handoff_time_us",
        "lift_threshold_time_us",
        "success_time_us",
        "selected_planner_fingerprint",
        "qualification_json",
        "data_shard",
        "row_group_index",
        "row_offset",
        "row_count",
    )
    assert EPISODE_SCHEMA.field("success_time_us").nullable is False
    assert tuple(EVENT_SCHEMA.names) == (
        "episode_id",
        "event_index",
        "kind",
        "physics_step",
        "time_us",
        "payload_json",
        "terminal_reason",
    )


@pytest.mark.parametrize("bad_role", ["deployment", "target", 1, None])
def test_fields_for_role_rejects_untyped_role_requests(bad_role: object) -> None:
    """Break caught: stringly typed role lookup bypasses the model-input allowlist."""
    from latency_meta_mdp.data.source.schema import fields_for_role

    with pytest.raises(TypeError, match="SourceFieldRole"):
        fields_for_role(bad_role)  # type: ignore[arg-type]
