from __future__ import annotations

import hashlib
import json
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64


def make_ids():
    from latency_meta_mdp.expert_realization.contracts import (
        AttemptId,
        ExpertRealizationId,
        ExpertRealizationKey,
        TaskInstanceId,
    )

    task = TaskInstanceId(1, 4000, SHA_A, SHA_B)
    key = ExpertRealizationKey(task, 0, SHA_C)
    realization = ExpertRealizationId(key, SHA_D)
    return task, realization, AttemptId(realization, 0)


def make_episode(*, boundary_count: int = 2):
    from latency_meta_mdp.expert_realization.contracts import StrategyFamily
    from latency_meta_mdp.expert_realization.recording_contracts import (
        ImplementationIdentity,
        StructuredBoundaryRecord,
        StructuredDeploymentRecord,
        StructuredEpisodeMetadata,
        StructuredExpertAuditRecord,
        StructuredPhysicalEventRecord,
        StructuredQualificationRecord,
        StructuredSynchronizedEpisode,
        StructuredTransitionRecord,
    )

    task, realization, attempt = make_ids()
    implementation = ImplementationIdentity("1" * 40, SHA_E, True)
    metadata = StructuredEpisodeMetadata(
        schema_version=1,
        episode_id="episode-l1-s4000-r000-a000",
        task_instance_id=task,
        expert_realization_id=realization,
        attempt_id=attempt,
        task_id="dynamic_grasp_lift",
        instruction="grasp and lift the moving ball",
        physics_dt_us=2_000,
        formal_tick_us=20_000,
        camera_height=2,
        camera_width=3,
        action_contract_id="panda_osc_pose_delta_v1",
        action_dim=7,
        actuator_dim=9,
        expert_id="panda_ball_structured_v1",
        record_profile="pilot_debug",
        strategy_family=StrategyFamily.CANONICAL_DIRECT,
        task_config_sha256=SHA_A,
        motion_config_sha256=SHA_B,
        runtime_config_sha256=SHA_C,
        controller_config_sha256=SHA_D,
        structured_expert_config_sha256=SHA_C,
        curobo_planner_config_sha256=SHA_E,
        pilot_config_sha256=SHA_F,
        pilot_gate_config_sha256="0" * 64,
        task_instance_manifest_sha256="1" * 64,
        frozen_plan_set_manifest_sha256=SHA_D,
        realization_universe_sha256="5" * 64,
        strategy_sha256="2" * 64,
        planner_candidates_sha256="3" * 64,
        selected_reference_sha256="4" * 64,
        implementation=implementation,
    )
    boundaries = []
    for tick in range(boundary_count):
        valid = tick > 0
        deployment = StructuredDeploymentRecord(
            source_physics_step=tick * 10,
            source_formal_tick=tick,
            source_time_us=tick * 20_000,
            agentview_rgb=np.full((2, 3, 3), tick, dtype=np.uint8),
            robot0_eye_in_hand_rgb=np.full((2, 3, 3), tick + 1, dtype=np.uint8),
            robot_qpos=np.arange(7, dtype=np.float64) + tick,
            robot_qvel=np.zeros(7, dtype=np.float64),
            gripper_qpos=np.zeros(2, dtype=np.float64),
            gripper_qvel=np.zeros(2, dtype=np.float64),
            eef_position_world=np.array([0.4, 0.0, 0.3], dtype=np.float64),
            eef_orientation_matrix_world=np.eye(3, dtype=np.float64),
        )
        qualification = StructuredQualificationRecord(
            object_pose=np.array([0.5, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            object_velocity=np.zeros(6, dtype=np.float64),
            commanded_motion_position=np.zeros(3, dtype=np.float64),
            commanded_motion_velocity=np.zeros(3, dtype=np.float64),
            commanded_motion_acceleration=np.zeros(3, dtype=np.float64),
            commanded_motion_segment_index=0,
            left_pad_contact=False,
            right_pad_contact=False,
            handoff_state="driven",
            relative_geometry=np.zeros(3, dtype=np.float64),
            actuator_ctrl=np.zeros(9, dtype=np.float64),
            applied_reference=np.zeros(7, dtype=np.float64) if valid else None,
            applied_reference_source_tick=tick - 1 if valid else None,
            nullspace_joint_position_error=np.zeros(7, dtype=np.float64) if valid else None,
            eef_position_error=np.zeros(3, dtype=np.float64) if valid else None,
            eef_orientation_error_rotvec=np.zeros(3, dtype=np.float64) if valid else None,
        )
        boundaries.append(
            StructuredBoundaryRecord(
                formal_tick_index=tick,
                physics_step_index=tick * 10,
                time_us=tick * 20_000,
                deployment=deployment,
                qualification=qualification,
                outcome_status="success" if tick == boundary_count - 1 else "running",
            )
        )
    transitions = []
    for tick in range(boundary_count - 1):
        audit = StructuredExpertAuditRecord(
            expert_realization_id=realization,
            source_physics_step=tick * 10,
            source_formal_tick=tick,
            source_time_us=tick * 20_000,
            phase_id="approach",
            reference_kind="shared_prefix" if tick == 0 else "selected_reference",
            selected_reference_index=None if tick == 0 else 0,
            target_eef_position_world=np.array([0.5, 0.0, 0.3], dtype=np.float64),
            target_eef_orientation_matrix_world=np.eye(3, dtype=np.float64),
            estimated_object_velocity_world=np.zeros(3, dtype=np.float64),
        )
        transitions.append(
            StructuredTransitionRecord(
                source_formal_tick=tick,
                target_formal_tick=tick + 1,
                expert_action=np.zeros(7, dtype=np.float64),
                action_mask=np.ones(7, dtype=np.bool_),
                expert_audit=audit,
            )
        )
    event = StructuredPhysicalEventRecord(
        kind="success",
        physics_step_index=(boundary_count - 1) * 10,
        time_us=(boundary_count - 1) * 20_000,
        payload={"height_m": 0.2, "contacts": ["left", "right"]},
        terminal_reason=None,
    )
    return StructuredSynchronizedEpisode(
        metadata=metadata,
        boundaries=tuple(boundaries),
        transitions=tuple(transitions),
        physical_events=(event,),
        terminal_status="success",
        terminal_reason="lift_threshold_reached",
    )


def test_structured_episode_requires_all_three_bound_identities_and_exact_nested_joins() -> None:
    """Break caught: an episode can be detached from its task, realization, or retry identity."""
    episode = make_episode()
    task, realization, attempt = make_ids()

    assert episode.metadata.task_instance_id == task
    assert episode.metadata.expert_realization_id == realization
    assert episode.metadata.attempt_id == attempt
    with pytest.raises(ValueError):
        replace(episode.metadata, frozen_plan_set_manifest_sha256=SHA_E)
    with pytest.raises(ValueError):
        replace(
            episode.metadata,
            attempt_id=replace(
                attempt,
                expert_realization_id=replace(realization, task_instance_plan_set_sha256=SHA_E),
            ),
        )


def test_record_arrays_are_exactly_typed_shaped_copied_and_read_only() -> None:
    """Break caught: mutable, coerced, or dimensionally ambiguous values enter the episode."""
    episode = make_episode()
    boundary = episode.boundaries[0]
    transition = episode.transitions[0]

    assert boundary.deployment.agentview_rgb.dtype == np.uint8
    assert boundary.deployment.agentview_rgb.shape == (2, 3, 3)
    assert boundary.deployment.robot_qpos.dtype == np.float64
    assert transition.expert_action.dtype == np.float64
    assert transition.action_mask.dtype == np.bool_
    assert not boundary.deployment.robot_qpos.flags.writeable
    with pytest.raises(ValueError):
        boundary.deployment.robot_qpos[0] = 9.0
    with pytest.raises(ValueError):
        replace(transition, expert_action=np.zeros(6, dtype=np.float64))
    with pytest.raises(ValueError):
        replace(boundary.deployment, agentview_rgb=np.zeros((2, 3, 3), dtype=np.float32))


def test_episode_enforces_contiguous_boundary_and_source_target_action_clocks() -> None:
    """Break caught: a recorded action loses which formal state produced and consumed it."""
    episode = make_episode(boundary_count=3)
    assert [(item.source_formal_tick, item.target_formal_tick) for item in episode.transitions] == [
        (0, 1),
        (1, 2),
    ]
    with pytest.raises(ValueError):
        replace(
            episode,
            transitions=(replace(episode.transitions[0], target_formal_tick=2),)
            + episode.transitions[1:],
        )
    with pytest.raises(ValueError):
        replace(
            episode,
            transitions=(
                replace(
                    episode.transitions[0],
                    expert_audit=replace(episode.transitions[0].expert_audit, source_formal_tick=1),
                ),
            )
            + episode.transitions[1:],
        )


def test_deployment_view_has_exact_public_field_sets_and_no_qualification_or_audit() -> None:
    """Break caught: privileged identity, audit, or controller values leak to policy data."""
    view = make_episode().deployment_view()

    assert set(view) == {
        "task_id",
        "instruction",
        "physics_dt_us",
        "formal_tick_us",
        "camera_height",
        "camera_width",
        "action_contract_id",
        "action_dim",
        "boundaries",
        "transitions",
    }
    assert set(view["boundaries"][0]) == {
        "formal_tick_index",
        "physics_step_index",
        "time_us",
        "agentview_rgb",
        "robot0_eye_in_hand_rgb",
        "robot_qpos",
        "robot_qvel",
        "gripper_qpos",
        "gripper_qvel",
        "eef_position_world",
        "eef_orientation_matrix_world",
        "outcome_status",
    }
    assert set(view["transitions"][0]) == {
        "source_formal_tick",
        "target_formal_tick",
        "expert_action",
        "action_mask",
    }


def test_episode_publication_round_trips_exact_npz_contract_and_rejects_corruption(
    tmp_path: Path,
) -> None:
    """Break caught: bytes omit fields, change dtypes, or load after tampering."""
    from latency_meta_mdp.expert_realization.recording_artifacts import (
        load_verified_structured_episode,
        write_structured_episode,
    )

    episode = make_episode()
    target = tmp_path / "synchronized_episode"
    manifest_path = write_structured_episode(episode, target)
    loaded = load_verified_structured_episode(target)

    assert manifest_path == target / "manifest.json"
    assert loaded.metadata == episode.metadata
    assert loaded.terminal_reason == "lift_threshold_reached"
    assert np.array_equal(loaded.boundaries[0].deployment.agentview_rgb, np.zeros((2, 3, 3)))
    assert not loaded.transitions[0].expert_action.flags.writeable
    with np.load(target / "arrays.npz", allow_pickle=False) as arrays:
        assert set(arrays.files) == {
            "boundary_formal_tick",
            "boundary_physics_step",
            "boundary_time_us",
            "agentview_rgb",
            "robot0_eye_in_hand_rgb",
            "robot_qpos",
            "robot_qvel",
            "gripper_qpos",
            "gripper_qvel",
            "eef_position_world",
            "eef_orientation_matrix_world",
            "boundary_outcome_status_code",
            "object_pose",
            "object_velocity",
            "commanded_motion_position",
            "commanded_motion_velocity",
            "commanded_motion_acceleration",
            "commanded_motion_segment_index",
            "left_pad_contact",
            "right_pad_contact",
            "handoff_state_code",
            "relative_geometry",
            "actuator_ctrl",
            "control_reference_valid",
            "applied_reference",
            "applied_reference_source_tick",
            "nullspace_joint_position_error",
            "eef_position_error",
            "eef_orientation_error_rotvec",
            "transition_source_tick",
            "transition_target_tick",
            "expert_action",
            "action_mask",
            "audit_source_physics_step",
            "audit_source_formal_tick",
            "audit_source_time_us",
            "audit_selected_reference_index",
            "audit_target_eef_position_world",
            "audit_target_eef_orientation_world",
            "audit_estimated_object_velocity",
        }
        assert arrays["boundary_formal_tick"].dtype == np.int64
        assert arrays["boundary_outcome_status_code"].dtype == np.int8
        assert arrays["agentview_rgb"].shape == (2, 2, 3, 3)
        assert arrays["expert_action"].shape == (1, 7)
    metadata_path = target / "metadata.json"
    metadata_path.write_bytes(metadata_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_verified_structured_episode(target)


def test_episode_loader_rejects_legacy_missing_identity_and_hidden_builds(tmp_path: Path) -> None:
    """Break caught: historical or incomplete episodes are mistaken for structured artifacts."""
    from latency_meta_mdp.expert_realization.recording_artifacts import (
        load_verified_structured_episode,
    )

    hidden = tmp_path / ".synchronized_episode.building-dead"
    hidden.mkdir()
    (hidden / "manifest.json").write_text(
        json.dumps({"format_id": "structured_expert_episode_npz_v1"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="hidden"):
        load_verified_structured_episode(hidden)
    old = tmp_path / "old"
    old.mkdir()
    (old / "manifest.json").write_text(
        json.dumps({"format_id": "synchronized_episode_npz_v3"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="format|inventory"):
        load_verified_structured_episode(old)
    missing = tmp_path / "missing"
    missing.mkdir()
    (missing / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "format_id": "structured_expert_episode_npz_v1"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_verified_structured_episode(missing)


def test_episode_writer_never_overwrites_empty_or_nonempty_targets(tmp_path: Path) -> None:
    """Break caught: publication replaces an artifact that another worker already owns."""
    from latency_meta_mdp.expert_realization.recording_artifacts import write_structured_episode

    for name, populated in (("empty", False), ("full", True)):
        target = tmp_path / name
        target.mkdir()
        if populated:
            (target / "owner.txt").write_text("keep", encoding="utf-8")
        with pytest.raises(FileExistsError):
            write_structured_episode(make_episode(), target)
        assert (target / "owner.txt").read_text(encoding="utf-8") == "keep" if populated else True


def test_structured_metadata_serializes_every_bound_identity_and_provenance_field() -> None:
    """Break caught: strategy, artifact, or implementation provenance is dropped."""
    from latency_meta_mdp.expert_realization.recording_contracts import StructuredEpisodeMetadata

    metadata = make_episode().metadata
    mapping = metadata.to_mapping()
    assert set(mapping) == {item.name for item in fields(StructuredEpisodeMetadata)}
    assert mapping["strategy_family"] == "canonical_direct"
    assert mapping["task_instance_id"] == metadata.task_instance_id.to_mapping()
    assert mapping["expert_realization_id"] == metadata.expert_realization_id.to_mapping()
    assert mapping["attempt_id"] == metadata.attempt_id.to_mapping()
    assert mapping["implementation"] == {
        "revision": "1" * 40,
        "source_sha256": SHA_E,
        "dirty": True,
    }


def test_schema_versions_require_exact_integer_one_not_boolean_true() -> None:
    """Break caught: JSON true crosses an integer schema-version trust boundary."""
    with pytest.raises(ValueError, match="schema_version"):
        replace(make_episode().metadata, schema_version=True)


def test_episode_loader_rejects_boolean_schema_versions_and_event_count(tmp_path: Path) -> None:
    """Break caught: boolean envelope versions and counts are accepted as integer one."""
    from latency_meta_mdp.expert_realization.recording_artifacts import (
        load_verified_structured_episode,
        write_structured_episode,
    )

    target = tmp_path / "episode"
    write_structured_episode(make_episode(), target)
    metadata_path = target / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["schema_version"] = True
    metadata["metadata"]["schema_version"] = True
    metadata_path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
    events_path = target / "events.json"
    events = json.loads(events_path.read_text(encoding="utf-8"))
    events["schema_version"] = True
    events_path.write_text(json.dumps(events, sort_keys=True), encoding="utf-8")
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = True
    manifest["event_count"] = True
    manifest["artifacts"]["metadata.json"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    manifest["artifacts"]["events.json"] = hashlib.sha256(events_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="schema|format"):
        load_verified_structured_episode(target)

    count_target = tmp_path / "count-episode"
    write_structured_episode(make_episode(), count_target)
    count_manifest_path = count_target / "manifest.json"
    count_manifest = json.loads(count_manifest_path.read_text(encoding="utf-8"))
    count_manifest["event_count"] = True
    count_manifest_path.write_text(json.dumps(count_manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="count"):
        load_verified_structured_episode(count_target)


def test_zero_transition_episode_round_trips_with_locked_empty_array_shapes(
    tmp_path: Path,
) -> None:
    """Break caught: empty transition arrays collapse their required trailing dimensions."""
    from latency_meta_mdp.expert_realization.recording_artifacts import (
        load_verified_structured_episode,
        write_structured_episode,
    )

    episode = make_episode(boundary_count=1)
    target = tmp_path / "episode"
    write_structured_episode(episode, target)
    loaded = load_verified_structured_episode(target)

    assert len(loaded.boundaries) == 1
    assert loaded.transitions == ()
    with np.load(target / "arrays.npz", allow_pickle=False) as arrays:
        assert arrays["expert_action"].shape == (0, 7)
        assert arrays["action_mask"].shape == (0, 7)
        assert arrays["audit_target_eef_orientation_world"].shape == (0, 3, 3)


def test_episode_loader_rejects_complete_episode_below_hidden_staging_ancestor(
    tmp_path: Path,
) -> None:
    """Break caught: a complete nested child becomes visible before its parent publication."""
    from latency_meta_mdp.expert_realization.recording_artifacts import (
        load_verified_structured_episode,
        write_structured_episode,
    )

    source = tmp_path / "episode"
    write_structured_episode(make_episode(), source)
    hidden = tmp_path / ".attempt.building-dead"
    hidden.mkdir()
    nested = hidden / "synchronized_episode"
    source.rename(nested)

    with pytest.raises(ValueError, match="hidden"):
        load_verified_structured_episode(nested)


def test_episode_loader_requires_exact_regular_nonsymlink_four_file_snapshot(
    tmp_path: Path,
) -> None:
    """Break caught: undeclared files, directories, or symlinks enter a verified episode."""
    from latency_meta_mdp.expert_realization.recording_artifacts import (
        load_verified_structured_episode,
        write_structured_episode,
    )

    extra = tmp_path / "extra"
    write_structured_episode(make_episode(), extra)
    (extra / "vision_feature_cache.json").write_text(
        json.dumps({"format_id": "vision_feature_cache_v1"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="inventory"):
        load_verified_structured_episode(extra)

    child_link = tmp_path / "child-link"
    write_structured_episode(make_episode(), child_link)
    original_events = (child_link / "events.json").read_bytes()
    outside = tmp_path / "outside.json"
    outside.write_bytes(original_events)
    (child_link / "events.json").unlink()
    (child_link / "events.json").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink|regular"):
        load_verified_structured_episode(child_link)

    real = tmp_path / "real"
    write_structured_episode(make_episode(), real)
    root_link = tmp_path / "root-link"
    root_link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink|final"):
        load_verified_structured_episode(root_link)


def test_episode_loader_rejects_intermediate_alias_to_hidden_build_ancestry(
    tmp_path: Path,
) -> None:
    """Break caught: a clean alias exposes an episode physically beneath hidden staging."""
    from latency_meta_mdp.expert_realization.recording_artifacts import (
        load_verified_structured_episode,
        write_structured_episode,
    )

    source = tmp_path / "episode"
    write_structured_episode(make_episode(), source)
    hidden = tmp_path / ".attempt.building-hidden"
    hidden.mkdir()
    physical = hidden / "synchronized_episode"
    source.rename(physical)
    alias = tmp_path / "alias"
    alias.symlink_to(hidden, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink|hidden"):
        load_verified_structured_episode(alias / "synchronized_episode")
