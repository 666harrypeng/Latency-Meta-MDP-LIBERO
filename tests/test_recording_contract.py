from __future__ import annotations

from dataclasses import fields, replace

import numpy as np
import pytest

from latency_meta_mdp.expert import ExpertPhase
from latency_meta_mdp.outcomes import OutcomeStatus, TerminalReason
from latency_meta_mdp.recording import (
    BoundaryRecord,
    CameraRecord,
    CommandedMotionRecord,
    ControlDebugRecord,
    DeploymentRecord,
    EpisodeMetadata,
    ExpertAuditRecord,
    HandoffState,
    PadContactRecord,
    PhysicalEventKind,
    PhysicalEventRecord,
    PrivilegedRecord,
    RecordProfile,
    SynchronizedEpisode,
    TransitionRecord,
)


def _metadata(
    profile: RecordProfile = RecordProfile.PILOT_DEBUG,
    *,
    action_contract_id: str = "panda_osc_pose_delta_v1",
    action_dim: int = 7,
    actuator_dim: int = 9,
) -> EpisodeMetadata:
    return EpisodeMetadata(
        schema_version=1,
        episode_id="l2-seed-000007-attempt-000",
        task_id="dynamic_grasp_lift",
        instruction="Grasp the moving ball and lift it.",
        level=2,
        scene_seed=7,
        motion_seed=7,
        expert_seed=19,
        physics_dt_us=2_000,
        formal_tick_us=20_000,
        action_contract_id=action_contract_id,
        action_dim=action_dim,
        actuator_dim=actuator_dim,
        expert_id="panda_ball_feedback_v1",
        record_profile=profile,
        config_sha256={
            "runtime": "1" * 64,
            "task": "2" * 64,
            "motion": "3" * 64,
            "control": "4" * 64,
            "expert": "5" * 64,
        },
        motion_profile={"schema_version": 1, "type": "cubic_hermite"},
    )


def _camera(name: str, tick: int) -> CameraRecord:
    return CameraRecord(
        name=name,
        source_physics_step=tick * 10,
        source_formal_tick=tick,
        source_time_us=tick * 20_000,
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
    )


def _deployment(tick: int) -> DeploymentRecord:
    return DeploymentRecord(
        images={name: _camera(name, tick) for name in ("agentview", "robot0_eye_in_hand")},
        robot_qpos=np.zeros(7),
        robot_qvel=np.zeros(7),
        gripper_qpos=np.zeros(2),
        gripper_qvel=np.zeros(2),
    )


def _privileged() -> PrivilegedRecord:
    return PrivilegedRecord(
        object_pose=np.zeros(7),
        object_velocity=np.zeros(6),
        commanded_motion=CommandedMotionRecord(
            position=np.zeros(3),
            velocity=np.zeros(3),
            acceleration=np.zeros(3),
            segment_index=0,
        ),
        contact=PadContactRecord(left=False, right=False),
        handoff_state=HandoffState.DRIVEN,
        relative_geometry=np.zeros(3),
    )


def _control_debug(
    tick: int = 0,
    *,
    action_dim: int = 7,
    actuator_dim: int = 9,
) -> ControlDebugRecord:
    return ControlDebugRecord(
        actuator_ctrl=np.zeros(actuator_dim),
        applied_reference=None if tick == 0 else np.zeros(action_dim),
        applied_reference_source_formal_tick=None if tick == 0 else tick - 1,
        nullspace_joint_position_error=None if tick == 0 else np.zeros(7),
        eef_position_error=None if tick == 0 else np.zeros(3),
        eef_orientation_error_rotvec=None if tick == 0 else np.zeros(3),
    )


def _boundary(
    tick: int,
    profile: RecordProfile = RecordProfile.PILOT_DEBUG,
    *,
    outcome_status: OutcomeStatus = OutcomeStatus.RUNNING,
    action_dim: int = 7,
    actuator_dim: int = 9,
) -> BoundaryRecord:
    return BoundaryRecord(
        formal_tick_index=tick,
        physics_step_index=tick * 10,
        time_us=tick * 20_000,
        deployment=_deployment(tick),
        privileged=(
            _privileged()
            if profile in (RecordProfile.BELIEF, RecordProfile.PILOT_DEBUG)
            else None
        ),
        control_debug=(
            _control_debug(
                tick,
                action_dim=action_dim,
                actuator_dim=actuator_dim,
            )
            if profile is RecordProfile.PILOT_DEBUG
            else None
        ),
        outcome_status=outcome_status,
    )


def _transition(tick: int, *, action_dim: int = 7) -> TransitionRecord:
    return TransitionRecord(
        source_formal_tick=tick,
        target_formal_tick=tick + 1,
        expert_action=np.zeros(action_dim),
        action_mask=np.ones(action_dim, dtype=bool),
        expert_audit=ExpertAuditRecord(
            expert_id="panda_ball_feedback_v1",
            source_physics_step=tick * 10,
            source_formal_tick=tick,
            source_time_us=tick * 20_000,
            phase=ExpertPhase.PREGRASP,
            history_start_time_us=max(0, tick - 5) * 20_000,
            history_sample_count=min(tick + 1, 6),
            target_eef_position_world=np.zeros(3),
            estimated_object_velocity_world=np.zeros(3),
        ),
    )


def _success_events() -> tuple[PhysicalEventRecord, ...]:
    return (
        PhysicalEventRecord(PhysicalEventKind.FIRST_CONTACT, 0, 0, {}),
        PhysicalEventRecord(PhysicalEventKind.STABLE_GRASP, 20, 40_000, {}),
        PhysicalEventRecord(PhysicalEventKind.HANDOFF, 20, 40_000, {}),
        PhysicalEventRecord(PhysicalEventKind.LIFT_THRESHOLD, 30, 60_000, {}),
        PhysicalEventRecord(
            PhysicalEventKind.SUCCESS,
            80,
            160_000,
            {},
            terminal_reason=TerminalReason.LIFT_SUCCEEDED,
        ),
    )


def _valid_success_episode() -> SynchronizedEpisode:
    return SynchronizedEpisode(
        metadata=_metadata(),
        boundaries=tuple(
            _boundary(
                tick,
                outcome_status=OutcomeStatus.SUCCESS if tick == 8 else OutcomeStatus.RUNNING,
            )
            for tick in range(9)
        ),
        transitions=tuple(_transition(tick) for tick in range(8)),
        physical_events=_success_events(),
        terminal_status=OutcomeStatus.SUCCESS,
        terminal_reason=TerminalReason.LIFT_SUCCEEDED,
    )


def test_episode_contract_accepts_adjacent_20ms_transitions_and_2ms_events() -> None:
    episode = _valid_success_episode()

    episode.validate_complete()
    assert episode.boundaries[-1].formal_tick_index == 8
    assert episode.physical_events[1].time_us == 40_000


def test_episode_contract_rejects_missing_or_misaligned_rows() -> None:
    metadata = _metadata()
    missing_transition = SynchronizedEpisode(
        metadata=metadata,
        boundaries=(
            _boundary(0),
            _boundary(1, outcome_status=OutcomeStatus.FAILURE),
        ),
        transitions=(),
        physical_events=(
            PhysicalEventRecord(
                PhysicalEventKind.FAILURE,
                10,
                20_000,
                {},
                terminal_reason=TerminalReason.GRASP_DEADLINE_MISSED,
            ),
        ),
        terminal_status=OutcomeStatus.FAILURE,
        terminal_reason=TerminalReason.GRASP_DEADLINE_MISSED,
    )
    with pytest.raises(ValueError, match=r"T transitions for T\+1 boundaries"):
        missing_transition.validate_complete()

    with pytest.raises(ValueError, match="formal clock"):
        BoundaryRecord(
            formal_tick_index=1,
            physics_step_index=10,
            time_us=19_000,
            deployment=_deployment(1),
            privileged=_privileged(),
            control_debug=_control_debug(),
            outcome_status=OutcomeStatus.RUNNING,
        ).validate(metadata)


def test_record_profiles_enforce_privileged_and_debug_boundaries() -> None:
    sft_metadata = _metadata(RecordProfile.SFT)
    _boundary(0, RecordProfile.SFT).validate(sft_metadata)

    with pytest.raises(ValueError, match="privileged record is disabled"):
        _boundary(0, RecordProfile.BELIEF).validate(sft_metadata)

    belief_metadata = _metadata(RecordProfile.BELIEF)
    with pytest.raises(ValueError, match="privileged record is required"):
        _boundary(0, RecordProfile.SFT).validate(belief_metadata)


def test_deployment_view_excludes_all_raw_privileged_and_audit_fields() -> None:
    view = _valid_success_episode().deployment_view()
    view_fields = {field.name for field in fields(view)}

    assert view_fields == {
        "schema_version",
        "task_id",
        "instruction",
        "formal_tick_us",
        "action_contract_id",
        "action_dim",
        "expert_id",
        "boundaries",
        "transitions",
    }
    assert all(boundary.deployment is not None for boundary in view.boundaries)
    assert not hasattr(view, "motion_profile")
    assert not hasattr(view, "physical_events")
    assert not hasattr(view.boundaries[0], "privileged")
    assert not hasattr(view.boundaries[0], "control_debug")
    assert not hasattr(view.boundaries[0], "outcome_status")
    assert not hasattr(view.transitions[0], "expert_audit")


def test_deployment_view_refuses_an_unvalidated_episode() -> None:
    episode = _valid_success_episode()
    invalid = SynchronizedEpisode(
        metadata=episode.metadata,
        boundaries=episode.boundaries,
        transitions=episode.transitions[:-1],
        physical_events=episode.physical_events,
        terminal_status=episode.terminal_status,
        terminal_reason=episode.terminal_reason,
    )

    with pytest.raises(ValueError, match=r"T transitions for T\+1 boundaries"):
        invalid.deployment_view()


def test_records_deep_copy_and_freeze_mutable_input() -> None:
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    camera = CameraRecord("agentview", 0, 0, 0, rgb)
    rgb[:] = 255
    assert np.all(camera.rgb == 0)
    assert camera.rgb.flags.writeable is False

    profile = {"schema_version": 1, "type": "constant_velocity"}
    metadata = EpisodeMetadata(
        schema_version=1,
        episode_id="episode",
        task_id="dynamic_grasp_lift",
        instruction="Grasp the moving ball and lift it.",
        level=1,
        scene_seed=1,
        motion_seed=2,
        expert_seed=3,
        physics_dt_us=2_000,
        formal_tick_us=20_000,
        action_contract_id="panda_osc_pose_delta_v1",
        action_dim=7,
        actuator_dim=9,
        expert_id="panda_ball_feedback_v1",
        record_profile=RecordProfile.SFT,
        config_sha256={
            name: str(index) * 64
            for index, name in enumerate(
                ("runtime", "task", "motion", "control", "expert"), start=1
            )
        },
        motion_profile=profile,
    )
    profile["future_object_pose"] = [1, 2, 3]
    assert "future_object_pose" not in metadata.motion_profile
    with pytest.raises(TypeError):
        metadata.motion_profile["future_object_pose"] = [1, 2, 3]

    payload = {"contact_impulse": np.zeros(3)}
    event = PhysicalEventRecord(PhysicalEventKind.FIRST_CONTACT, 0, 0, payload)
    payload["oracle_future"] = np.ones(3)
    assert "oracle_future" not in event.payload
    assert event.payload["contact_impulse"].flags.writeable is False


def test_physical_events_must_use_the_2ms_grid_and_episode_bounds() -> None:
    event = PhysicalEventRecord(
        PhysicalEventKind.FIRST_CONTACT,
        physics_step_index=1,
        time_us=2_001,
        payload={},
    )

    with pytest.raises(ValueError, match="physics clock"):
        event.validate(_metadata(), final_time_us=20_000)

    off_grid_handoff = PhysicalEventRecord(
        PhysicalEventKind.HANDOFF,
        physics_step_index=7,
        time_us=14_000,
        payload={},
    )
    with pytest.raises(ValueError, match="formal clock"):
        off_grid_handoff.validate(_metadata(), final_time_us=20_000)


def test_typed_payloads_reject_invalid_camera_and_state_shapes() -> None:
    with pytest.raises(ValueError, match="RGB uint8 HxWx3"):
        CameraRecord("agentview", 0, 0, 0, np.zeros((2, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match="robot_qpos"):
        DeploymentRecord(
            images={name: _camera(name, 0) for name in ("agentview", "robot0_eye_in_hand")},
            robot_qpos=np.zeros(6),
            robot_qvel=np.zeros(7),
            gripper_qpos=np.zeros(2),
            gripper_qvel=np.zeros(2),
        )
    with pytest.raises(ValueError, match="object_pose"):
        PrivilegedRecord(
            object_pose=np.full(7, np.nan),
            object_velocity=np.zeros(6),
            commanded_motion=CommandedMotionRecord(
                np.zeros(3), np.zeros(3), np.zeros(3), 0
            ),
            contact=PadContactRecord(False, False),
            handoff_state=HandoffState.DRIVEN,
            relative_geometry=np.zeros(3),
        )
    metadata = _metadata()
    invalid_debug = ControlDebugRecord(
        actuator_ctrl=np.zeros(9),
        applied_reference=np.zeros(6),
        applied_reference_source_formal_tick=0,
        nullspace_joint_position_error=np.zeros(7),
        eef_position_error=np.zeros(3),
        eef_orientation_error_rotvec=np.zeros(3),
    )
    with pytest.raises(ValueError, match="applied_reference"):
        BoundaryRecord(
            formal_tick_index=1,
            physics_step_index=10,
            time_us=20_000,
            deployment=_deployment(1),
            privileged=_privileged(),
            control_debug=invalid_debug,
            outcome_status=OutcomeStatus.RUNNING,
        ).validate(metadata)


def test_control_debug_identifies_the_previous_action_and_empty_initial_boundary() -> None:
    metadata = _metadata()
    initial = _boundary(0)
    after_action = _boundary(1)

    initial.validate(metadata)
    after_action.validate(metadata)
    assert initial.control_debug is not None
    assert initial.control_debug.applied_reference is None
    assert initial.control_debug.applied_reference_source_formal_tick is None
    assert initial.control_debug.nullspace_joint_position_error is None
    assert initial.control_debug.eef_position_error is None
    assert initial.control_debug.eef_orientation_error_rotvec is None
    assert after_action.control_debug is not None
    assert after_action.control_debug.applied_reference_source_formal_tick == 0
    assert after_action.control_debug.nullspace_joint_position_error is not None
    assert after_action.control_debug.eef_position_error is not None
    assert after_action.control_debug.eef_orientation_error_rotvec is not None

    invalid_initial = replace(
        initial,
        control_debug=replace(
            initial.control_debug,
            applied_reference=np.zeros(7),
            applied_reference_source_formal_tick=0,
            nullspace_joint_position_error=np.zeros(7),
            eef_position_error=np.zeros(3),
            eef_orientation_error_rotvec=np.zeros(3),
        ),
    )
    with pytest.raises(ValueError, match="initial boundary"):
        invalid_initial.validate(metadata)

    invalid_source = replace(
        after_action,
        control_debug=replace(
            after_action.control_debug,
            applied_reference_source_formal_tick=1,
        ),
    )
    with pytest.raises(ValueError, match="previous formal tick"):
        invalid_source.validate(metadata)


def test_episode_metadata_rejects_a_mixed_controller_contract() -> None:
    with pytest.raises(ValueError, match="selected Panda action contract"):
        _metadata(
            action_contract_id="panda_joint_absolute_v1",
            action_dim=8,
            actuator_dim=9,
        )

    metadata = _metadata()
    wrong_action = _transition(0, action_dim=8)
    with pytest.raises(ValueError, match="action_dim"):
        wrong_action.validate(metadata)


def test_transition_rejects_misaligned_expert_audit_source_time() -> None:
    metadata = _metadata()
    transition = _transition(0)
    invalid = replace(
        transition,
        expert_audit=replace(
            transition.expert_audit,
            source_physics_step=10,
            source_formal_tick=1,
            source_time_us=20_000,
            history_start_time_us=20_000,
        ),
    )

    with pytest.raises(ValueError, match="expert audit source"):
        invalid.validate(metadata)


def test_complete_episode_rejects_impossible_success_event_order() -> None:
    episode = _valid_success_episode()
    invalid = SynchronizedEpisode(
        metadata=episode.metadata,
        boundaries=episode.boundaries,
        transitions=episode.transitions,
        physical_events=(
            PhysicalEventRecord(PhysicalEventKind.FIRST_CONTACT, 0, 0, {}),
            PhysicalEventRecord(
                PhysicalEventKind.SUCCESS,
                80,
                160_000,
                {},
                terminal_reason=TerminalReason.LIFT_SUCCEEDED,
            ),
        ),
        terminal_status=OutcomeStatus.SUCCESS,
        terminal_reason=TerminalReason.LIFT_SUCCEEDED,
    )

    with pytest.raises(ValueError, match="success milestones"):
        invalid.validate_complete()


def test_complete_episode_rejects_terminal_status_before_final_boundary() -> None:
    episode = _valid_success_episode()
    boundaries = list(episode.boundaries)
    boundaries[3] = _boundary(3, outcome_status=OutcomeStatus.SUCCESS)
    invalid = SynchronizedEpisode(
        metadata=episode.metadata,
        boundaries=tuple(boundaries),
        transitions=episode.transitions,
        physical_events=episode.physical_events,
        terminal_status=episode.terminal_status,
        terminal_reason=episode.terminal_reason,
    )

    with pytest.raises(ValueError, match="non-final boundaries"):
        invalid.validate_complete()


def test_complete_episode_rejects_opposite_terminal_event() -> None:
    episode = _valid_success_episode()
    invalid = SynchronizedEpisode(
        metadata=episode.metadata,
        boundaries=episode.boundaries,
        transitions=episode.transitions,
        physical_events=episode.physical_events[:-1]
        + (
            PhysicalEventRecord(
                PhysicalEventKind.FAILURE,
                80,
                160_000,
                {},
                terminal_reason=TerminalReason.LIFT_DEADLINE_MISSED,
            ),
        ),
        terminal_status=OutcomeStatus.SUCCESS,
        terminal_reason=TerminalReason.LIFT_SUCCEEDED,
    )

    with pytest.raises(ValueError, match="opposite terminal event"):
        invalid.validate_complete()


def test_failure_event_reason_must_match_episode_reason() -> None:
    metadata = _metadata()
    episode = SynchronizedEpisode(
        metadata=metadata,
        boundaries=(
            _boundary(0),
            _boundary(1, outcome_status=OutcomeStatus.FAILURE),
        ),
        transitions=(_transition(0),),
        physical_events=(
            PhysicalEventRecord(
                PhysicalEventKind.FAILURE,
                10,
                20_000,
                {},
                terminal_reason=TerminalReason.GRASP_DEADLINE_MISSED,
            ),
        ),
        terminal_status=OutcomeStatus.FAILURE,
        terminal_reason=TerminalReason.LIFT_DEADLINE_MISSED,
    )

    with pytest.raises(ValueError, match="terminal reason"):
        episode.validate_complete()


def test_episode_container_copies_sequence_inputs() -> None:
    valid = _valid_success_episode()
    boundaries = list(valid.boundaries)
    transitions = list(valid.transitions)
    events = list(valid.physical_events)
    episode = SynchronizedEpisode(
        metadata=valid.metadata,
        boundaries=boundaries,
        transitions=transitions,
        physical_events=events,
        terminal_status=valid.terminal_status,
        terminal_reason=valid.terminal_reason,
    )
    boundaries.clear()
    transitions.clear()
    events.clear()

    episode.validate_complete()
    assert len(episode.boundaries) == 9
    assert isinstance(episode.boundaries, tuple)
