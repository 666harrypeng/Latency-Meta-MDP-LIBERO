from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from expert_realization_test_support import make_formal_source_metadata


def _snapshot(tick: int):
    from latency_meta_mdp.envs.snapshots import BoundarySnapshot, CameraSample

    cameras = {
        name: CameraSample(
            name=name,
            rgb=np.full((2, 3, 3), tick + offset, dtype=np.uint8),
            segmentation=None,
            source_physics_step=tick * 10,
            source_formal_tick=tick,
            source_time_us=tick * 20_000,
        )
        for offset, name in enumerate(("agentview", "robot0_eye_in_hand"))
    }
    return BoundarySnapshot(
        physics_step_index=tick * 10,
        formal_tick_index=tick,
        time_us=tick * 20_000,
        sim_time_seconds=tick * 0.02,
        qpos=np.zeros(16, dtype=np.float64),
        qvel=np.zeros(15, dtype=np.float64),
        act=np.empty(0, dtype=np.float64),
        actuator_ctrl=np.arange(9, dtype=np.float64),
        robot_qpos=np.arange(7, dtype=np.float64) + tick,
        robot_qvel=np.full(7, tick, dtype=np.float64),
        robot_gripper_qpos=np.array([0.02, -0.02], dtype=np.float64),
        robot_gripper_qvel=np.zeros(2, dtype=np.float64),
        eef_pos=np.array([0.4, 0.0, 0.3], dtype=np.float64),
        eef_xmat=np.eye(3, dtype=np.float64),
        object_qpos=np.array([0.5, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        object_qvel=np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        object_body_pos=np.array([0.5, 0.0, 0.2], dtype=np.float64),
        object_body_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        commanded_world={
            "target_position": np.array([0.5, 0.0, 0.2], dtype=np.float64),
            "target_velocity": np.array([0.1, 0.0, 0.0], dtype=np.float64),
            "target_acceleration": np.zeros(3, dtype=np.float64),
            "segment_index": np.array(0, dtype=np.int64),
            "contact_left": np.array(tick == 1, dtype=np.bool_),
            "contact_right": np.array(tick == 1, dtype=np.bool_),
            "handoff_state": np.array("physical" if tick == 1 else "driven"),
        },
        cameras=cameras,
    )


def _runtime():
    arm = SimpleNamespace(
        origin_pos=np.zeros(3, dtype=np.float64),
        origin_ori=np.eye(3, dtype=np.float64),
        goal_pos=np.array([0.4, 0.0, 0.3], dtype=np.float64),
        goal_ori=np.eye(3, dtype=np.float64),
        initial_joint=np.arange(7, dtype=np.float64),
    )
    robot = SimpleNamespace(part_controllers={"right": arm})
    return SimpleNamespace(env=SimpleNamespace(robots=[robot]))


def _passing_safety_report():
    from latency_meta_mdp.data.collection.config import load_pilot_gate_config
    from latency_meta_mdp.data.collection.safety import ActualRolloutSafetyReport

    gate = load_pilot_gate_config(Path("configs/analysis/panda_ball_structured_pilot_gate.yaml"))
    return ActualRolloutSafetyReport(
        terminal_success=True,
        physical_handoff=True,
        phase_order_valid=True,
        minimum_non_contact_environment_clearance_m=gate.non_contact_environment_clearance_m,
        maximum_intentional_contact_penetration_m=0.1,
        maximum_pad_ball_impulse_ns=gate.peak_pad_ball_impulse_per_physics_contact_event_ns,
        unintended_pregrasp_ball_contacts=0,
        other_link_ball_contacts=0,
        robot_environment_contacts=0,
        robot_self_contacts=0,
        minimum_joint_position_margin_rad=gate.joint_position_margin_rad,
        maximum_joint_velocity_fraction=gate.joint_velocity_fraction_of_model_limit,
        maximum_eef_speed_mps=gate.eef_speed_mps,
        maximum_eef_acceleration_mps2=gate.eef_acceleration_mps2,
        maximum_eef_jerk_mps3=1_000.0,
        maximum_reference_tracking_error_m=10.0,
        maximum_pregrasp_translation_error_m=1.0,
        maximum_pregrasp_rotation_error_degrees=180.0,
        minimum_osc_action=-1.0,
        maximum_osc_action=1.0,
        pre_handoff_saturation_fraction=gate.pre_handoff_saturation_fraction,
    )


def _successful_episode():
    from latency_meta_mdp.data.source.recording import SourceEpisodeRecorder
    from latency_meta_mdp.envs.outcomes import OutcomeEvent, TerminalReason

    metadata = make_formal_source_metadata()
    recorder = SourceEpisodeRecorder(metadata)
    first, second = _snapshot(0), _snapshot(1)
    action = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float64)
    recorder.append_boundary(
        first, runtime=_runtime(), previous_action=None, outcome_status="running"
    )
    recorder.append_shared_transition(first, action)
    recorder.append_boundary(
        second, runtime=_runtime(), previous_action=action, outcome_status="success"
    )
    return recorder.build_success(
        terminal_reason="lift_succeeded",
        outcome_events=(OutcomeEvent("success", 20_000, TerminalReason.LIFT_SUCCEEDED),),
        handoff_release_qpos=None,
        handoff_release_qvel=None,
    )


def test_source_episode_recorder_preserves_boundary_transition_and_event_alignment() -> None:
    """Break caught: the full source recorder shifts actions or drops physical state."""
    from latency_meta_mdp.data.source.recording import SourceEpisodeRecorder
    from latency_meta_mdp.envs.outcomes import OutcomeEvent, TerminalReason

    metadata = make_formal_source_metadata()
    recorder = SourceEpisodeRecorder(metadata)
    first = _snapshot(0)
    second = _snapshot(1)
    action = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float64)

    recorder.append_boundary(
        first, runtime=_runtime(), previous_action=None, outcome_status="running"
    )
    recorder.append_shared_transition(first, action)
    recorder.append_boundary(
        second, runtime=_runtime(), previous_action=action, outcome_status="success"
    )
    episode = recorder.build_success(
        terminal_reason="lift_succeeded",
        outcome_events=(OutcomeEvent("success", 20_000, TerminalReason.LIFT_SUCCEEDED),),
        handoff_release_qpos=None,
        handoff_release_qvel=None,
    )

    assert len(episode.boundaries) == 2
    assert len(episode.transitions) == 1
    assert np.array_equal(episode.transitions[0].expert_action, action)
    assert episode.transitions[0].expert_audit.reference_kind == "shared_prefix"
    assert episode.boundaries[1].qualification.left_pad_contact is True
    assert episode.boundaries[1].qualification.handoff_state == "physical"
    assert np.array_equal(
        episode.boundaries[1].qualification.commanded_motion_velocity,
        [0.1, 0.0, 0.0],
    )
    assert episode.physical_events[0].time_us == 20_000
    assert episode.physical_events[0].terminal_reason == "lift_succeeded"


def test_source_episode_recorder_uses_post_prefix_decision_audit() -> None:
    """Break caught: post-prefix targets are mislabeled as shared-prefix actions."""
    from latency_meta_mdp.data.collection.executor import (
        StructuredExpertDecision,
        StructuredExpertPhase,
    )
    from latency_meta_mdp.data.source.recording import SourceEpisodeRecorder

    recorder = SourceEpisodeRecorder(make_formal_source_metadata())
    recorder.append_boundary(
        _snapshot(0), runtime=_runtime(), previous_action=None, outcome_status="running"
    )
    decision = StructuredExpertDecision(
        source_physics_step=0,
        source_formal_tick=0,
        source_time_us=0,
        phase=StructuredExpertPhase.SMOOTH_APPROACH,
        reference_kind="selected_reference",
        selected_reference_index=1,
        target_eef_position_world=np.array([0.5, 0.1, 0.3], dtype=np.float64),
        target_eef_orientation_world=np.eye(3, dtype=np.float64),
        estimated_object_velocity_world=np.array([0.1, 0.0, 0.0], dtype=np.float64),
        action=np.array([0.1, 0, 0, 0, 0, 0, -1], dtype=np.float64),
        position_error_base=np.array([0.01, 0.0, 0.0], dtype=np.float64),
        orientation_error_rotvec_base=np.zeros(3, dtype=np.float64),
        saturated=False,
    )

    recorder.append_decision_transition(decision)

    transition = recorder.transitions[0]
    assert transition.expert_audit.phase_id == "smooth_approach"
    assert transition.expert_audit.selected_reference_index == 1
    assert np.array_equal(transition.expert_audit.target_eef_position_world, [0.5, 0.1, 0.3])


def test_source_recorder_rejects_noncontiguous_appends_and_non_success_build() -> None:
    """Break caught: partial or failed traces are materialized as source episodes."""
    from latency_meta_mdp.data.source.recording import SourceEpisodeRecorder

    recorder = SourceEpisodeRecorder(make_formal_source_metadata())
    with pytest.raises(ValueError, match="next boundary"):
        recorder.append_boundary(
            _snapshot(1), runtime=_runtime(), previous_action=None, outcome_status="running"
        )
    recorder.append_boundary(
        _snapshot(0), runtime=_runtime(), previous_action=None, outcome_status="running"
    )
    with pytest.raises(ValueError, match="complete successful trace"):
        recorder.build_success(
            terminal_reason="lift_succeeded",
            outcome_events=(),
            handoff_release_qpos=None,
            handoff_release_qvel=None,
        )


def test_formal_source_execution_returns_episode_beside_physical_qualification(
    monkeypatch,
) -> None:
    """Break caught: terminal success is published without the actual-physics gate."""
    import latency_meta_mdp.data.collection.rollout as rollout_module
    from latency_meta_mdp.data.collection.config import load_pilot_gate_config
    from latency_meta_mdp.data.collection.robot_bridge import PandaPlanningBridge
    from latency_meta_mdp.data.source.recording import (
        QualifiedSourceRecording,
        execute_structured_source_recording,
    )

    metadata = make_formal_source_metadata()
    episode = _successful_episode()
    report = _passing_safety_report()
    monkeypatch.setattr(
        rollout_module,
        "_execute_structured_realization",
        lambda **_kwargs: (SimpleNamespace(terminal_status="success"), episode, report),
    )

    result = execute_structured_source_recording(
        task_instance=object(),
        intent=object(),
        reference=object(),
        metadata=metadata,
        maximum_formal_ticks=10,
        planning_bridge=object.__new__(PandaPlanningBridge),
        gate=load_pilot_gate_config(Path("configs/analysis/panda_ball_structured_pilot_gate.yaml")),
    )

    assert isinstance(result, QualifiedSourceRecording)
    assert result.episode is episode
    assert result.safety_report is report
    assert result.qualification.eligible is True
    assert result.qualification_mapping()["eligible"] is True


def test_formal_source_execution_rejects_a_task_success_that_fails_physics(
    monkeypatch,
) -> None:
    """Break caught: a successful payload can bypass the source qualification decision."""
    import latency_meta_mdp.data.collection.rollout as rollout_module
    from latency_meta_mdp.data.collection.config import load_pilot_gate_config
    from latency_meta_mdp.data.collection.robot_bridge import PandaPlanningBridge
    from latency_meta_mdp.data.source.recording import (
        SourceQualificationFailure,
        execute_structured_source_recording,
    )

    report = replace(_passing_safety_report(), robot_environment_contacts=1)
    episode = _successful_episode()
    monkeypatch.setattr(
        rollout_module,
        "_execute_structured_realization",
        lambda **_kwargs: (
            SimpleNamespace(terminal_status="success"),
            episode,
            report,
        ),
    )
    with pytest.raises(SourceQualificationFailure, match="robot_environment_contact"):
        execute_structured_source_recording(
            task_instance=object(),
            intent=object(),
            reference=object(),
            metadata=make_formal_source_metadata(),
            maximum_formal_ticks=10,
            planning_bridge=object.__new__(PandaPlanningBridge),
            gate=load_pilot_gate_config(
                Path("configs/analysis/panda_ball_structured_pilot_gate.yaml")
            ),
        )
