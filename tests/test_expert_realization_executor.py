from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from latency_meta_mdp.control import load_action_contract


def _rotation_z(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _contract():
    return load_action_contract(Path("configs/control/panda_osc_pose_delta_v1.yaml"))


@pytest.fixture(scope="module")
def structured_case():
    from latency_meta_mdp.expert_realization.contracts import (
        ExpertRealizationKey,
        StrategyFamily,
    )
    from latency_meta_mdp.expert_realization.selector import SelectedReference
    from latency_meta_mdp.expert_realization.strategy import (
        StructuredStrategyConfig,
        sample_strategy,
    )
    from latency_meta_mdp.expert_realization.task_instance import materialize_task_instance
    from latency_meta_mdp.expert_realization.trajectory_intent import build_trajectory_intent

    root = Path.cwd()
    task = materialize_task_instance(project_root=root, level=1, task_instance_seed=4000)
    config = StructuredStrategyConfig.from_path(
        root / "configs/expert_realization/panda_ball_structured.yaml"
    )
    key = ExpertRealizationKey(task.task_instance_id, 0, config.source_sha256)
    strategy = sample_strategy(
        task,
        key,
        config,
        assigned_family=StrategyFamily.CANONICAL_DIRECT,
    )
    intent = build_trajectory_intent(task, task.expected_anchor, strategy)
    duration_ticks = intent.approach.funnel_entry_target_tick - task.decision_source_tick
    timestamps = np.arange(duration_ticks + 1, dtype=np.float64) * 0.02
    progress = np.linspace(0.0, 1.0, duration_ticks + 1, dtype=np.float64)
    eef = (1.0 - progress[:, None]) * intent.approach.start_position_world + progress[
        :, None
    ] * intent.approach.funnel_entry_position_world
    terminal_step = 0.005 * intent.approach.funnel_entry_tangent_world
    eef[-2] = eef[-1] - terminal_step
    reference = SelectedReference(
        expert_realization_key=key,
        source_candidate_fingerprint="d" * 64,
        qpos_path=np.repeat(
            task.expected_anchor.anchor_robot_qpos[None], duration_ticks + 1, axis=0
        ),
        timestamps_seconds=timestamps,
        eef_positions_world=eef,
        fixed_orientation_world=intent.approach.fixed_orientation_world,
        fingerprint="e" * 64,
    )
    return task, intent, reference


def _snapshot(*, tick: int, object_position: np.ndarray, eef_position: np.ndarray):
    from latency_meta_mdp.snapshots import BoundarySnapshot

    zeros = np.zeros
    return BoundarySnapshot(
        physics_step_index=10 * tick,
        formal_tick_index=tick,
        time_us=20_000 * tick,
        sim_time_seconds=0.02 * tick,
        qpos=zeros(16, dtype=np.float64),
        qvel=zeros(15, dtype=np.float64),
        act=zeros(9, dtype=np.float64),
        actuator_ctrl=zeros(9, dtype=np.float64),
        robot_qpos=zeros(7, dtype=np.float64),
        robot_qvel=zeros(7, dtype=np.float64),
        robot_gripper_qpos=zeros(2, dtype=np.float64),
        robot_gripper_qvel=zeros(2, dtype=np.float64),
        eef_pos=np.asarray(eef_position, dtype=np.float64),
        eef_xmat=np.eye(3, dtype=np.float64),
        object_qpos=zeros(7, dtype=np.float64),
        object_qvel=zeros(6, dtype=np.float64),
        object_body_pos=np.asarray(object_position, dtype=np.float64),
        object_body_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        commanded_world=MappingProxyType({}),
        cameras=MappingProxyType({}),
    )


def _object_at(intent, tick: int) -> np.ndarray:
    return intent.capture_position_world - intent.capture_velocity_world * (
        (intent.grasp_funnel.close_target_tick - tick) * 0.02
    )


def _executor(case):
    from latency_meta_mdp.expert_realization.executor import StructuredExpertExecutor

    task, intent, reference = case
    return StructuredExpertExecutor(
        action_contract=_contract(),
        intent=intent,
        reference=reference,
        decision_source_tick=task.decision_source_tick,
        world_to_base_rotation=np.eye(3, dtype=np.float64),
    )


def test_zero_pose_error_produces_zero_arm_action() -> None:
    """Break caught: converting an already-achieved reference moves the arm."""
    from latency_meta_mdp.expert_realization.executor import osc_action_from_reference

    result = osc_action_from_reference(
        achieved_position_world=np.array([0.5, 0.0, 1.0], dtype=np.float64),
        achieved_orientation_world=np.eye(3, dtype=np.float64),
        target_position_world=np.array([0.5, 0.0, 1.0], dtype=np.float64),
        target_orientation_world=np.eye(3, dtype=np.float64),
        world_to_base_rotation=np.eye(3, dtype=np.float64),
        action_contract=_contract(),
        gripper_command=-1.0,
        translation_clip_m=0.03,
    )

    np.testing.assert_allclose(result.action, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    assert result.saturated is False


def test_world_translation_and_rotation_use_controller_base_axes() -> None:
    """Break caught: a world-frame pose error is sent on the wrong controller axes."""
    from latency_meta_mdp.expert_realization.executor import osc_action_from_reference

    world_to_base = np.array(
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    result = osc_action_from_reference(
        achieved_position_world=np.zeros(3, dtype=np.float64),
        achieved_orientation_world=_rotation_z(np.deg2rad(179.0)),
        target_position_world=np.array([0.05, 0.0, 0.0], dtype=np.float64),
        target_orientation_world=_rotation_z(np.deg2rad(-179.0)),
        world_to_base_rotation=world_to_base,
        action_contract=_contract(),
        gripper_command=1.0,
        translation_clip_m=0.05,
    )

    np.testing.assert_allclose(result.position_error_base, [0.0, -0.05, 0.0], atol=1e-15)
    assert result.action[-1] == 1.0


def test_source_tick_five_targets_reference_index_one(structured_case) -> None:
    """Break caught: u[5] repeats S[5] instead of advancing the smooth reference."""
    from latency_meta_mdp.expert_realization.executor import StructuredExpertPhase
    from latency_meta_mdp.handoff import HandoffState

    task, intent, reference = structured_case
    decision = _executor(structured_case).next_action(
        snapshot=_snapshot(
            tick=5,
            object_position=_object_at(intent, 5),
            eef_position=reference.eef_positions_world[0],
        ),
        handoff_state=HandoffState.DRIVEN,
        left_pad_contact=False,
        right_pad_contact=False,
    )

    assert decision.phase is StructuredExpertPhase.SMOOTH_APPROACH
    assert decision.selected_reference_index == 1
    np.testing.assert_array_equal(
        decision.target_eef_position_world, reference.eef_positions_world[1]
    )
    assert decision.action[-1] == -1.0
    assert task.decision_source_tick == 5


def test_funnel_is_tangent_continuous_then_close_and_lift_are_event_gated(
    structured_case,
) -> None:
    """Break caught: approach, descent, close, and lift are disconnected clock-driven primitives."""
    from latency_meta_mdp.expert_realization.executor import StructuredExpertPhase
    from latency_meta_mdp.handoff import HandoffState

    _task, intent, reference = structured_case
    executor = _executor(structured_case)
    decisions = {}
    eef = np.array(reference.eef_positions_world[0], copy=True)
    close_tick = None
    for tick in range(5, intent.grasp_funnel.handoff_deadline_tick):
        physical = close_tick is not None and tick >= close_tick + 1
        nominal_descent_end = (
            intent.approach.funnel_entry_target_tick + intent.strategy.funnel_descent_ticks
        )
        if close_tick is None and tick >= nominal_descent_end:
            eef = _object_at(intent, tick)
        decisions[tick] = executor.next_action(
            snapshot=_snapshot(
                tick=tick,
                object_position=_object_at(intent, tick),
                eef_position=eef,
            ),
            handoff_state=HandoffState.PHYSICAL if physical else HandoffState.DRIVEN,
            left_pad_contact=close_tick is not None,
            right_pad_contact=close_tick is not None,
        )
        eef = np.array(decisions[tick].target_eef_position_world, copy=True)
        if decisions[tick].phase is StructuredExpertPhase.CLOSE_STABILIZE and close_tick is None:
            close_tick = tick
        if decisions[tick].phase is StructuredExpertPhase.LIFT:
            break

    assert close_tick is not None
    entry_tick = intent.approach.funnel_entry_target_tick
    before = decisions[entry_tick - 1].target_eef_position_world
    entry = decisions[entry_tick].target_eef_position_world
    after = decisions[entry_tick + 1].target_eef_position_world
    before_direction = (entry - before) / np.linalg.norm(entry - before)
    after_direction = (after - entry) / np.linalg.norm(after - entry)
    assert float(np.dot(before_direction, after_direction)) > 0.8
    assert decisions[entry_tick].phase is StructuredExpertPhase.GRASP_FUNNEL
    assert decisions[close_tick].phase is StructuredExpertPhase.CLOSE_STABILIZE
    assert decisions[close_tick].action[-1] == 1.0
    close_offset = (
        decisions[close_tick].target_eef_position_world
        - _object_at(intent, close_tick)
    )
    follow_offset = (
        decisions[close_tick + 1].target_eef_position_world
        - _object_at(intent, close_tick + 1)
    )
    np.testing.assert_allclose(
        close_offset,
        intent.capture_velocity_world * intent.strategy.prediction_lead_seconds,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        follow_offset,
        intent.capture_velocity_world * intent.strategy.prediction_lead_seconds,
        atol=1.0e-12,
    )
    lift = decisions[close_tick + intent.grasp_funnel.close_dwell_ticks]
    assert lift.phase is StructuredExpertPhase.LIFT
    assert lift.action[-1] == 1.0


def test_unilateral_contact_must_become_bilateral_inside_acquisition_window(
    structured_case,
) -> None:
    """Break caught: a one-pad push is admitted as a valid grasp realization."""
    from latency_meta_mdp.expert_realization.executor import SemanticExecutionFailure
    from latency_meta_mdp.handoff import HandoffState

    _task, intent, reference = structured_case
    executor = _executor(structured_case)
    eef = np.array(reference.eef_positions_world[0], copy=True)
    close_tick = intent.grasp_funnel.close_target_tick
    for tick in range(5, close_tick + intent.grasp_funnel.bilateral_contact_acquisition_ticks):
        decision = executor.next_action(
            snapshot=_snapshot(
                tick=tick,
                object_position=_object_at(intent, tick),
                eef_position=eef,
            ),
            handoff_state=HandoffState.DRIVEN,
            left_pad_contact=tick >= close_tick,
            right_pad_contact=False,
        )
        eef = np.array(decision.target_eef_position_world, copy=True)

    with pytest.raises(SemanticExecutionFailure, match="bilateral"):
        executor.next_action(
            snapshot=_snapshot(
                tick=close_tick + intent.grasp_funnel.bilateral_contact_acquisition_ticks,
                object_position=_object_at(
                    intent,
                    close_tick + intent.grasp_funnel.bilateral_contact_acquisition_ticks,
                ),
                eef_position=eef,
            ),
            handoff_state=HandoffState.DRIVEN,
            left_pad_contact=True,
            right_pad_contact=False,
        )


def test_post_target_funnel_tracks_current_object_until_geometry_is_ready(
    structured_case,
) -> None:
    """Break caught: a late arm keeps chasing the stale nominal capture point."""
    from latency_meta_mdp.expert_realization.executor import StructuredExpertPhase
    from latency_meta_mdp.handoff import HandoffState

    _task, intent, reference = structured_case
    executor = _executor(structured_case)
    eef = np.array(reference.eef_positions_world[0], copy=True)
    close_tick = intent.grasp_funnel.close_target_tick
    decisions = {}
    for tick in range(5, close_tick + 2):
        object_position = _object_at(intent, tick)
        decisions[tick] = executor.next_action(
            snapshot=_snapshot(
                tick=tick,
                object_position=object_position,
                eef_position=eef,
            ),
            handoff_state=HandoffState.DRIVEN,
            left_pad_contact=False,
            right_pad_contact=False,
        )
        if tick < intent.grasp_funnel.close_earliest_tick - 1:
            eef = np.array(decisions[tick].target_eef_position_world, copy=True)
        else:
            eef = np.array(reference.eef_positions_world[0], copy=True)

    late = decisions[close_tick + 1]
    assert late.phase is StructuredExpertPhase.GRASP_FUNNEL
    assert not np.array_equal(late.target_eef_position_world, intent.capture_position_world)
    np.testing.assert_allclose(
        late.target_eef_position_world,
        _object_at(intent, close_tick + 1)
        + late.estimated_object_velocity_world * intent.strategy.prediction_lead_seconds,
        atol=1.0e-12,
    )


def test_funnel_transition_waits_for_achieved_entry_and_fails_at_entry_deadline(
    structured_case,
) -> None:
    """Break caught: nominal time advances into the funnel while the real EEF is still behind."""
    from latency_meta_mdp.expert_realization.executor import (
        SemanticExecutionFailure,
        StructuredExpertPhase,
    )
    from latency_meta_mdp.handoff import HandoffState

    _task, intent, reference = structured_case
    executor = _executor(structured_case)
    far = np.array(reference.eef_positions_world[0], copy=True)
    deadline = intent.approach.funnel_entry_deadline_tick
    decisions = {}
    for tick in range(5, deadline + 1):
        decisions[tick] = executor.next_action(
            snapshot=_snapshot(
                tick=tick,
                object_position=_object_at(intent, tick),
                eef_position=far,
            ),
            handoff_state=HandoffState.DRIVEN,
            left_pad_contact=False,
            right_pad_contact=False,
        )

    assert decisions[intent.approach.funnel_entry_target_tick].phase is (
        StructuredExpertPhase.SMOOTH_APPROACH
    )
    with pytest.raises(SemanticExecutionFailure, match="funnel entry"):
        executor.next_action(
            snapshot=_snapshot(
                tick=deadline + 1,
                object_position=_object_at(intent, deadline + 1),
                eef_position=far,
            ),
            handoff_state=HandoffState.DRIVEN,
            left_pad_contact=False,
            right_pad_contact=False,
        )


def test_funnel_endpoint_is_reanchored_to_observed_object_at_actual_entry(
    structured_case,
) -> None:
    """Break caught: a delayed funnel still descends to the stale nominal capture point."""
    from latency_meta_mdp.expert_realization.executor import StructuredExpertPhase
    from latency_meta_mdp.handoff import HandoffState

    _task, intent, reference = structured_case
    executor = _executor(structured_case)
    shift = np.array([0.02, -0.01, 0.0], dtype=np.float64)
    eef = np.array(reference.eef_positions_world[0], copy=True)
    decisions = {}
    maximum_tick = (
        intent.approach.funnel_entry_deadline_tick + intent.strategy.funnel_descent_ticks
    )
    for tick in range(5, maximum_tick):
        decisions[tick] = executor.next_action(
            snapshot=_snapshot(
                tick=tick,
                object_position=_object_at(intent, tick) + shift,
                eef_position=eef,
            ),
            handoff_state=HandoffState.DRIVEN,
            left_pad_contact=False,
            right_pad_contact=False,
        )
        eef = np.array(decisions[tick].target_eef_position_world, copy=True)

    funnel_start = next(
        tick
        for tick, decision in decisions.items()
        if decision.phase is StructuredExpertPhase.GRASP_FUNNEL
    )
    effective_close = funnel_start + intent.strategy.funnel_descent_ticks
    final_funnel_command = decisions[effective_close - 1]
    assert final_funnel_command.phase is StructuredExpertPhase.GRASP_FUNNEL
    np.testing.assert_allclose(
        final_funnel_command.target_eef_position_world,
        _object_at(intent, effective_close - 1)
        + shift
        + final_funnel_command.estimated_object_velocity_world
        * intent.strategy.prediction_lead_seconds,
        atol=1.0e-12,
    )


def test_funnel_centers_xy_at_safe_height_before_final_descent(structured_case) -> None:
    """Break caught: the gripper descends while still laterally offset and hits one pad first."""
    from latency_meta_mdp.handoff import HandoffState

    _task, intent, reference = structured_case
    executor = _executor(structured_case)
    eef = np.array(reference.eef_positions_world[0], copy=True)
    entry_tick = intent.approach.funnel_entry_target_tick
    midpoint_tick = entry_tick + int(round(0.6 * intent.strategy.funnel_descent_ticks)) - 1
    decisions = {}
    for tick in range(5, midpoint_tick + 1):
        decisions[tick] = executor.next_action(
            snapshot=_snapshot(
                tick=tick,
                object_position=_object_at(intent, tick),
                eef_position=eef,
            ),
            handoff_state=HandoffState.DRIVEN,
            left_pad_contact=False,
            right_pad_contact=False,
        )
        eef = np.array(decisions[tick].target_eef_position_world, copy=True)

    target = decisions[midpoint_tick].target_eef_position_world
    target_object = _object_at(intent, midpoint_tick + 1)
    assert np.linalg.norm(target[:2] - target_object[:2]) < 0.025
    assert target[2] - target_object[2] > 0.045
