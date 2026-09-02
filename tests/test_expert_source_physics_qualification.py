from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


def _prepared(
    *,
    step: int,
    phase,
    contacts=(),
    clearance: float = 0.05,
    joint_margin: float = 0.4,
    velocity_fraction: float = 0.2,
):
    from latency_meta_mdp.expert_realization.safety import PreparedPhysicsSafetySample

    return PreparedPhysicsSafetySample(
        physics_step_index=step,
        time_us=step * 2_000,
        phase=phase,
        minimum_environment_clearance_m=clearance,
        minimum_joint_position_margin_rad=joint_margin,
        maximum_joint_velocity_fraction=velocity_fraction,
        contacts=contacts,
    )


def _completed(*, step: int, forces=()):
    from latency_meta_mdp.expert_realization.safety import CompletedPhysicsSafetySample

    return CompletedPhysicsSafetySample(
        physics_step_index=step,
        time_us=step * 2_000,
        normal_forces_n=forces,
    )


def test_physics_accumulator_pairs_step1_geometry_with_step2_force() -> None:
    """Break caught: impulse uses a force from a different split-step contact set."""
    from latency_meta_mdp.expert_realization.executor import StructuredExpertPhase
    from latency_meta_mdp.expert_realization.safety import (
        ContactKind,
        PhysicsContactObservation,
        PhysicsSafetyAccumulator,
    )

    contact = PhysicsContactObservation(
        contact_index=4,
        kind=ContactKind.PAD_BALL,
        penetration_m=0.001,
    )
    accumulator = PhysicsSafetyAccumulator(physics_dt_us=2_000)
    accumulator.observe_prepared(
        _prepared(step=0, phase=StructuredExpertPhase.CLOSE_STABILIZE, contacts=(contact,))
    )
    accumulator.observe_completed(_completed(step=1, forces=((4, 10.0),)))
    summary = accumulator.finalize()

    assert summary.maximum_intentional_contact_penetration_m == 0.001
    assert summary.maximum_pad_ball_impulse_ns == 0.02
    assert summary.unintended_pregrasp_ball_contacts == 0
    assert summary.other_link_ball_contacts == 0


def test_physics_accumulator_classifies_forbidden_contacts_without_conflation() -> None:
    """Break caught: pregrasp, other-link, environment, and self contacts share one vague count."""
    from latency_meta_mdp.expert_realization.executor import StructuredExpertPhase
    from latency_meta_mdp.expert_realization.safety import (
        ContactKind,
        PhysicsContactObservation,
        PhysicsSafetyAccumulator,
    )

    contacts = tuple(
        PhysicsContactObservation(index, kind, 0.0005)
        for index, kind in enumerate(
            (
                ContactKind.PAD_BALL,
                ContactKind.OTHER_ROBOT_BALL,
                ContactKind.ROBOT_ENVIRONMENT,
                ContactKind.ROBOT_SELF,
            )
        )
    )
    accumulator = PhysicsSafetyAccumulator(physics_dt_us=2_000)
    accumulator.observe_prepared(
        _prepared(
            step=0,
            phase=StructuredExpertPhase.SMOOTH_APPROACH,
            contacts=contacts,
            clearance=0.0,
        )
    )
    accumulator.observe_completed(
        _completed(step=1, forces=tuple((index, 1.0) for index in range(4)))
    )
    summary = accumulator.finalize()

    assert summary.unintended_pregrasp_ball_contacts == 1
    assert summary.other_link_ball_contacts == 1
    assert summary.robot_environment_contacts == 1
    assert summary.robot_self_contacts == 1
    assert summary.minimum_non_contact_environment_clearance_m == 0.0


def test_physics_accumulator_reduces_joint_and_clearance_extrema() -> None:
    """Break caught: final report uses only the last physics point instead of the worst point."""
    from latency_meta_mdp.expert_realization.executor import StructuredExpertPhase
    from latency_meta_mdp.expert_realization.safety import PhysicsSafetyAccumulator

    accumulator = PhysicsSafetyAccumulator(physics_dt_us=2_000)
    accumulator.observe_prepared(
        _prepared(
            step=0,
            phase=StructuredExpertPhase.SMOOTH_APPROACH,
            clearance=0.04,
            joint_margin=0.3,
            velocity_fraction=0.2,
        )
    )
    accumulator.observe_completed(_completed(step=1))
    accumulator.observe_prepared(
        _prepared(
            step=1,
            phase=StructuredExpertPhase.GRASP_FUNNEL,
            clearance=0.01,
            joint_margin=0.2,
            velocity_fraction=0.6,
        )
    )
    accumulator.observe_completed(_completed(step=2))
    summary = accumulator.finalize()

    assert summary.minimum_non_contact_environment_clearance_m == 0.01
    assert summary.minimum_joint_position_margin_rad == 0.2
    assert summary.maximum_joint_velocity_fraction == 0.6


def test_physics_accumulator_rejects_clock_and_contact_force_mismatch() -> None:
    """Break caught: missing or reordered callbacks silently corrupt contact impulse."""
    from latency_meta_mdp.expert_realization.executor import StructuredExpertPhase
    from latency_meta_mdp.expert_realization.safety import (
        ContactKind,
        PhysicsContactObservation,
        PhysicsSafetyAccumulator,
    )

    accumulator = PhysicsSafetyAccumulator(physics_dt_us=2_000)
    contact = PhysicsContactObservation(3, ContactKind.PAD_BALL, 0.0)
    accumulator.observe_prepared(
        _prepared(step=0, phase=StructuredExpertPhase.CLOSE_STABILIZE, contacts=(contact,))
    )
    with pytest.raises(ValueError, match="contact-force indices"):
        accumulator.observe_completed(_completed(step=1, forces=((2, 1.0),)))
    with pytest.raises(RuntimeError, match="pending prepared physics point"):
        accumulator.finalize()


def test_compiled_geometry_classifies_contacts_by_exact_geom_ids() -> None:
    """Break caught: contact type is guessed from unstable geom-name prefixes."""
    import numpy as np

    from latency_meta_mdp.expert_realization.safety import (
        CompiledSafetyGeometry,
        ContactKind,
        classify_contact_pair,
    )

    geometry = CompiledSafetyGeometry(
        left_pad_geom_ids=frozenset({1}),
        right_pad_geom_ids=frozenset({2}),
        grasp_contact_geom_ids=frozenset({1, 2, 8, 9}),
        ball_geom_ids=frozenset({3}),
        movable_robot_geom_ids=frozenset({1, 2, 4, 8, 9}),
        robot_assembly_geom_ids=frozenset({1, 2, 4, 5, 8, 9}),
        environment_geom_ids=frozenset({6, 7}),
        robot_qpos_indices=np.arange(7, dtype=np.int64),
        robot_qvel_indices=np.arange(7, dtype=np.int64),
        joint_lower=np.full(7, -1.0, dtype=np.float64),
        joint_upper=np.full(7, 1.0, dtype=np.float64),
        joint_velocity_limit=np.full(7, 2.0, dtype=np.float64),
    )

    assert classify_contact_pair(geometry, 1, 3) is ContactKind.PAD_BALL
    assert classify_contact_pair(geometry, 3, 2) is ContactKind.PAD_BALL
    assert classify_contact_pair(geometry, 8, 3) is ContactKind.PAD_BALL
    assert classify_contact_pair(geometry, 4, 3) is ContactKind.OTHER_ROBOT_BALL
    assert classify_contact_pair(geometry, 4, 6) is ContactKind.ROBOT_ENVIRONMENT
    assert classify_contact_pair(geometry, 5, 4) is ContactKind.ROBOT_SELF
    assert classify_contact_pair(geometry, 3, 6) is None


def test_compiled_geometry_arrays_are_exact_and_immutable() -> None:
    """Break caught: runtime geometry carries malformed or mutable joint limits."""
    import numpy as np

    from latency_meta_mdp.expert_realization.safety import CompiledSafetyGeometry

    kwargs = dict(
        left_pad_geom_ids=frozenset({1}),
        right_pad_geom_ids=frozenset({2}),
        grasp_contact_geom_ids=frozenset({1, 2}),
        ball_geom_ids=frozenset({3}),
        movable_robot_geom_ids=frozenset({1, 2, 4}),
        robot_assembly_geom_ids=frozenset({1, 2, 4, 5}),
        environment_geom_ids=frozenset({6, 7}),
        robot_qpos_indices=np.arange(7, dtype=np.int64),
        robot_qvel_indices=np.arange(7, dtype=np.int64),
        joint_lower=np.full(7, -1.0, dtype=np.float64),
        joint_upper=np.full(7, 1.0, dtype=np.float64),
        joint_velocity_limit=np.full(7, 2.0, dtype=np.float64),
    )
    geometry = CompiledSafetyGeometry(**kwargs)
    assert not geometry.joint_lower.flags.writeable
    with pytest.raises(ValueError):
        geometry.joint_lower[0] = 0.0
    with pytest.raises((TypeError, ValueError), match="joint_lower"):
        CompiledSafetyGeometry(**{**kwargs, "joint_lower": np.zeros(6, dtype=np.float64)})


def test_joint_margin_is_signed_so_a_limit_violation_cannot_be_hidden() -> None:
    """Break caught: a joint outside its limit is clipped to a safe zero margin."""
    from latency_meta_mdp.expert_realization.executor import StructuredExpertPhase

    sample = _prepared(
        step=0,
        phase=StructuredExpertPhase.SMOOTH_APPROACH,
        joint_margin=-0.001,
    )
    assert sample.minimum_joint_position_margin_rad == -0.001


def test_runtime_monitor_binds_boundary_geometry_to_the_next_action_phase() -> None:
    """Break caught: boundary step1 contact is mislabeled with the preceding phase."""
    from latency_meta_mdp.backend import CompletedPhysicsStep, PreparedPhysicsPoint
    from latency_meta_mdp.expert_realization.actual_physics import RuntimePhysicsSafetyMonitor
    from latency_meta_mdp.expert_realization.executor import StructuredExpertPhase
    from latency_meta_mdp.expert_realization.safety import ContactKind, PhysicsContactObservation

    prepared_calls = []
    completed_calls = []

    def prepared_reader(_env, point, _geometry):
        prepared_calls.append(point.physics_step_index)
        return SimpleNamespace(
            minimum_environment_clearance_m=0.02,
            minimum_joint_position_margin_rad=0.3,
            maximum_joint_velocity_fraction=0.4,
            contacts=(PhysicsContactObservation(2, ContactKind.PAD_BALL, 0.001),),
        )

    def completed_reader(_env, point, contact_indices):
        completed_calls.append((point.physics_step_index, contact_indices))
        return ((2, 5.0),)

    monitor = RuntimePhysicsSafetyMonitor(
        env=object(),
        geometry=object(),
        prepared_reader=prepared_reader,
        completed_reader=completed_reader,
    )
    monitor.on_prepared(
        PreparedPhysicsPoint(0, 0, 0, True)
    )
    monitor.set_active_interval_phase(StructuredExpertPhase.CLOSE_STABILIZE)
    monitor.on_completed(
        CompletedPhysicsStep(1, 0, 2_000, False)
    )
    monitor.on_prepared(
        PreparedPhysicsPoint(1, 0, 2_000, False)
    )
    monitor.set_active_interval_phase(StructuredExpertPhase.LIFT)
    monitor.on_completed(
        CompletedPhysicsStep(2, 0, 4_000, False)
    )
    summary = monitor.finalize()

    assert prepared_calls == [0, 1]
    assert completed_calls == [(1, (2,)), (2, (2,))]
    assert summary.maximum_intentional_contact_penetration_m == 0.001
    assert summary.maximum_pad_ball_impulse_ns == 0.01


def test_runtime_monitor_rejects_completion_without_interval_phase() -> None:
    """Break caught: a split-step contact is silently assigned to an unknown phase."""
    from latency_meta_mdp.backend import CompletedPhysicsStep, PreparedPhysicsPoint
    from latency_meta_mdp.expert_realization.actual_physics import RuntimePhysicsSafetyMonitor

    monitor = RuntimePhysicsSafetyMonitor(
        env=object(),
        geometry=object(),
        prepared_reader=lambda *_: SimpleNamespace(
            minimum_environment_clearance_m=0.02,
            minimum_joint_position_margin_rad=0.3,
            maximum_joint_velocity_fraction=0.4,
            contacts=(),
        ),
        completed_reader=lambda *_: (),
    )
    monitor.on_prepared(PreparedPhysicsPoint(0, 0, 0, True))
    with pytest.raises(RuntimeError, match="active interval phase"):
        monitor.on_completed(CompletedPhysicsStep(1, 0, 2_000, False))


def test_runtime_monitor_only_discards_an_unexecuted_formal_boundary() -> None:
    """Break caught: finalize hides a missing step2 from the middle of a formal interval."""
    from latency_meta_mdp.backend import PreparedPhysicsPoint
    from latency_meta_mdp.expert_realization.actual_physics import RuntimePhysicsSafetyMonitor

    monitor = RuntimePhysicsSafetyMonitor(
        env=object(),
        geometry=object(),
        prepared_reader=lambda *_: SimpleNamespace(
            minimum_environment_clearance_m=0.02,
            minimum_joint_position_margin_rad=0.3,
            maximum_joint_velocity_fraction=0.4,
            contacts=(),
        ),
        completed_reader=lambda *_: (),
    )
    monitor.on_prepared(PreparedPhysicsPoint(1, 0, 2_000, False))
    with pytest.raises(RuntimeError, match="non-boundary"):
        monitor.finalize()


def test_rollout_report_uses_full_trace_and_phase_specific_errors() -> None:
    """Break caught: report extrema use only the terminal boundary or conflate contact phases."""
    from latency_meta_mdp.expert_realization.executor import StructuredExpertPhase
    from latency_meta_mdp.expert_realization.safety import (
        PhysicsSafetySummary,
        build_actual_rollout_safety_report,
    )

    decisions = (
        SimpleNamespace(
            phase=StructuredExpertPhase.SMOOTH_APPROACH,
            target_eef_position_world=np.array([0.007, 0.0, 0.0], dtype=np.float64),
            position_error_base=np.array([0.003, 0.004, 0.0], dtype=np.float64),
            orientation_error_rotvec_base=np.array([0.0, 0.0, np.deg2rad(1.0)]),
            saturated=False,
            source_formal_tick=0,
        ),
        SimpleNamespace(
            phase=StructuredExpertPhase.GRASP_FUNNEL,
            target_eef_position_world=np.zeros(3, dtype=np.float64),
            position_error_base=np.array([0.012, 0.0, 0.0], dtype=np.float64),
            orientation_error_rotvec_base=np.array([0.0, 0.0, np.deg2rad(1.0)]),
            saturated=True,
            source_formal_tick=1,
        ),
        SimpleNamespace(
            phase=StructuredExpertPhase.CLOSE_STABILIZE,
            target_eef_position_world=np.zeros(3, dtype=np.float64),
            position_error_base=np.array([0.5, 0.0, 0.0], dtype=np.float64),
            orientation_error_rotvec_base=np.zeros(3, dtype=np.float64),
            saturated=False,
            source_formal_tick=2,
        ),
        SimpleNamespace(
            phase=StructuredExpertPhase.LIFT,
            target_eef_position_world=np.zeros(3, dtype=np.float64),
            position_error_base=np.zeros(3, dtype=np.float64),
            orientation_error_rotvec_base=np.zeros(3, dtype=np.float64),
            saturated=False,
            source_formal_tick=3,
        ),
    )
    rollout = SimpleNamespace(
        terminal_status="success",
        physical_handoff_tick=3,
        phase_sequence=tuple(item.phase.value for item in decisions),
        decisions=decisions,
        actions=np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        eef_positions_world=np.array(
            [[0.0, 0.0, 0.0], [0.002, 0.0, 0.0], [0.006, 0.0, 0.0],
             [0.012, 0.0, 0.0], [0.020, 0.0, 0.0]],
            dtype=np.float64,
        ),
        object_positions_world=np.array(
            [[0.0, 0.0, 0.0], [-0.003, 0.0, 0.0], [0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.float64,
        ),
    )
    physics = PhysicsSafetySummary(
        minimum_non_contact_environment_clearance_m=0.03,
        maximum_intentional_contact_penetration_m=0.001,
        maximum_pad_ball_impulse_ns=0.02,
        unintended_pregrasp_ball_contacts=0,
        other_link_ball_contacts=0,
        robot_environment_contacts=0,
        robot_self_contacts=0,
        minimum_joint_position_margin_rad=0.2,
        maximum_joint_velocity_fraction=0.3,
    )

    report = build_actual_rollout_safety_report(
        rollout=rollout,
        physics=physics,
    )

    assert report.phase_order_valid is True
    assert report.maximum_eef_speed_mps == pytest.approx(0.4)
    assert report.maximum_eef_acceleration_mps2 == pytest.approx(5.0)
    assert report.maximum_eef_jerk_mps3 == pytest.approx(0.0)
    assert report.maximum_reference_tracking_error_m == pytest.approx(0.012)
    assert report.maximum_pregrasp_translation_error_m == pytest.approx(0.005)
    assert report.maximum_pregrasp_rotation_error_degrees == pytest.approx(1.0)
    assert report.minimum_osc_action == -1.0
    assert report.maximum_osc_action == 1.0
    assert report.pre_handoff_saturation_fraction == pytest.approx(1.0 / 3.0)


def test_rollout_report_rejects_nonmonotonic_phase_sequence() -> None:
    """Break caught: a return from close to approach is reported as valid phase order."""
    from latency_meta_mdp.expert_realization.executor import StructuredExpertPhase
    from latency_meta_mdp.expert_realization.safety import phase_order_is_valid

    assert phase_order_is_valid(
        (
            StructuredExpertPhase.SMOOTH_APPROACH.value,
            StructuredExpertPhase.GRASP_FUNNEL.value,
            StructuredExpertPhase.CLOSE_STABILIZE.value,
            StructuredExpertPhase.LIFT.value,
        )
    )
    assert not phase_order_is_valid(
        (
            StructuredExpertPhase.CLOSE_STABILIZE.value,
            StructuredExpertPhase.SMOOTH_APPROACH.value,
        )
    )


def test_physics_observer_composition_preserves_order_and_rejects_overlap() -> None:
    """Break caught: safety callback replaces handoff updates or overwrites their fields."""
    from latency_meta_mdp.backend import PreparedPhysicsPoint
    from latency_meta_mdp.expert_realization.task_instance import _compose_physics_observers

    calls = []
    point = PreparedPhysicsPoint(0, 0, 0, True)

    def first(_point):
        calls.append("handoff")
        return {"handoff": np.array(True)}

    def second(_point):
        calls.append("safety")
        return {"safety": np.array(True)}

    composed = _compose_physics_observers(first, second)
    assert set(composed(point)) == {"handoff", "safety"}
    assert calls == ["handoff", "safety"]

    conflicting = _compose_physics_observers(first, lambda _point: {"handoff": np.array(False)})
    with pytest.raises(ValueError, match="overlapping fields"):
        conflicting(point)


def test_real_panda_runtime_compiles_and_samples_exact_safety_roles() -> None:
    """Break caught: unit-only geometry IDs do not match the compiled RoboSuite model."""
    from pathlib import Path

    from latency_meta_mdp.expert_realization.executor import StructuredExpertPhase
    from latency_meta_mdp.expert_realization.robot_bridge import build_panda_planning_bridge
    from latency_meta_mdp.expert_realization.task_instance import (
        _build_task_instance_runtime,
        materialize_task_instance,
    )

    root = Path.cwd()
    bridge = build_panda_planning_bridge(root)
    task = materialize_task_instance(project_root=root, level=1, task_instance_seed=4000)
    runtime = _build_task_instance_runtime(task, safety_bridge=bridge)
    try:
        runtime.executor.initialize()
        geometry = runtime.safety_monitor.geometry
        names = runtime.env.sim.model.geom_id2name
        assert {names(index) for index in geometry.environment_geom_ids} == {
            "floor",
            "table_collision",
        }
        assert {
            names(index) for index in geometry.left_pad_geom_ids
        } == {"gripper0_right_finger1_pad_collision"}
        assert {
            names(index) for index in geometry.right_pad_geom_ids
        } == {"gripper0_right_finger2_pad_collision"}
        runtime.safety_monitor.set_active_interval_phase(
            StructuredExpertPhase.SMOOTH_APPROACH
        )
        runtime.executor.step_formal(task.expected_anchor.shared_actions[0])
        summary = runtime.safety_monitor.finalize()
        assert summary.minimum_non_contact_environment_clearance_m > 0.002
        assert summary.robot_environment_contacts == 0
        assert summary.robot_self_contacts == 0
    finally:
        runtime.close()
