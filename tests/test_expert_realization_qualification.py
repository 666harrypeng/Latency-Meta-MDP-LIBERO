from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest


def _gate():
    from latency_meta_mdp.data.collection.config import load_pilot_gate_config

    return load_pilot_gate_config(Path("configs/analysis/panda_ball_structured_pilot_gate.yaml"))


def _report():
    from latency_meta_mdp.data.collection.safety import ActualRolloutSafetyReport

    gate = _gate()
    return ActualRolloutSafetyReport(
        terminal_success=True,
        physical_handoff=True,
        phase_order_valid=True,
        minimum_non_contact_environment_clearance_m=(gate.non_contact_environment_clearance_m),
        maximum_intentional_contact_penetration_m=gate.intentional_contact_penetration_m,
        maximum_pad_ball_impulse_ns=gate.peak_pad_ball_impulse_per_physics_contact_event_ns,
        unintended_pregrasp_ball_contacts=gate.unintended_pregrasp_ball_contacts,
        other_link_ball_contacts=0,
        robot_environment_contacts=0,
        robot_self_contacts=0,
        minimum_joint_position_margin_rad=gate.joint_position_margin_rad,
        maximum_joint_velocity_fraction=gate.joint_velocity_fraction_of_model_limit,
        maximum_eef_speed_mps=gate.eef_speed_mps,
        maximum_eef_acceleration_mps2=gate.eef_acceleration_mps2,
        maximum_eef_jerk_mps3=gate.eef_jerk_mps3,
        maximum_reference_tracking_error_m=(gate.reference_to_achieved_eef_error_outside_contact_m),
        maximum_pregrasp_translation_error_m=gate.pregrasp_tracking_translation_m,
        maximum_pregrasp_rotation_error_degrees=gate.pregrasp_tracking_rotation_degrees,
        minimum_osc_action=-1.0,
        maximum_osc_action=1.0,
        pre_handoff_saturation_fraction=gate.pre_handoff_saturation_fraction,
    )


def test_exact_gate_edges_are_admitted() -> None:
    """Break caught: a mathematically equal threshold is rejected by inconsistent inequalities."""
    from latency_meta_mdp.data.collection.qualification import qualify_actual_rollout

    decision = qualify_actual_rollout(_report(), gate=_gate())
    assert decision.eligible is True
    assert decision.failures == ()


@pytest.mark.parametrize(
    ("field", "bad_value", "failure"),
    [
        ("terminal_success", False, "terminal_success"),
        ("physical_handoff", False, "physical_handoff"),
        ("phase_order_valid", False, "phase_order"),
        ("minimum_non_contact_environment_clearance_m", 0.001999, "environment_clearance"),
        ("maximum_pad_ball_impulse_ns", 1.000001, "pad_ball_impulse"),
        ("unintended_pregrasp_ball_contacts", 1, "pregrasp_ball_contact"),
        ("other_link_ball_contacts", 1, "other_link_ball_contact"),
        ("robot_environment_contacts", 1, "robot_environment_contact"),
        ("robot_self_contacts", 1, "robot_self_contact"),
        ("minimum_joint_position_margin_rad", 0.000099, "joint_position_margin"),
        ("maximum_joint_velocity_fraction", 0.800001, "joint_velocity"),
        ("maximum_eef_speed_mps", 0.750001, "eef_speed"),
        ("maximum_eef_acceleration_mps2", 5.000001, "eef_acceleration"),
        ("minimum_osc_action", -1.000001, "osc_action_bounds"),
        ("maximum_osc_action", 1.000001, "osc_action_bounds"),
        ("pre_handoff_saturation_fraction", 0.050001, "pre_handoff_saturation"),
    ],
)
def test_each_rollout_gate_rejects_just_beyond_threshold(
    field: str,
    bad_value: object,
    failure: str,
) -> None:
    """Break caught: a recorded safety metric is omitted from actual-rollout admission."""
    from latency_meta_mdp.data.collection.qualification import qualify_actual_rollout

    decision = qualify_actual_rollout(replace(_report(), **{field: bad_value}), gate=_gate())
    assert decision.eligible is False
    assert failure in decision.failures


def test_model_sensitive_quantities_are_diagnostic_not_admission_proxies() -> None:
    """Break caught: soft-contact or finite-difference diagnostics override direct safety."""
    from latency_meta_mdp.data.collection.qualification import qualify_actual_rollout

    decision = qualify_actual_rollout(
        replace(
            _report(),
            maximum_reference_tracking_error_m=10.0,
            maximum_intentional_contact_penetration_m=0.1,
            maximum_eef_jerk_mps3=1_000.0,
            maximum_pregrasp_translation_error_m=1.0,
            maximum_pregrasp_rotation_error_degrees=180.0,
        ),
        gate=_gate(),
    )
    assert decision.eligible is True
    assert decision.failures == ()
