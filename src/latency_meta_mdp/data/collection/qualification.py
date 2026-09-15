"""Hard admission decision for one actual structured-expert rollout."""

from __future__ import annotations

from dataclasses import dataclass

from latency_meta_mdp.data.collection.config import PilotGateConfig
from latency_meta_mdp.data.collection.safety import ActualRolloutSafetyReport


@dataclass(frozen=True)
class QualificationDecision:
    eligible: bool
    failures: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.eligible) is not bool:
            raise TypeError("eligible must be boolean")
        if type(self.failures) is not tuple or any(
            type(item) is not str or not item for item in self.failures
        ):
            raise TypeError("failures must be a tuple of non-empty strings")
        if self.eligible != (not self.failures):
            raise ValueError("eligible must equal absence of failures")

    def to_mapping(self) -> dict[str, object]:
        return {"eligible": self.eligible, "failures": list(self.failures)}


def qualify_actual_rollout(
    report: ActualRolloutSafetyReport,
    *,
    gate: PilotGateConfig,
) -> QualificationDecision:
    if not isinstance(report, ActualRolloutSafetyReport):
        raise TypeError("report must be an ActualRolloutSafetyReport")
    if not isinstance(gate, PilotGateConfig):
        raise TypeError("gate must be a PilotGateConfig")
    failures = []
    checks = (
        (report.terminal_success, "terminal_success"),
        (report.physical_handoff, "physical_handoff"),
        (report.phase_order_valid, "phase_order"),
        (
            report.minimum_non_contact_environment_clearance_m
            >= gate.non_contact_environment_clearance_m,
            "environment_clearance",
        ),
        (
            report.maximum_pad_ball_impulse_ns
            <= gate.peak_pad_ball_impulse_per_physics_contact_event_ns,
            "pad_ball_impulse",
        ),
        (
            report.unintended_pregrasp_ball_contacts <= gate.unintended_pregrasp_ball_contacts,
            "pregrasp_ball_contact",
        ),
        (report.other_link_ball_contacts == 0, "other_link_ball_contact"),
        (report.robot_environment_contacts == 0, "robot_environment_contact"),
        (report.robot_self_contacts == 0, "robot_self_contact"),
        (
            report.minimum_joint_position_margin_rad >= gate.joint_position_margin_rad,
            "joint_position_margin",
        ),
        (
            report.maximum_joint_velocity_fraction <= gate.joint_velocity_fraction_of_model_limit,
            "joint_velocity",
        ),
        (report.maximum_eef_speed_mps <= gate.eef_speed_mps, "eef_speed"),
        (
            report.maximum_eef_acceleration_mps2 <= gate.eef_acceleration_mps2,
            "eef_acceleration",
        ),
        (report.minimum_osc_action >= gate.osc_action_bounds[0], "osc_action_bounds"),
        (report.maximum_osc_action <= gate.osc_action_bounds[1], "osc_action_bounds"),
        (
            report.pre_handoff_saturation_fraction <= gate.pre_handoff_saturation_fraction,
            "pre_handoff_saturation",
        ),
    )
    for passed, name in checks:
        if not passed and name not in failures:
            failures.append(name)
    return QualificationDecision(eligible=not failures, failures=tuple(failures))
