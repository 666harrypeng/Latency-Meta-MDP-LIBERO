"""Physical-safety metrics for structured expert references and MuJoCo rollouts."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import Any

import numpy as np

from latency_meta_mdp.expert_realization.executor import StructuredExpertPhase


class ContactKind(str, Enum):
    PAD_BALL = "pad_ball"
    OTHER_ROBOT_BALL = "other_robot_ball"
    ROBOT_ENVIRONMENT = "robot_environment"
    ROBOT_SELF = "robot_self"


def contact_is_allowed(kind: ContactKind, phase: StructuredExpertPhase) -> bool:
    if not isinstance(kind, ContactKind) or not isinstance(phase, StructuredExpertPhase):
        raise TypeError("contact admission requires typed kind and phase")
    return bool(
        kind is ContactKind.PAD_BALL
        and phase in {StructuredExpertPhase.CLOSE_STABILIZE, StructuredExpertPhase.LIFT}
    )


@dataclass(frozen=True)
class SweptClearanceReport:
    safe: bool
    minimum_distance_m: float
    minimum_clearance_m: float
    minimum_time_us: int
    sample_count: int

    def __post_init__(self) -> None:
        if type(self.safe) is not bool:
            raise TypeError("safe must be boolean")
        for name in ("minimum_distance_m", "minimum_clearance_m"):
            value = getattr(self, name)
            if type(value) is not float or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite float")
        if type(self.minimum_time_us) is not int or self.minimum_time_us < 0:
            raise ValueError("minimum_time_us must be non-negative")
        if type(self.sample_count) is not int or self.sample_count < 2:
            raise ValueError("sample_count must be at least two")


def _trajectory(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.dtype != np.float64
        or array.ndim != 2
        or array.shape[1:] != (3,)
        or len(array) < 2
        or not np.all(np.isfinite(array))
    ):
        raise ValueError(f"{name} must be finite float64[N,3]")
    return array


def evaluate_moving_target_swept_clearance(
    *,
    reference_positions_world: np.ndarray,
    object_positions_world: np.ndarray,
    reference_start_time_us: int,
    minimum_center_distance_m: float,
) -> SweptClearanceReport:
    """Evaluate paired 50 Hz paths on every intervening certified 2 ms physics point."""
    reference = _trajectory(reference_positions_world, name="reference_positions_world")
    objects = _trajectory(object_positions_world, name="object_positions_world")
    if reference.shape != objects.shape:
        raise ValueError("reference and object trajectories must be aligned")
    if type(reference_start_time_us) is not int or reference_start_time_us < 0:
        raise ValueError("reference_start_time_us must be non-negative")
    if reference_start_time_us % 20_000:
        raise ValueError("reference_start_time_us must lie on the formal grid")
    if (
        type(minimum_center_distance_m) is not float
        or not np.isfinite(minimum_center_distance_m)
        or minimum_center_distance_m <= 0.0
    ):
        raise ValueError("minimum_center_distance_m must be a positive finite float")

    distances = []
    times = []
    for interval in range(len(reference) - 1):
        for substep in range(10):
            alpha = substep / 10.0
            reference_point = (1.0 - alpha) * reference[interval] + alpha * reference[interval + 1]
            object_point = (1.0 - alpha) * objects[interval] + alpha * objects[interval + 1]
            distances.append(float(np.linalg.norm(reference_point - object_point)))
            times.append(reference_start_time_us + interval * 20_000 + substep * 2_000)
    distances.append(float(np.linalg.norm(reference[-1] - objects[-1])))
    times.append(reference_start_time_us + (len(reference) - 1) * 20_000)
    minimum_index = int(np.argmin(distances))
    minimum_distance = distances[minimum_index]
    return SweptClearanceReport(
        safe=minimum_distance >= minimum_center_distance_m,
        minimum_distance_m=float(minimum_distance),
        minimum_clearance_m=float(minimum_distance - minimum_center_distance_m),
        minimum_time_us=times[minimum_index],
        sample_count=len(distances),
    )


@dataclass(frozen=True)
class ActualRolloutSafetyReport:
    terminal_success: bool
    physical_handoff: bool
    phase_order_valid: bool
    minimum_non_contact_environment_clearance_m: float
    maximum_intentional_contact_penetration_m: float
    maximum_pad_ball_impulse_ns: float
    unintended_pregrasp_ball_contacts: int
    other_link_ball_contacts: int
    minimum_joint_position_margin_rad: float
    maximum_joint_velocity_fraction: float
    maximum_eef_speed_mps: float
    maximum_eef_acceleration_mps2: float
    maximum_eef_jerk_mps3: float
    maximum_reference_tracking_error_m: float
    maximum_pregrasp_translation_error_m: float
    maximum_pregrasp_rotation_error_degrees: float
    minimum_osc_action: float
    maximum_osc_action: float
    pre_handoff_saturation_fraction: float

    def __post_init__(self) -> None:
        for name in ("terminal_success", "physical_handoff", "phase_order_valid"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be boolean")
        for name in ("unintended_pregrasp_ball_contacts", "other_link_ball_contacts"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        excluded = {
            "terminal_success",
            "physical_handoff",
            "phase_order_valid",
            "unintended_pregrasp_ball_contacts",
            "other_link_ball_contacts",
        }
        for item in fields(self):
            if item.name in excluded:
                continue
            value = getattr(self, item.name)
            if type(value) is not float or not np.isfinite(value):
                raise ValueError(f"{item.name} must be a finite float")
        nonnegative = {
            "minimum_non_contact_environment_clearance_m",
            "maximum_intentional_contact_penetration_m",
            "maximum_pad_ball_impulse_ns",
            "minimum_joint_position_margin_rad",
            "maximum_joint_velocity_fraction",
            "maximum_eef_speed_mps",
            "maximum_eef_acceleration_mps2",
            "maximum_eef_jerk_mps3",
            "maximum_reference_tracking_error_m",
            "maximum_pregrasp_translation_error_m",
            "maximum_pregrasp_rotation_error_degrees",
            "pre_handoff_saturation_fraction",
        }
        if any(getattr(self, name) < 0.0 for name in nonnegative):
            raise ValueError("rollout safety magnitudes must be non-negative")
        if self.minimum_osc_action > self.maximum_osc_action:
            raise ValueError("OSC action extrema are reversed")
