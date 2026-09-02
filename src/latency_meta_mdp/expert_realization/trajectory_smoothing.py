"""Whole-approach Cartesian reference generation from semantic curve intent."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CartesianApproachReference:
    timestamps_seconds: np.ndarray
    positions_world: np.ndarray
    velocity_world: np.ndarray
    acceleration_world: np.ndarray
    jerk_world: np.ndarray
    control_points_world: np.ndarray

    def __post_init__(self) -> None:
        rows = len(self.timestamps_seconds)
        contracts = {
            "timestamps_seconds": (rows,),
            "positions_world": (rows, 3),
            "velocity_world": (rows, 3),
            "acceleration_world": (rows, 3),
            "jerk_world": (rows, 3),
            "control_points_world": (6, 3),
        }
        for name, shape in contracts.items():
            value = np.asarray(getattr(self, name))
            if value.dtype != np.float64 or value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite float64{shape}")
            object.__setattr__(self, name, _readonly(value))
        if rows < 2 or self.timestamps_seconds[0] != 0.0 or np.any(
            np.diff(self.timestamps_seconds) <= 0.0
        ):
            raise ValueError("Cartesian approach timestamps must start at zero and increase")


def _bezier(control_points: np.ndarray, parameter: np.ndarray) -> np.ndarray:
    degree = len(control_points) - 1
    return sum(
        math.comb(degree, index)
        * (1.0 - parameter[:, None]) ** (degree - index)
        * parameter[:, None] ** index
        * point
        for index, point in enumerate(control_points)
    )


def build_cartesian_approach_reference(
    *,
    start_position_world: np.ndarray,
    soft_guide_regions_world: np.ndarray,
    funnel_entry_position_world: np.ndarray,
    funnel_entry_tangent_world: np.ndarray,
    duration_seconds: float,
    sample_period_seconds: float,
) -> CartesianApproachReference:
    """Build one degree-five Cartesian curve; guides shape it but never create stops."""
    start = np.asarray(start_position_world)
    guides = np.asarray(soft_guide_regions_world)
    entry = np.asarray(funnel_entry_position_world)
    tangent = np.asarray(funnel_entry_tangent_world)
    if any(value.dtype != np.float64 for value in (start, guides, entry, tangent)) or (
        start.shape != (3,)
        or guides.ndim != 2
        or guides.shape[1:] != (3,)
        or len(guides) > 2
        or entry.shape != (3,)
        or tangent.shape != (3,)
        or not all(np.all(np.isfinite(value)) for value in (start, guides, entry, tangent))
        or not np.isclose(np.linalg.norm(tangent), 1.0, atol=1.0e-12, rtol=0.0)
    ):
        raise ValueError("Cartesian approach geometry is invalid")
    if (
        type(duration_seconds) is not float
        or type(sample_period_seconds) is not float
        or duration_seconds <= 0.0
        or sample_period_seconds <= 0.0
    ):
        raise ValueError("Cartesian approach timing must be positive floats")
    interval_count_float = duration_seconds / sample_period_seconds
    interval_count = int(round(interval_count_float))
    if not np.isclose(interval_count_float, interval_count, atol=1.0e-10, rtol=0.0):
        raise ValueError("Cartesian approach duration must align with its sample period")

    controls = np.stack(
        [
            start,
            start,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            entry - 0.04 * tangent,
            entry,
        ]
    )
    if len(guides) == 0:
        controls[2] = 0.35 * start + 0.65 * entry
        controls[3] = 0.15 * start + 0.85 * entry
    else:
        guide_parameters = np.array(
            [0.5] if len(guides) == 1 else [1.0 / 3.0, 2.0 / 3.0],
            dtype=np.float64,
        )
        coefficient_rows = []
        residual_rows = []
        for parameter_value, guide in zip(guide_parameters, guides, strict=True):
            weights = np.array(
                [
                    math.comb(5, index)
                    * (1.0 - parameter_value) ** (5 - index)
                    * parameter_value**index
                    for index in range(6)
                ],
                dtype=np.float64,
            )
            coefficient_rows.append(weights[2:4])
            residual_rows.append(
                guide
                - weights[0] * controls[0]
                - weights[1] * controls[1]
                - weights[4] * controls[4]
                - weights[5] * controls[5]
            )
        if len(guides) == 1:
            shared = residual_rows[0] / sum(coefficient_rows[0])
            controls[2] = shared
            controls[3] = shared
        else:
            controls[2:4] = np.linalg.solve(
                np.asarray(coefficient_rows, dtype=np.float64),
                np.asarray(residual_rows, dtype=np.float64),
            )
    timestamps = np.linspace(0.0, duration_seconds, interval_count + 1, dtype=np.float64)
    parameter = timestamps / duration_seconds
    positions = _bezier(controls, parameter)
    first_controls = 5.0 * np.diff(controls, axis=0)
    second_controls = 4.0 * np.diff(first_controls, axis=0)
    third_controls = 3.0 * np.diff(second_controls, axis=0)
    velocity = _bezier(first_controls, parameter) / duration_seconds
    acceleration = _bezier(second_controls, parameter) / duration_seconds**2
    jerk = _bezier(third_controls, parameter) / duration_seconds**3
    positions[0] = start
    positions[-1] = entry
    return CartesianApproachReference(
        timestamps_seconds=timestamps,
        positions_world=positions,
        velocity_world=velocity,
        acceleration_world=acceleration,
        jerk_world=jerk,
        control_points_world=controls,
    )
