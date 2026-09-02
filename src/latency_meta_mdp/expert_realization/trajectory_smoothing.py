"""Whole-approach joint-path smoothing for CuRobo geometric proposals."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SmoothedJointApproach:
    timestamps_seconds: np.ndarray
    qpos: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray

    def __post_init__(self) -> None:
        rows = len(self.timestamps_seconds)
        contracts = {
            "timestamps_seconds": (rows,),
            "qpos": (rows, 7),
            "velocity": (rows, 7),
            "acceleration": (rows, 7),
            "jerk": (rows, 7),
        }
        for name, shape in contracts.items():
            value = np.asarray(getattr(self, name))
            if value.dtype != np.float64 or value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite float64{shape}")
            object.__setattr__(self, name, _readonly(value))
        if rows < 2 or self.timestamps_seconds[0] != 0.0 or np.any(
            np.diff(self.timestamps_seconds) <= 0.0
        ):
            raise ValueError("smoothed timestamps must start at zero and increase")


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


def _validated_limit(value: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype != np.float64 or result.shape != (7,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite float64[7]")
    return result


def _progression(
    normalized_time: np.ndarray, *, terminal_rate: float = 0.6
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Quintic time law with rest start, nonzero terminal rate, and zero end acceleration."""
    coefficients = np.linalg.solve(
        np.array(
            [
                [1.0, 1.0, 1.0],
                [3.0, 4.0, 5.0],
                [6.0, 12.0, 20.0],
            ],
            dtype=np.float64,
        ),
        np.array([1.0, terminal_rate, 0.0], dtype=np.float64),
    )
    a3, a4, a5 = coefficients
    r = normalized_time
    position = a3 * r**3 + a4 * r**4 + a5 * r**5
    first = 3.0 * a3 * r**2 + 4.0 * a4 * r**3 + 5.0 * a5 * r**4
    second = 6.0 * a3 * r + 12.0 * a4 * r**2 + 20.0 * a5 * r**3
    third = 6.0 * a3 + 24.0 * a4 * r + 60.0 * a5 * r**2
    return position, first, second, third


def _corner_cutting(points: np.ndarray, *, rounds: int = 3) -> np.ndarray:
    """Round interior proposal corners while preserving the two endpoint configurations."""
    result = np.array(points, copy=True)
    for _ in range(rounds):
        rows = [result[0]]
        for left, right in zip(result[:-1], result[1:], strict=True):
            rows.append(0.75 * left + 0.25 * right)
            rows.append(0.25 * left + 0.75 * right)
        rows.append(result[-1])
        result = np.asarray(rows, dtype=np.float64)
    return result


def _drop_consecutive_duplicates(points: np.ndarray) -> np.ndarray:
    keep = np.concatenate(
        [np.array([True]), np.linalg.norm(np.diff(points, axis=0), axis=1) > 1.0e-12]
    )
    return points[keep]


def _low_pass_geometric_knots(points: np.ndarray, *, maximum_landmarks: int = 8) -> np.ndarray:
    filtered = np.array(points, copy=True)
    if len(filtered) >= 7:
        window = min(11, len(filtered) if len(filtered) % 2 else len(filtered) - 1)
        filtered = savgol_filter(
            filtered,
            window_length=window,
            polyorder=3,
            axis=0,
            mode="interp",
        )
        filtered[0] = points[0]
        filtered[-1] = points[-1]
    filtered = _drop_consecutive_duplicates(filtered)
    if len(filtered) <= maximum_landmarks:
        return filtered
    arc = np.concatenate(
        [np.array([0.0]), np.cumsum(np.linalg.norm(np.diff(filtered, axis=0), axis=1))]
    )
    targets = np.linspace(0.0, arc[-1], maximum_landmarks, dtype=np.float64)
    landmarks = np.stack(
        [np.interp(targets, arc, filtered[:, joint]) for joint in range(7)],
        axis=1,
    )
    landmarks[0] = points[0]
    landmarks[-1] = points[-1]
    return landmarks


def smooth_joint_approach(
    raw_qpos_path: np.ndarray,
    *,
    duration_seconds: float,
    sample_period_seconds: float,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    joint_velocity: np.ndarray,
    joint_acceleration: np.ndarray,
) -> SmoothedJointApproach:
    """Fit and sample one global C2 curve over a complete geometric proposal."""
    raw = np.asarray(raw_qpos_path)
    if raw.dtype != np.float64 or raw.ndim != 2 or raw.shape[1:] != (7,):
        raise ValueError("raw_qpos_path must be float64[N,7]")
    if not np.all(np.isfinite(raw)):
        raise ValueError("raw_qpos_path must be finite")
    if (
        type(duration_seconds) is not float
        or type(sample_period_seconds) is not float
        or not np.isfinite(duration_seconds)
        or not np.isfinite(sample_period_seconds)
        or duration_seconds <= 0.0
        or sample_period_seconds <= 0.0
    ):
        raise ValueError("duration and sample period must be positive finite floats")
    sample_count_float = duration_seconds / sample_period_seconds
    sample_intervals = int(round(sample_count_float))
    if not np.isclose(sample_count_float, sample_intervals, atol=1.0e-10, rtol=0.0):
        raise ValueError("duration must be an exact multiple of the sample period")

    lower = _validated_limit(joint_lower, name="joint_lower")
    upper = _validated_limit(joint_upper, name="joint_upper")
    velocity_limit = _validated_limit(joint_velocity, name="joint_velocity")
    acceleration_limit = _validated_limit(joint_acceleration, name="joint_acceleration")
    if np.any(lower >= upper) or np.any(velocity_limit <= 0.0) or np.any(
        acceleration_limit <= 0.0
    ):
        raise ValueError("joint limits are invalid")

    knots = _drop_consecutive_duplicates(raw)
    if len(knots) < 2:
        raise ValueError("raw_qpos_path must contain at least two distinct configurations")
    if len(knots) > 2:
        knots = _low_pass_geometric_knots(knots)
        knots = _drop_consecutive_duplicates(_corner_cutting(knots))
    arc = np.concatenate(
        [np.array([0.0]), np.cumsum(np.linalg.norm(np.diff(knots, axis=0), axis=1))]
    )
    parameter = arc / arc[-1]
    if np.any(np.diff(parameter) <= 0.0):
        deltas = np.diff(parameter)
        minimum = float(np.min(deltas))
        zero_count = int(np.count_nonzero(deltas <= 0.0))
        raise ValueError(
            "smoothed geometric parameter is not strictly increasing: "
            f"raw_rows={len(raw)}, knot_rows={len(knots)}, "
            f"nonpositive_count={zero_count}, min_delta={minimum:.17g}"
        )
    spline = CubicSpline(parameter, knots, axis=0)

    timestamps = np.linspace(0.0, duration_seconds, sample_intervals + 1, dtype=np.float64)
    normalized_time = timestamps / duration_seconds
    progress, dprogress_dr, d2progress_dr2, d3progress_dr3 = _progression(normalized_time)
    dprogress_dt = dprogress_dr / duration_seconds
    d2progress_dt2 = d2progress_dr2 / duration_seconds**2
    d3progress_dt3 = d3progress_dr3 / duration_seconds**3

    qpos = np.asarray(spline(progress), dtype=np.float64)
    dq_dp = np.asarray(spline(progress, 1), dtype=np.float64)
    d2q_dp2 = np.asarray(spline(progress, 2), dtype=np.float64)
    d3q_dp3 = np.asarray(spline(progress, 3), dtype=np.float64)
    velocity = dq_dp * dprogress_dt[:, None]
    acceleration = (
        d2q_dp2 * dprogress_dt[:, None] ** 2 + dq_dp * d2progress_dt2[:, None]
    )
    jerk = (
        d3q_dp3 * dprogress_dt[:, None] ** 3
        + 3.0 * d2q_dp2 * dprogress_dt[:, None] * d2progress_dt2[:, None]
        + dq_dp * d3progress_dt3[:, None]
    )

    tolerance = 1.0e-9
    if np.any(qpos < lower - tolerance) or np.any(qpos > upper + tolerance):
        raise ValueError("smoothed path violates joint position limits")
    if np.any(np.abs(velocity) > velocity_limit + tolerance):
        raise ValueError("smoothed path violates joint velocity limits")
    if np.any(np.abs(acceleration) > acceleration_limit + tolerance):
        ratio = np.abs(acceleration) / acceleration_limit[None, :]
        sample_index, joint_index = np.unravel_index(int(np.argmax(ratio)), ratio.shape)
        raise ValueError(
            "smoothed path violates joint acceleration limits: "
            f"maximum_ratio={ratio[sample_index, joint_index]:.6f}, "
            f"sample={sample_index}, joint={joint_index}"
        )
    return SmoothedJointApproach(
        timestamps_seconds=timestamps,
        qpos=qpos,
        velocity=velocity,
        acceleration=acceleration,
        jerk=jerk,
    )
