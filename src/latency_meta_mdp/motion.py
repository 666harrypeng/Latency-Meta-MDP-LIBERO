"""Time-indexed L0-L3 target motion profiles for the synchronized world driver."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import yaml


def _tuple(value: Any, *, name: str, length: int, integer: bool = False) -> tuple[Any, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must be a list of length {length}")
    expected_type = int if integer else (int, float)
    if any(isinstance(item, bool) or not isinstance(item, expected_type) for item in value):
        raise TypeError(f"{name} contains an invalid value")
    converter = int if integer else float
    converted = tuple(converter(item) for item in value)
    if not integer and not all(np.isfinite(item) for item in converted):
        raise ValueError(f"{name} must contain finite values")
    return converted


@dataclass(frozen=True)
class MotionConfig:
    schema_version: int
    profile_id: str
    level: int
    path_bounds_xy: tuple[float, float, float, float]
    stationary_position_xy: tuple[float, float] | None
    anchor_time_us: int
    min_chord_length_m: float | None
    max_chord_length_m: float | None
    min_path_length_m: float | None
    max_path_length_m: float | None
    max_speed_mps: float | None
    max_continuous_acceleration_mps2: float | None
    curve_deviation_range_m: tuple[float, float] | None
    segment_count_three_probability: float | None
    cubic_segment_probability: float | None
    two_segment_change_time_us_range: tuple[int, int] | None
    three_segment_first_change_time_us_range: tuple[int, int] | None
    three_segment_second_change_time_us_range: tuple[int, int] | None
    turn_angle_degrees_range: tuple[float, float] | None
    velocity_jump_range_mps: tuple[float, float] | None
    max_retries: int

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> MotionConfig:
        if raw.get("schema_version") != 2:
            raise ValueError("motion schema_version must be 2")
        level = raw.get("level")
        if isinstance(level, bool) or not isinstance(level, int) or level not in range(4):
            raise ValueError("motion level must be one of 0, 1, 2, 3")
        common = {
            "anchor_time_us",
            "level",
            "max_retries",
            "path_bounds_xy",
            "profile_id",
            "schema_version",
        }
        dynamic = {
            "cubic_segment_probability",
            "curve_deviation_range_m",
            "max_chord_length_m",
            "max_continuous_acceleration_mps2",
            "max_path_length_m",
            "max_speed_mps",
            "min_chord_length_m",
            "min_path_length_m",
            "segment_count_three_probability",
            "three_segment_first_change_time_us_range",
            "three_segment_second_change_time_us_range",
            "turn_angle_degrees_range",
            "two_segment_change_time_us_range",
            "velocity_jump_range_mps",
        }
        expected = common | ({"stationary_position_xy"} if level == 0 else dynamic)
        unknown = sorted(set(raw) - expected)
        missing = sorted(expected - set(raw))
        if unknown:
            raise ValueError(f"unknown motion config fields: {unknown}")
        if missing:
            raise ValueError(f"missing motion config fields: {missing}")
        expected_profile_id = (
            "dynamic_grasp_lift_l0"
            if level == 0
            else f"dynamic_grasp_lift_l{level}_v3"
        )
        if raw["profile_id"] != expected_profile_id:
            raise ValueError("motion profile_id does not match level")
        for name in ("anchor_time_us", "max_retries"):
            value = raw[name]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        scalar_names = (
            "cubic_segment_probability",
            "max_chord_length_m",
            "max_continuous_acceleration_mps2",
            "max_path_length_m",
            "max_speed_mps",
            "min_chord_length_m",
            "min_path_length_m",
            "segment_count_three_probability",
        )
        scalars: dict[str, float] = {}
        if level:
            for name in scalar_names:
                value = raw[name]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError(f"{name} must be numeric")
                scalars[name] = float(value)
                if not np.isfinite(scalars[name]):
                    raise ValueError(f"{name} must be finite")
        config = cls(
            schema_version=2,
            profile_id=raw["profile_id"],
            level=level,
            path_bounds_xy=_tuple(raw["path_bounds_xy"], name="path_bounds_xy", length=4),
            stationary_position_xy=(
                _tuple(raw["stationary_position_xy"], name="stationary_position_xy", length=2)
                if level == 0
                else None
            ),
            anchor_time_us=raw["anchor_time_us"],
            min_chord_length_m=scalars.get("min_chord_length_m"),
            max_chord_length_m=scalars.get("max_chord_length_m"),
            min_path_length_m=scalars.get("min_path_length_m"),
            max_path_length_m=scalars.get("max_path_length_m"),
            max_speed_mps=scalars.get("max_speed_mps"),
            max_continuous_acceleration_mps2=scalars.get(
                "max_continuous_acceleration_mps2"
            ),
            curve_deviation_range_m=(
                _tuple(raw["curve_deviation_range_m"], name="curve_deviation_range_m", length=2)
                if level
                else None
            ),
            segment_count_three_probability=scalars.get(
                "segment_count_three_probability"
            ),
            cubic_segment_probability=scalars.get("cubic_segment_probability"),
            two_segment_change_time_us_range=(
                _tuple(
                    raw["two_segment_change_time_us_range"],
                    name="two_segment_change_time_us_range",
                    length=2,
                    integer=True,
                )
                if level
                else None
            ),
            three_segment_first_change_time_us_range=(
                _tuple(
                    raw["three_segment_first_change_time_us_range"],
                    name="three_segment_first_change_time_us_range",
                    length=2,
                    integer=True,
                )
                if level
                else None
            ),
            three_segment_second_change_time_us_range=(
                _tuple(
                    raw["three_segment_second_change_time_us_range"],
                    name="three_segment_second_change_time_us_range",
                    length=2,
                    integer=True,
                )
                if level
                else None
            ),
            turn_angle_degrees_range=(
                _tuple(
                    raw["turn_angle_degrees_range"],
                    name="turn_angle_degrees_range",
                    length=2,
                )
                if level
                else None
            ),
            velocity_jump_range_mps=(
                _tuple(
                    raw["velocity_jump_range_mps"],
                    name="velocity_jump_range_mps",
                    length=2,
                )
                if level
                else None
            ),
            max_retries=raw["max_retries"],
        )
        config.validate()
        return config

    def validate(self) -> None:
        x_min, x_max, y_min, y_max = self.path_bounds_xy
        if not x_min < x_max or not y_min < y_max:
            raise ValueError("path bounds must be ordered")
        if self.anchor_time_us % 20_000:
            raise ValueError("anchor_time_us must align with the 20 ms formal grid")
        if self.level == 0:
            if self.stationary_position_xy is None or not self.contains(
                self.stationary_position_xy
            ):
                raise ValueError("stationary position lies outside path bounds")
            return
        required = (
            self.min_chord_length_m,
            self.max_chord_length_m,
            self.min_path_length_m,
            self.max_path_length_m,
            self.max_speed_mps,
            self.max_continuous_acceleration_mps2,
            self.curve_deviation_range_m,
            self.segment_count_three_probability,
            self.cubic_segment_probability,
            self.two_segment_change_time_us_range,
            self.three_segment_first_change_time_us_range,
            self.three_segment_second_change_time_us_range,
            self.turn_angle_degrees_range,
            self.velocity_jump_range_mps,
        )
        if any(value is None for value in required):
            raise ValueError("dynamic motion configuration is incomplete")
        if not 0 < self.min_chord_length_m < self.max_chord_length_m:
            raise ValueError("chord length range is invalid")
        if not 0 < self.min_path_length_m <= self.min_chord_length_m:
            raise ValueError("minimum path length is invalid")
        if not self.max_path_length_m >= self.max_chord_length_m:
            raise ValueError("maximum path length is invalid")
        if self.max_speed_mps <= 0 or self.max_continuous_acceleration_mps2 <= 0:
            raise ValueError("speed and acceleration limits must be positive")
        if not 0 < self.curve_deviation_range_m[0] < self.curve_deviation_range_m[1]:
            raise ValueError("curve deviation range is invalid")
        for name in ("segment_count_three_probability", "cubic_segment_probability"):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f"{name} must lie strictly inside (0, 1)")
        for name in (
            "two_segment_change_time_us_range",
            "three_segment_first_change_time_us_range",
            "three_segment_second_change_time_us_range",
        ):
            lower, upper = getattr(self, name)
            if not 0 < lower <= upper < self.anchor_time_us or lower % 20_000 or upper % 20_000:
                raise ValueError(f"{name} must be ordered on the formal grid")
        if self.three_segment_first_change_time_us_range[1] >= (
            self.three_segment_second_change_time_us_range[0]
        ):
            raise ValueError("three-segment transition windows must not overlap")
        if not 0 < self.turn_angle_degrees_range[0] < self.turn_angle_degrees_range[1] < 180:
            raise ValueError("turn angle range is invalid")
        if not 0 < self.velocity_jump_range_mps[0] < self.velocity_jump_range_mps[1]:
            raise ValueError("velocity jump range is invalid")

    def contains(self, position_xy: Any) -> bool:
        x_min, x_max, y_min, y_max = self.path_bounds_xy
        x, y = np.asarray(position_xy, dtype=float)
        return bool(x_min <= x <= x_max and y_min <= y <= y_max)


@dataclass(frozen=True)
class SharedTrajectoryGeometry:
    start_xy: np.ndarray
    end_xy: np.ndarray

    def __post_init__(self) -> None:
        for name in ("start_xy", "end_xy"):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            if value.shape != (2,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite 2D point")
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def sample_shared_geometry(*, config: MotionConfig, seed: int) -> SharedTrajectoryGeometry:
    if config.level == 0:
        raise ValueError("stationary L0 does not use dynamic shared geometry")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("motion seed must be a non-negative integer")
    if config.min_chord_length_m is None or config.max_chord_length_m is None:
        raise ValueError("dynamic chord bounds are missing")
    x_min, x_max, y_min, y_max = config.path_bounds_xy
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x47454F4D]))
    for _ in range(config.max_retries):
        start = rng.uniform([x_min, y_min], [x_max, y_max])
        end = rng.uniform([x_min, y_min], [x_max, y_max])
        chord_length = float(np.linalg.norm(end - start))
        if config.min_chord_length_m < chord_length < config.max_chord_length_m:
            return SharedTrajectoryGeometry(start_xy=start, end_xy=end)
    raise RuntimeError("failed to sample shared trajectory endpoints")


@dataclass(frozen=True)
class MotionSample:
    time_us: int
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    segment_index: int
    terminal: bool

    def __post_init__(self) -> None:
        for value in (self.position, self.velocity, self.acceleration):
            value.setflags(write=False)


class MotionProfile(Protocol):
    segment_count: int
    change_times_us: tuple[int, ...]

    def sample(self, time_us: int) -> MotionSample: ...

    def sample_side(self, time_us: int, *, side: Literal["left", "right"]) -> MotionSample: ...

    def to_mapping(self) -> dict[str, Any]: ...


def motion_sample_to_mapping(
    sample: MotionSample,
    *,
    motion_level: int,
) -> dict[str, np.ndarray]:
    return {
        "motion_level": np.array(motion_level, dtype=np.int64),
        "motion_terminal": np.array(sample.terminal, dtype=np.bool_),
        "segment_index": np.array(sample.segment_index, dtype=np.int64),
        "target_acceleration": sample.acceleration,
        "target_position": sample.position,
        "target_velocity": sample.velocity,
    }


@dataclass(frozen=True)
class DrivenBallWorld:
    """Apply an analytic motion profile to the task object before each MuJoCo ``step1``."""

    profile: MotionProfile
    motion_level: int

    def __post_init__(self) -> None:
        if self.motion_level not in range(4):
            raise ValueError("motion_level must be one of 0, 1, 2, 3")

    def __call__(self, env: Any, current_time_us: int) -> dict[str, np.ndarray]:
        sample = self.profile.sample(current_time_us)
        if sample.terminal:
            env.mark_motion_deadline(current_time_us)
        qpos = np.concatenate([sample.position, np.array([1.0, 0.0, 0.0, 0.0])])
        qvel = np.concatenate([sample.velocity, np.zeros(3)])
        env.sim.data.set_joint_qpos(env.task_object.joints[0], qpos)
        env.sim.data.set_joint_qvel(env.task_object.joints[0], qvel)
        return motion_sample_to_mapping(sample, motion_level=self.motion_level)


def _sample(
    time_us: int,
    position_xy: Any,
    velocity_xy: Any,
    acceleration_xy: Any,
    segment: int,
    z: float,
    *,
    terminal: bool = False,
) -> MotionSample:
    if isinstance(time_us, bool) or not isinstance(time_us, int) or time_us < 0:
        raise ValueError("time_us must be a non-negative integer")
    return MotionSample(
        time_us=time_us,
        position=np.array([*np.asarray(position_xy, dtype=float), z], dtype=float),
        velocity=np.array([*np.asarray(velocity_xy, dtype=float), 0.0], dtype=float),
        acceleration=np.array([*np.asarray(acceleration_xy, dtype=float), 0.0], dtype=float),
        segment_index=segment,
        terminal=terminal,
    )


@dataclass(frozen=True)
class StationaryProfile:
    position_xy: np.ndarray
    workspace_z: float
    segment_count: int = 1
    change_times_us: tuple[int, ...] = ()

    def sample(self, time_us: int) -> MotionSample:
        return _sample(time_us, self.position_xy, np.zeros(2), np.zeros(2), 0, self.workspace_z)

    def sample_side(self, time_us: int, *, side: Literal["left", "right"]) -> MotionSample:
        del side
        return self.sample(time_us)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "position_xy": self.position_xy.tolist(),
            "schema_version": 1,
            "type": "stationary",
            "workspace_z": self.workspace_z,
        }


@dataclass(frozen=True)
class ConstantVelocityProfile:
    start_xy: np.ndarray
    end_xy: np.ndarray
    velocity_xy: np.ndarray
    anchor_time_us: int
    workspace_z: float
    segment_count: int = 1
    change_times_us: tuple[int, ...] = ()

    def sample(self, time_us: int) -> MotionSample:
        if time_us <= self.anchor_time_us:
            seconds = time_us / 1_000_000
            position = self.start_xy + self.velocity_xy * seconds
            velocity = self.velocity_xy
            if time_us == self.anchor_time_us:
                position = self.end_xy
        else:
            position = self.end_xy
            velocity = np.zeros(2)
        return _sample(
            time_us,
            position,
            velocity,
            np.zeros(2),
            0,
            self.workspace_z,
            terminal=time_us >= self.anchor_time_us,
        )

    def sample_side(self, time_us: int, *, side: Literal["left", "right"]) -> MotionSample:
        del side
        return self.sample(time_us)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "anchor_time_us": self.anchor_time_us,
            "end_xy": self.end_xy.tolist(),
            "schema_version": 1,
            "start_xy": self.start_xy.tolist(),
            "type": "constant_velocity",
            "velocity_xy": self.velocity_xy.tolist(),
            "workspace_z": self.workspace_z,
        }


@dataclass(frozen=True)
class CubicPolynomialProfile:
    waypoint_times_us: np.ndarray
    waypoint_positions_xy: np.ndarray
    coefficients_xy: np.ndarray
    anchor_time_us: int
    workspace_z: float
    segment_count: int = 1
    change_times_us: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        times = np.array(self.waypoint_times_us, dtype=np.int64, copy=True)
        positions = np.array(self.waypoint_positions_xy, dtype=float, copy=True)
        coefficients = np.array(self.coefficients_xy, dtype=float, copy=True)
        expected_times = np.linspace(0, self.anchor_time_us, 4, dtype=np.int64)
        if times.shape != (4,) or not np.array_equal(times, expected_times):
            raise ValueError("cubic waypoint times must divide the anchor duration into thirds")
        if positions.shape != (4, 2) or coefficients.shape != (4, 2):
            raise ValueError("cubic positions and coefficients have invalid shapes")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(coefficients)):
            raise ValueError("cubic positions and coefficients must be finite")
        for value in (times, positions, coefficients):
            value.setflags(write=False)
        object.__setattr__(self, "waypoint_times_us", times)
        object.__setattr__(self, "waypoint_positions_xy", positions)
        object.__setattr__(self, "coefficients_xy", coefficients)

    def sample(self, time_us: int) -> MotionSample:
        if time_us <= self.anchor_time_us:
            u = np.clip(time_us / self.anchor_time_us, 0.0, 1.0)
            c0, c1, c2, c3 = self.coefficients_xy
            position = c0 + c1 * u + c2 * u**2 + c3 * u**3
            duration_s = self.anchor_time_us / 1_000_000
            velocity = (c1 + 2 * c2 * u + 3 * c3 * u**2) / duration_s
            acceleration = (2 * c2 + 6 * c3 * u) / duration_s**2
            if time_us == 0:
                position = self.waypoint_positions_xy[0]
            elif time_us == self.anchor_time_us:
                position = self.waypoint_positions_xy[-1]
        else:
            position = self.waypoint_positions_xy[-1]
            velocity = np.zeros(2)
            acceleration = np.zeros(2)
        return _sample(
            time_us,
            position,
            velocity,
            acceleration,
            0,
            self.workspace_z,
            terminal=time_us >= self.anchor_time_us,
        )

    def sample_side(self, time_us: int, *, side: Literal["left", "right"]) -> MotionSample:
        del side
        return self.sample(time_us)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "anchor_time_us": self.anchor_time_us,
            "coefficients_xy": self.coefficients_xy.tolist(),
            "schema_version": 2,
            "type": "cubic_polynomial",
            "waypoint_positions_xy": self.waypoint_positions_xy.tolist(),
            "waypoint_times_us": self.waypoint_times_us.tolist(),
            "workspace_z": self.workspace_z,
        }

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> CubicPolynomialProfile:
        expected = {
            "anchor_time_us",
            "coefficients_xy",
            "schema_version",
            "type",
            "waypoint_positions_xy",
            "waypoint_times_us",
            "workspace_z",
        }
        if set(raw) != expected or raw["schema_version"] != 2:
            raise ValueError("cubic polynomial profile fields are invalid")
        anchor_time_us = raw["anchor_time_us"]
        if isinstance(anchor_time_us, bool) or not isinstance(anchor_time_us, int):
            raise ValueError("cubic polynomial anchor_time_us must be an integer")
        return cls(
            waypoint_times_us=np.asarray(raw["waypoint_times_us"]),
            waypoint_positions_xy=np.asarray(raw["waypoint_positions_xy"]),
            coefficients_xy=np.asarray(raw["coefficients_xy"]),
            anchor_time_us=anchor_time_us,
            workspace_z=float(raw["workspace_z"]),
        )


@dataclass(frozen=True)
class PolynomialSegment:
    kind: Literal["line", "cubic"]
    start_xy: np.ndarray
    end_xy: np.ndarray
    start_velocity_xy: np.ndarray
    end_velocity_xy: np.ndarray
    duration_us: int

    def __post_init__(self) -> None:
        if self.kind not in {"line", "cubic"}:
            raise ValueError("polynomial segment kind must be line or cubic")
        if (
            isinstance(self.duration_us, bool)
            or not isinstance(self.duration_us, int)
            or self.duration_us <= 0
            or self.duration_us % 20_000
        ):
            raise ValueError(
                "polynomial segment duration must be a positive 20 ms formal-grid time"
            )
        for name in ("start_xy", "end_xy", "start_velocity_xy", "end_velocity_xy"):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            if value.shape != (2,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite 2D vector")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if self.kind == "line":
            expected = (self.end_xy - self.start_xy) / (self.duration_us / 1_000_000)
            if not np.allclose(self.start_velocity_xy, expected, atol=1e-12, rtol=0) or not (
                np.allclose(self.end_velocity_xy, expected, atol=1e-12, rtol=0)
            ):
                raise ValueError("line segment velocities must produce exact linear motion")

    def evaluate(self, elapsed_us: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        duration_s = self.duration_us / 1_000_000
        u = np.clip(elapsed_us / self.duration_us, 0.0, 1.0)
        p0, p1 = self.start_xy, self.end_xy
        v0, v1 = self.start_velocity_xy, self.end_velocity_xy
        h00 = 2 * u**3 - 3 * u**2 + 1
        h10 = u**3 - 2 * u**2 + u
        h01 = -2 * u**3 + 3 * u**2
        h11 = u**3 - u**2
        position = h00 * p0 + h10 * duration_s * v0 + h01 * p1 + h11 * duration_s * v1
        derivative_u = (
            (6 * u**2 - 6 * u) * p0
            + (3 * u**2 - 4 * u + 1) * duration_s * v0
            + (-6 * u**2 + 6 * u) * p1
            + (3 * u**2 - 2 * u) * duration_s * v1
        )
        second_u = (
            (12 * u - 6) * p0
            + (6 * u - 4) * duration_s * v0
            + (-12 * u + 6) * p1
            + (6 * u - 2) * duration_s * v1
        )
        if elapsed_us == 0:
            position = self.start_xy
        elif elapsed_us == self.duration_us:
            position = self.end_xy
        return position, derivative_u / duration_s, second_u / duration_s**2

    def to_mapping(self) -> dict[str, Any]:
        return {
            "duration_us": self.duration_us,
            "end_velocity_xy": self.end_velocity_xy.tolist(),
            "end_xy": self.end_xy.tolist(),
            "kind": self.kind,
            "start_velocity_xy": self.start_velocity_xy.tolist(),
            "start_xy": self.start_xy.tolist(),
        }

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> PolynomialSegment:
        expected = {
            "duration_us",
            "end_velocity_xy",
            "end_xy",
            "kind",
            "start_velocity_xy",
            "start_xy",
        }
        if set(raw) != expected:
            raise ValueError("polynomial segment mapping fields are invalid")
        return cls(
            kind=raw["kind"],
            start_xy=np.asarray(raw["start_xy"]),
            end_xy=np.asarray(raw["end_xy"]),
            start_velocity_xy=np.asarray(raw["start_velocity_xy"]),
            end_velocity_xy=np.asarray(raw["end_velocity_xy"]),
            duration_us=raw["duration_us"],
        )


@dataclass(frozen=True)
class PiecewisePolynomialProfile:
    segments: tuple[PolynomialSegment, ...]
    workspace_z: float
    change_times_us: tuple[int, ...]

    def __post_init__(self) -> None:
        if not 2 <= len(self.segments) <= 3:
            raise ValueError("piecewise polynomial profile requires two or three segments")
        expected_changes = tuple(
            sum(segment.duration_us for segment in self.segments[: index + 1])
            for index in range(len(self.segments) - 1)
        )
        if self.change_times_us != expected_changes:
            raise ValueError("piecewise polynomial change times do not match segment durations")

    @property
    def segment_count(self) -> int:
        return len(self.segments)

    @property
    def anchor_time_us(self) -> int:
        return sum(segment.duration_us for segment in self.segments)

    def sample(self, time_us: int) -> MotionSample:
        return self.sample_side(time_us, side="right")

    def sample_side(self, time_us: int, *, side: Literal["left", "right"]) -> MotionSample:
        if isinstance(time_us, bool) or not isinstance(time_us, int) or time_us < 0:
            raise ValueError("time_us must be a non-negative integer")
        if time_us > self.anchor_time_us:
            return _sample(
                time_us,
                self.segments[-1].end_xy,
                np.zeros(2),
                np.zeros(2),
                len(self.segments) - 1,
                self.workspace_z,
                terminal=True,
            )
        start_us = 0
        for index, segment in enumerate(self.segments):
            end_us = start_us + segment.duration_us
            if time_us < end_us or (
                time_us == end_us and (side == "left" or index == len(self.segments) - 1)
            ):
                position, velocity, acceleration = segment.evaluate(time_us - start_us)
                return _sample(
                    time_us,
                    position,
                    velocity,
                    acceleration,
                    index,
                    self.workspace_z,
                    terminal=time_us >= self.anchor_time_us,
                )
            start_us = end_us
        raise RuntimeError("piecewise polynomial segment lookup failed")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "segments": [segment.to_mapping() for segment in self.segments],
            "type": "piecewise_polynomial",
            "workspace_z": self.workspace_z,
        }

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> PiecewisePolynomialProfile:
        if set(raw) != {"schema_version", "segments", "type", "workspace_z"} or raw[
            "schema_version"
        ] != 2:
            raise ValueError("piecewise polynomial profile fields are invalid")
        if not isinstance(raw["segments"], list) or not 2 <= len(raw["segments"]) <= 3:
            raise ValueError("piecewise polynomial profile requires two or three segments")
        segments = tuple(PolynomialSegment.from_mapping(segment) for segment in raw["segments"])
        changes = tuple(
            sum(segment.duration_us for segment in segments[: index + 1])
            for index in range(len(segments) - 1)
        )
        return cls(
            segments=segments,
            workspace_z=float(raw["workspace_z"]),
            change_times_us=changes,
        )

def load_motion_config(path: Path) -> MotionConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("motion config must be a YAML mapping")
    return MotionConfig.from_mapping(raw)


def motion_profile_from_mapping(raw: dict[str, Any], *, config: MotionConfig) -> MotionProfile:
    if not isinstance(raw, dict) or raw.get("schema_version") not in {1, 2}:
        raise ValueError("motion profile mapping has an unsupported schema_version")
    profile_type = raw.get("type")
    workspace_z = raw.get("workspace_z")
    if (
        isinstance(workspace_z, bool)
        or not isinstance(workspace_z, (int, float))
        or not np.isfinite(workspace_z)
    ):
        raise ValueError("motion profile workspace_z must be finite and numeric")
    if profile_type == "stationary":
        if set(raw) != {"position_xy", "schema_version", "type", "workspace_z"}:
            raise ValueError("stationary motion profile fields are invalid")
        profile: MotionProfile = StationaryProfile(
            np.asarray(_tuple(raw["position_xy"], name="position_xy", length=2)),
            float(workspace_z),
        )
    elif profile_type == "constant_velocity":
        if set(raw) != {
            "anchor_time_us",
            "end_xy",
            "schema_version",
            "start_xy",
            "type",
            "velocity_xy",
            "workspace_z",
        }:
            raise ValueError("constant-velocity motion profile fields are invalid")
        anchor_time_us = raw["anchor_time_us"]
        if isinstance(anchor_time_us, bool) or not isinstance(anchor_time_us, int):
            raise ValueError("constant-velocity anchor_time_us must be an integer")
        profile = ConstantVelocityProfile(
            np.asarray(_tuple(raw["start_xy"], name="start_xy", length=2)),
            np.asarray(_tuple(raw["end_xy"], name="end_xy", length=2)),
            np.asarray(_tuple(raw["velocity_xy"], name="velocity_xy", length=2)),
            anchor_time_us,
            float(workspace_z),
        )
    elif profile_type == "cubic_polynomial":
        profile = CubicPolynomialProfile.from_mapping(raw)
    elif profile_type == "piecewise_polynomial":
        profile = PiecewisePolynomialProfile.from_mapping(raw)
    else:
        raise ValueError(f"unknown motion profile type: {profile_type}")
    _validate_profile_contract(profile, config)
    return profile


def _profile_is_bounded(profile: MotionProfile, config: MotionConfig) -> bool:
    for time_us in range(0, config.anchor_time_us + 1, 2_000):
        sample = profile.sample(time_us)
        if not config.contains(sample.position[:2]):
            return False
        if config.level == 0:
            continue
        if config.max_speed_mps is None or config.max_continuous_acceleration_mps2 is None:
            raise ValueError("dynamic motion limits are missing")
        if np.linalg.norm(sample.velocity[:2]) > config.max_speed_mps + 1e-12:
            return False
        if (
            np.linalg.norm(sample.acceleration[:2])
            > config.max_continuous_acceleration_mps2 + 1e-12
        ):
            return False
    return True


def _path_metrics(profile: MotionProfile, config: MotionConfig) -> tuple[float, float]:
    positions = np.stack(
        [
            profile.sample(time_us).position[:2]
            for time_us in range(0, config.anchor_time_us + 1, 2_000)
        ]
    )
    path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    chord = positions[-1] - positions[0]
    chord_length = float(np.linalg.norm(chord))
    if chord_length <= 0:
        raise ValueError("dynamic trajectory chord must be non-zero")
    relative = positions - positions[0]
    deviations = np.abs(chord[0] * relative[:, 1] - chord[1] * relative[:, 0])
    return path_length, float(deviations.max() / chord_length)


def _validate_profile_contract(profile: MotionProfile, config: MotionConfig) -> None:
    expected_types = {
        0: StationaryProfile,
        1: ConstantVelocityProfile,
        2: CubicPolynomialProfile,
        3: PiecewisePolynomialProfile,
    }
    if not isinstance(profile, expected_types[config.level]):
        raise ValueError("motion profile type does not match configured level")
    if isinstance(profile, StationaryProfile):
        if config.stationary_position_xy is None or not np.array_equal(
            profile.position_xy, np.asarray(config.stationary_position_xy)
        ):
            raise ValueError("stationary profile does not match configured position")
    elif isinstance(profile, ConstantVelocityProfile):
        if profile.anchor_time_us != config.anchor_time_us:
            raise ValueError("constant-velocity anchor time does not match config")
        expected_velocity = (profile.end_xy - profile.start_xy) / (
            profile.anchor_time_us / 1_000_000
        )
        if not np.allclose(profile.velocity_xy, expected_velocity, atol=1e-12, rtol=0):
            raise ValueError("constant-velocity endpoint and velocity are inconsistent")
    elif isinstance(profile, CubicPolynomialProfile):
        if profile.anchor_time_us != config.anchor_time_us:
            raise ValueError("cubic polynomial duration does not match config")
        if np.linalg.norm(profile.coefficients_xy[3]) <= 1e-4:
            raise ValueError("Level 2 polynomial must retain a non-trivial cubic term")
    elif isinstance(profile, PiecewisePolynomialProfile):
        if profile.anchor_time_us != config.anchor_time_us:
            raise ValueError("piecewise polynomial duration does not match config")
        if config.velocity_jump_range_mps is None or config.turn_angle_degrees_range is None:
            raise ValueError("Level 3 transition constraints are missing")
        for left, right in zip(profile.segments, profile.segments[1:]):
            if not np.allclose(left.end_xy, right.start_xy, atol=1e-12, rtol=0):
                raise ValueError("piecewise polynomial position continuity is violated")
            jump = float(np.linalg.norm(right.start_velocity_xy - left.end_velocity_xy))
            if not config.velocity_jump_range_mps[0] <= jump <= config.velocity_jump_range_mps[1]:
                raise ValueError("piecewise polynomial velocity jump is outside configured range")
            cosine = np.clip(
                np.dot(left.end_velocity_xy, right.start_velocity_xy)
                / (
                    np.linalg.norm(left.end_velocity_xy)
                    * np.linalg.norm(right.start_velocity_xy)
                ),
                -1.0,
                1.0,
            )
            angle = float(np.degrees(np.arccos(cosine)))
            if not config.turn_angle_degrees_range[0] <= angle <= (
                config.turn_angle_degrees_range[1]
            ):
                raise ValueError("piecewise polynomial turn angle is outside configured range")
        if profile.segment_count == 2:
            if config.two_segment_change_time_us_range is None or not (
                config.two_segment_change_time_us_range[0]
                <= profile.change_times_us[0]
                <= config.two_segment_change_time_us_range[1]
            ):
                raise ValueError("two-segment change time is outside configured window")
        else:
            first_range = config.three_segment_first_change_time_us_range
            second_range = config.three_segment_second_change_time_us_range
            if first_range is None or second_range is None or not (
                first_range[0] <= profile.change_times_us[0] <= first_range[1]
                and second_range[0] <= profile.change_times_us[1] <= second_range[1]
            ):
                raise ValueError("three-segment change times are outside configured windows")
    if not _profile_is_bounded(profile, config):
        raise ValueError("motion profile violates configured physical contract")
    if config.level:
        path_length, deviation = _path_metrics(profile, config)
        if config.min_path_length_m is None or config.max_path_length_m is None:
            raise ValueError("dynamic path length bounds are missing")
        if not config.min_path_length_m <= path_length <= config.max_path_length_m:
            raise ValueError("motion profile path length is outside configured bounds")
        if config.level in {2, 3}:
            if config.curve_deviation_range_m is None or not (
                config.curve_deviation_range_m[0]
                <= deviation
                <= config.curve_deviation_range_m[1]
            ):
                raise ValueError(
                    f"Level {config.level} curve deviation is outside configured bounds"
                )


def _build_level1(
    config: MotionConfig,
    geometry: SharedTrajectoryGeometry,
    workspace_z: float,
) -> MotionProfile:
    duration_s = config.anchor_time_us / 1_000_000
    velocity = (geometry.end_xy - geometry.start_xy) / duration_s
    profile = ConstantVelocityProfile(
        geometry.start_xy,
        geometry.end_xy,
        velocity,
        config.anchor_time_us,
        workspace_z,
    )
    _validate_profile_contract(profile, config)
    return profile


def _cubic_from_waypoints(
    *,
    waypoints_xy: np.ndarray,
    anchor_time_us: int,
    workspace_z: float,
) -> CubicPolynomialProfile:
    parameters = np.array([0.0, 1 / 3, 2 / 3, 1.0])
    vandermonde = np.stack(
        [np.ones(4), parameters, parameters**2, parameters**3], axis=1
    )
    coefficients = np.linalg.solve(vandermonde, waypoints_xy)
    return CubicPolynomialProfile(
        waypoint_times_us=np.array(
            [0, anchor_time_us // 3, 2 * anchor_time_us // 3, anchor_time_us]
        ),
        waypoint_positions_xy=waypoints_xy,
        coefficients_xy=coefficients,
        anchor_time_us=anchor_time_us,
        workspace_z=workspace_z,
    )


def _build_level2(
    config: MotionConfig,
    rng: np.random.Generator,
    geometry: SharedTrajectoryGeometry,
    workspace_z: float,
) -> MotionProfile:
    if config.curve_deviation_range_m is None:
        raise ValueError("Level 2 curve deviation range is missing")
    chord = geometry.end_xy - geometry.start_xy
    direction = chord / np.linalg.norm(chord)
    perpendicular = np.array([-direction[1], direction[0]])
    for _ in range(config.max_retries):
        curve_sign = -1.0 if int(rng.integers(0, 2)) == 0 else 1.0
        first_offset = curve_sign * rng.uniform(*config.curve_deviation_range_m)
        second_offset = curve_sign * rng.uniform(*config.curve_deviation_range_m)
        waypoints = np.stack(
            [
                geometry.start_xy,
                geometry.start_xy + chord / 3 + perpendicular * first_offset,
                geometry.start_xy + 2 * chord / 3 + perpendicular * second_offset,
                geometry.end_xy,
            ]
        )
        profile = _cubic_from_waypoints(
            waypoints_xy=waypoints,
            anchor_time_us=config.anchor_time_us,
            workspace_z=workspace_z,
        )
        try:
            _validate_profile_contract(profile, config)
        except ValueError:
            continue
        return profile
    raise RuntimeError("failed to generate a valid Level 2 motion profile")


def _sample_grid_time(
    rng: np.random.Generator,
    bounds_us: tuple[int, int],
) -> int:
    lower_tick = bounds_us[0] // 20_000
    upper_tick = bounds_us[1] // 20_000
    return int(rng.integers(lower_tick, upper_tick + 1)) * 20_000


def _build_polynomial_segment(
    *,
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    duration_us: int,
    kind: Literal["line", "cubic"],
    rng: np.random.Generator,
) -> PolynomialSegment:
    average_velocity = (end_xy - start_xy) / (duration_us / 1_000_000)
    if kind == "line":
        start_velocity = average_velocity
        end_velocity = average_velocity
    else:
        speed = float(np.linalg.norm(average_velocity))
        direction = average_velocity / speed
        perpendicular = np.array([-direction[1], direction[0]])
        lateral = rng.choice([-1.0, 1.0]) * rng.uniform(0.15, 0.35) * speed
        start_velocity = average_velocity + perpendicular * lateral
        end_velocity = average_velocity - perpendicular * lateral
    return PolynomialSegment(
        kind=kind,
        start_xy=start_xy,
        end_xy=end_xy,
        start_velocity_xy=start_velocity,
        end_velocity_xy=end_velocity,
        duration_us=duration_us,
    )


def _build_level3(
    config: MotionConfig,
    rng: np.random.Generator,
    geometry: SharedTrajectoryGeometry,
    workspace_z: float,
) -> MotionProfile:
    required = (
        config.segment_count_three_probability,
        config.cubic_segment_probability,
        config.curve_deviation_range_m,
        config.two_segment_change_time_us_range,
        config.three_segment_first_change_time_us_range,
        config.three_segment_second_change_time_us_range,
    )
    if any(value is None for value in required):
        raise ValueError("Level 3 sampling configuration is incomplete")
    count = 3 if rng.random() < config.segment_count_three_probability else 2
    if count == 2:
        change_times = (_sample_grid_time(rng, config.two_segment_change_time_us_range),)
    else:
        change_times = (
            _sample_grid_time(rng, config.three_segment_first_change_time_us_range),
            _sample_grid_time(rng, config.three_segment_second_change_time_us_range),
        )
    boundaries = (0, *change_times, config.anchor_time_us)
    durations = tuple(
        boundaries[index + 1] - boundaries[index] for index in range(len(boundaries) - 1)
    )
    chord = geometry.end_xy - geometry.start_xy
    direction = chord / np.linalg.norm(chord)
    perpendicular = np.array([-direction[1], direction[0]])
    for _ in range(config.max_retries):
        sign = -1.0 if int(rng.integers(0, 2)) == 0 else 1.0
        if count == 2:
            fraction = change_times[0] / config.anchor_time_us
            offset = sign * rng.uniform(*config.curve_deviation_range_m)
            waypoints = [
                geometry.start_xy,
                geometry.start_xy + fraction * chord + perpendicular * offset,
                geometry.end_xy,
            ]
        else:
            first_fraction = change_times[0] / config.anchor_time_us
            second_fraction = change_times[1] / config.anchor_time_us
            first_offset = sign * rng.uniform(*config.curve_deviation_range_m)
            second_sign = -sign if rng.random() < 0.75 else sign
            second_offset = second_sign * rng.uniform(*config.curve_deviation_range_m)
            waypoints = [
                geometry.start_xy,
                geometry.start_xy + first_fraction * chord + perpendicular * first_offset,
                geometry.start_xy + second_fraction * chord + perpendicular * second_offset,
                geometry.end_xy,
            ]
        kinds = tuple(
            "cubic" if rng.random() < config.cubic_segment_probability else "line"
            for _ in range(count)
        )
        segments = tuple(
            _build_polynomial_segment(
                start_xy=np.asarray(waypoints[index]),
                end_xy=np.asarray(waypoints[index + 1]),
                duration_us=durations[index],
                kind=kinds[index],
                rng=rng,
            )
            for index in range(count)
        )
        profile = PiecewisePolynomialProfile(segments, workspace_z, change_times)
        try:
            _validate_profile_contract(profile, config)
        except ValueError:
            continue
        return profile
    raise RuntimeError("failed to generate a valid Level 3 motion profile")


def build_motion_profile(*, config: MotionConfig, seed: int, workspace_z: float) -> MotionProfile:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("motion seed must be a non-negative integer")
    if not np.isfinite(workspace_z):
        raise ValueError("workspace_z must be finite")
    if config.level == 0:
        if config.stationary_position_xy is None:
            raise ValueError("stationary position is missing")
        return StationaryProfile(np.asarray(config.stationary_position_xy), workspace_z)
    geometry = sample_shared_geometry(config=config, seed=seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed, config.level, 0x44594E41]))
    if config.level == 1:
        return _build_level1(config, geometry, workspace_z)
    if config.level == 2:
        return _build_level2(config, rng, geometry, workspace_z)
    return _build_level3(config, rng, geometry, workspace_z)
