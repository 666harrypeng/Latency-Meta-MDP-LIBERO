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
    center_bounds_xy: tuple[float, float, float, float]
    anchor_position_xy: tuple[float, float]
    anchor_time_us: int
    min_speed_mps: float
    max_speed_mps: float
    max_continuous_acceleration_mps2: float
    curve_offset_max_m: float
    segment_count_range: tuple[int, int]
    min_segment_duration_us: int
    velocity_jump_range_mps: tuple[float, float]
    max_retries: int

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> MotionConfig:
        expected = {
            "anchor_position_xy",
            "anchor_time_us",
            "center_bounds_xy",
            "curve_offset_max_m",
            "level",
            "max_continuous_acceleration_mps2",
            "max_retries",
            "max_speed_mps",
            "min_segment_duration_us",
            "min_speed_mps",
            "profile_id",
            "schema_version",
            "segment_count_range",
            "velocity_jump_range_mps",
        }
        unknown = sorted(set(raw) - expected)
        missing = sorted(expected - set(raw))
        if unknown:
            raise ValueError(f"unknown motion config fields: {unknown}")
        if missing:
            raise ValueError(f"missing motion config fields: {missing}")
        if raw["schema_version"] != 1:
            raise ValueError("motion schema_version must be 1")
        level = raw["level"]
        if isinstance(level, bool) or not isinstance(level, int) or level not in range(4):
            raise ValueError("motion level must be one of 0, 1, 2, 3")
        expected_profile_id = (
            "dynamic_grasp_lift_l0"
            if level == 0
            else f"dynamic_grasp_lift_l{level}_v2"
        )
        if raw["profile_id"] != expected_profile_id:
            raise ValueError("motion profile_id does not match level")
        integer_names = ("anchor_time_us", "min_segment_duration_us", "max_retries")
        integers: dict[str, int] = {}
        for name in integer_names:
            value = raw[name]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            integers[name] = value
        scalar_names = (
            "curve_offset_max_m",
            "max_continuous_acceleration_mps2",
            "max_speed_mps",
            "min_speed_mps",
        )
        scalars: dict[str, float] = {}
        for name in scalar_names:
            value = raw[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name} must be positive")
            scalars[name] = float(value)
            if not np.isfinite(scalars[name]):
                raise ValueError(f"{name} must be finite")
        config = cls(
            schema_version=1,
            profile_id=raw["profile_id"],
            level=level,
            center_bounds_xy=_tuple(raw["center_bounds_xy"], name="center_bounds_xy", length=4),
            anchor_position_xy=_tuple(
                raw["anchor_position_xy"], name="anchor_position_xy", length=2
            ),
            anchor_time_us=integers["anchor_time_us"],
            min_speed_mps=scalars["min_speed_mps"],
            max_speed_mps=scalars["max_speed_mps"],
            max_continuous_acceleration_mps2=scalars["max_continuous_acceleration_mps2"],
            curve_offset_max_m=scalars["curve_offset_max_m"],
            segment_count_range=_tuple(
                raw["segment_count_range"],
                name="segment_count_range",
                length=2,
                integer=True,
            ),
            min_segment_duration_us=integers["min_segment_duration_us"],
            velocity_jump_range_mps=_tuple(
                raw["velocity_jump_range_mps"],
                name="velocity_jump_range_mps",
                length=2,
            ),
            max_retries=integers["max_retries"],
        )
        config.validate()
        return config

    def validate(self) -> None:
        x_min, x_max, y_min, y_max = self.center_bounds_xy
        if not x_min < x_max or not y_min < y_max:
            raise ValueError("center bounds must be ordered")
        if not self.contains(self.anchor_position_xy):
            raise ValueError("anchor position lies outside center bounds")
        if self.anchor_time_us % 20_000:
            raise ValueError("anchor_time_us must align with the 20 ms formal grid")
        if self.min_segment_duration_us % 20_000:
            raise ValueError("min_segment_duration_us must align with the 20 ms formal grid")
        if not 0 < self.min_speed_mps <= self.max_speed_mps:
            raise ValueError("speed range is invalid")
        if self.segment_count_range[0] < 2 or self.segment_count_range[1] > 3:
            raise ValueError("segment count range must stay within 2-3")
        if self.segment_count_range[0] > self.segment_count_range[1]:
            raise ValueError("segment count range is invalid")
        if self.min_segment_duration_us * self.segment_count_range[1] > self.anchor_time_us:
            raise ValueError("minimum segment duration cannot fit anchor duration")
        if not 0 < self.velocity_jump_range_mps[0] <= self.velocity_jump_range_mps[1]:
            raise ValueError("velocity jump range is invalid")

    def contains(self, position_xy: Any) -> bool:
        x_min, x_max, y_min, y_max = self.center_bounds_xy
        x, y = np.asarray(position_xy, dtype=float)
        return bool(x_min <= x <= x_max and y_min <= y <= y_max)


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
        else:
            seconds = self.anchor_time_us / 1_000_000
            position = self.start_xy + self.velocity_xy * seconds
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
            "schema_version": 1,
            "start_xy": self.start_xy.tolist(),
            "type": "constant_velocity",
            "velocity_xy": self.velocity_xy.tolist(),
            "workspace_z": self.workspace_z,
        }


@dataclass(frozen=True)
class HermiteSegment:
    start_xy: np.ndarray
    end_xy: np.ndarray
    start_velocity_xy: np.ndarray
    end_velocity_xy: np.ndarray
    duration_us: int

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
        return position, derivative_u / duration_s, second_u / duration_s**2

    def to_mapping(self) -> dict[str, Any]:
        return {
            "duration_us": self.duration_us,
            "end_velocity_xy": self.end_velocity_xy.tolist(),
            "end_xy": self.end_xy.tolist(),
            "start_velocity_xy": self.start_velocity_xy.tolist(),
            "start_xy": self.start_xy.tolist(),
        }

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> HermiteSegment:
        if set(raw) != {
            "duration_us",
            "end_velocity_xy",
            "end_xy",
            "start_velocity_xy",
            "start_xy",
        }:
            raise ValueError("Hermite segment mapping fields are invalid")
        duration_us = raw["duration_us"]
        if isinstance(duration_us, bool) or not isinstance(duration_us, int) or duration_us <= 0:
            raise ValueError("Hermite segment duration_us must be positive")
        return cls(
            start_xy=np.asarray(_tuple(raw["start_xy"], name="start_xy", length=2)),
            end_xy=np.asarray(_tuple(raw["end_xy"], name="end_xy", length=2)),
            start_velocity_xy=np.asarray(
                _tuple(raw["start_velocity_xy"], name="start_velocity_xy", length=2)
            ),
            end_velocity_xy=np.asarray(
                _tuple(raw["end_velocity_xy"], name="end_velocity_xy", length=2)
            ),
            duration_us=duration_us,
        )


@dataclass(frozen=True)
class HermiteProfile:
    segment: HermiteSegment
    workspace_z: float
    segment_count: int = 1
    change_times_us: tuple[int, ...] = ()

    def sample(self, time_us: int) -> MotionSample:
        if time_us <= self.segment.duration_us:
            position, velocity, acceleration = self.segment.evaluate(time_us)
        else:
            position = self.segment.end_xy
            velocity = np.zeros(2)
            acceleration = np.zeros(2)
        return _sample(
            time_us,
            position,
            velocity,
            acceleration,
            0,
            self.workspace_z,
            terminal=time_us >= self.segment.duration_us,
        )

    def sample_side(self, time_us: int, *, side: Literal["left", "right"]) -> MotionSample:
        del side
        return self.sample(time_us)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "segment": self.segment.to_mapping(),
            "type": "hermite",
            "workspace_z": self.workspace_z,
        }


@dataclass(frozen=True)
class PiecewiseHermiteProfile:
    segments: tuple[HermiteSegment, ...]
    workspace_z: float
    change_times_us: tuple[int, ...]

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
            last = self.segments[-1]
            return _sample(
                time_us,
                last.end_xy,
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
        raise RuntimeError("motion segment lookup failed")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "segments": [segment.to_mapping() for segment in self.segments],
            "type": "piecewise_hermite",
            "workspace_z": self.workspace_z,
        }


def load_motion_config(path: Path) -> MotionConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("motion config must be a YAML mapping")
    return MotionConfig.from_mapping(raw)


def motion_profile_from_mapping(raw: dict[str, Any], *, config: MotionConfig) -> MotionProfile:
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("motion profile mapping must use schema_version 1")
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
            np.asarray(_tuple(raw["velocity_xy"], name="velocity_xy", length=2)),
            anchor_time_us,
            float(workspace_z),
        )
    elif profile_type == "hermite":
        if set(raw) != {"schema_version", "segment", "type", "workspace_z"}:
            raise ValueError("Hermite motion profile fields are invalid")
        profile = HermiteProfile(HermiteSegment.from_mapping(raw["segment"]), float(workspace_z))
    elif profile_type == "piecewise_hermite":
        if set(raw) != {"schema_version", "segments", "type", "workspace_z"}:
            raise ValueError("piecewise-Hermite motion profile fields are invalid")
        if not isinstance(raw["segments"], list) or not 2 <= len(raw["segments"]) <= 3:
            raise ValueError("piecewise-Hermite profile requires two or three segments")
        segments = tuple(HermiteSegment.from_mapping(segment) for segment in raw["segments"])
        change_times = tuple(
            sum(segment.duration_us for segment in segments[: index + 1])
            for index in range(len(segments) - 1)
        )
        profile = PiecewiseHermiteProfile(segments, float(workspace_z), change_times)
    else:
        raise ValueError(f"unknown motion profile type: {profile_type}")
    _validate_profile_contract(profile, config)
    return profile


def _sample_speed(
    rng: np.random.Generator, config: MotionConfig, *, upper_scale: float = 1.0
) -> float:
    upper = config.max_speed_mps * upper_scale
    mean = 0.8 * upper
    sigma = upper / 3.0
    for _ in range(config.max_retries):
        candidate = abs(float(rng.normal(mean, sigma)))
        if config.min_speed_mps < candidate < upper:
            return candidate
    raise RuntimeError("failed to sample a speed inside the configured open interval")


def _profile_is_bounded(profile: MotionProfile, config: MotionConfig) -> bool:
    for time_us in range(0, config.anchor_time_us + 1, 2_000):
        sample = profile.sample(time_us)
        if not config.contains(sample.position[:2]):
            return False
        if np.linalg.norm(sample.velocity[:2]) > config.max_speed_mps + 1e-12:
            return False
        if (
            np.linalg.norm(sample.acceleration[:2])
            > config.max_continuous_acceleration_mps2 + 1e-12
        ):
            return False
    return True


def _validate_profile_contract(profile: MotionProfile, config: MotionConfig) -> None:
    expected_types = {
        0: StationaryProfile,
        1: ConstantVelocityProfile,
        2: HermiteProfile,
        3: PiecewiseHermiteProfile,
    }
    if not isinstance(profile, expected_types[config.level]):
        raise ValueError("motion profile type does not match configured level")
    if isinstance(profile, StationaryProfile):
        if not np.array_equal(profile.position_xy, np.asarray(config.anchor_position_xy)):
            raise ValueError("stationary profile does not match configured anchor")
    elif isinstance(profile, ConstantVelocityProfile):
        if profile.anchor_time_us != config.anchor_time_us:
            raise ValueError("constant-velocity anchor time does not match config")
        if np.linalg.norm(profile.velocity_xy) < config.min_speed_mps:
            raise ValueError("constant-velocity profile is below configured minimum speed")
    elif isinstance(profile, HermiteProfile):
        if profile.segment.duration_us != config.anchor_time_us:
            raise ValueError("Hermite duration does not match configured anchor time")
        if profile.segment.duration_us % 20_000:
            raise ValueError("Hermite duration must align with the 20 ms grid")
    elif isinstance(profile, PiecewiseHermiteProfile):
        for segment in profile.segments:
            if segment.duration_us % 20_000:
                raise ValueError(
                    "piecewise-Hermite segment duration must align with the 20 ms grid"
                )
            if segment.duration_us < config.min_segment_duration_us:
                raise ValueError("piecewise-Hermite segment is shorter than configured minimum")
        if profile.anchor_time_us != config.anchor_time_us:
            raise ValueError("piecewise-Hermite duration does not match configured anchor time")
        for left, right in zip(profile.segments, profile.segments[1:]):
            if not np.allclose(left.end_xy, right.start_xy, atol=1e-12, rtol=0):
                raise ValueError("piecewise-Hermite position continuity is violated")
            jump = np.linalg.norm(right.start_velocity_xy - left.end_velocity_xy)
            if not (config.velocity_jump_range_mps[0] <= jump <= config.velocity_jump_range_mps[1]):
                raise ValueError("piecewise-Hermite velocity jump is outside configured range")
    if not _profile_is_bounded(profile, config):
        raise ValueError("motion profile violates configured physical contract")
    anchor = profile.sample(config.anchor_time_us)
    if not np.allclose(anchor.position[:2], config.anchor_position_xy, atol=1e-12, rtol=0):
        raise ValueError("motion profile does not reach configured anchor")


def _build_level1(
    config: MotionConfig, rng: np.random.Generator, workspace_z: float
) -> MotionProfile:
    duration_s = config.anchor_time_us / 1_000_000
    anchor = np.asarray(config.anchor_position_xy)
    for _ in range(config.max_retries):
        speed = _sample_speed(rng, config)
        angle = rng.uniform(0.0, 2 * np.pi)
        velocity = speed * np.array([np.cos(angle), np.sin(angle)])
        profile = ConstantVelocityProfile(
            anchor - velocity * duration_s,
            velocity,
            config.anchor_time_us,
            workspace_z,
        )
        if _profile_is_bounded(profile, config):
            return profile
    raise RuntimeError("failed to generate a valid Level 1 motion profile")


def _build_level2(
    config: MotionConfig, rng: np.random.Generator, workspace_z: float
) -> MotionProfile:
    duration_s = config.anchor_time_us / 1_000_000
    anchor = np.asarray(config.anchor_position_xy)
    for _ in range(config.max_retries):
        chord_speed = _sample_speed(rng, config, upper_scale=0.65)
        angle = rng.uniform(0.0, 2 * np.pi)
        direction = np.array([np.cos(angle), np.sin(angle)])
        perpendicular = np.array([-direction[1], direction[0]])
        start = anchor - direction * chord_speed * duration_s
        curve_sign = -1.0 if int(rng.integers(0, 2)) == 0 else 1.0
        curve_speed = curve_sign * rng.uniform(0.2, 0.55) * chord_speed
        start_velocity = direction * chord_speed + perpendicular * curve_speed
        end_velocity = direction * (0.55 * chord_speed) - perpendicular * (0.4 * curve_speed)
        profile = HermiteProfile(
            HermiteSegment(start, anchor, start_velocity, end_velocity, config.anchor_time_us),
            workspace_z,
        )
        if _profile_is_bounded(profile, config):
            midpoint = profile.sample(config.anchor_time_us // 2).position[:2]
            chord_midpoint = 0.5 * (start + anchor)
            midpoint_offset = np.linalg.norm(midpoint - chord_midpoint)
            if 1e-3 <= midpoint_offset <= config.curve_offset_max_m:
                return profile
    raise RuntimeError("failed to generate a valid Level 2 motion profile")


def _segment_durations(
    config: MotionConfig, rng: np.random.Generator, count: int
) -> tuple[int, ...]:
    total_ticks = config.anchor_time_us // 20_000
    minimum_ticks = config.min_segment_duration_us // 20_000
    remaining = total_ticks - count * minimum_ticks
    extras = np.zeros(count, dtype=int)
    if remaining:
        probabilities = rng.dirichlet(np.ones(count))
        extras = np.floor(probabilities * remaining).astype(int)
        for index in range(remaining - int(extras.sum())):
            extras[index % count] += 1
    return tuple(int((minimum_ticks + extra) * 20_000) for extra in extras)


def _build_level3(
    config: MotionConfig, rng: np.random.Generator, workspace_z: float
) -> MotionProfile:
    anchor = np.asarray(config.anchor_position_xy)
    for _ in range(config.max_retries):
        count = int(rng.integers(config.segment_count_range[0], config.segment_count_range[1] + 1))
        durations = _segment_durations(config, rng, count)
        average_velocities = []
        displacements = []
        for duration_us in durations:
            speed = _sample_speed(rng, config, upper_scale=0.60)
            angle = rng.uniform(0.0, 2 * np.pi)
            velocity = speed * np.array([np.cos(angle), np.sin(angle)])
            average_velocities.append(velocity)
            displacements.append(velocity * (duration_us / 1_000_000))
        start = anchor - np.sum(displacements, axis=0)
        positions = [start]
        for displacement in displacements:
            positions.append(positions[-1] + displacement)
        positions[-1] = anchor

        segments = []
        for index, (duration_us, average_velocity) in enumerate(
            zip(durations, average_velocities, strict=True)
        ):
            direction = average_velocity / np.linalg.norm(average_velocity)
            perpendicular = np.array([-direction[1], direction[0]])
            curve_speed = rng.uniform(-0.25, 0.25) * np.linalg.norm(average_velocity)
            start_velocity = average_velocity + perpendicular * curve_speed
            end_velocity = average_velocity - perpendicular * curve_speed
            segments.append(
                HermiteSegment(
                    np.asarray(positions[index]),
                    np.asarray(positions[index + 1]),
                    start_velocity,
                    end_velocity,
                    duration_us,
                )
            )
        change_times = tuple(int(sum(durations[: index + 1])) for index in range(count - 1))
        profile = PiecewiseHermiteProfile(tuple(segments), workspace_z, change_times)
        if not _profile_is_bounded(profile, config):
            continue
        if any(
            np.linalg.norm(
                segment.evaluate(segment.duration_us // 2)[0]
                - 0.5 * (segment.start_xy + segment.end_xy)
            )
            > config.curve_offset_max_m
            for segment in segments
        ):
            continue
        jumps = [
            np.linalg.norm(segments[index + 1].start_velocity_xy - segments[index].end_velocity_xy)
            for index in range(count - 1)
        ]
        if all(
            config.velocity_jump_range_mps[0] <= jump <= config.velocity_jump_range_mps[1]
            for jump in jumps
        ):
            return profile
    raise RuntimeError("failed to generate a valid Level 3 motion profile")


def build_motion_profile(*, config: MotionConfig, seed: int, workspace_z: float) -> MotionProfile:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("motion seed must be an integer")
    if not np.isfinite(workspace_z):
        raise ValueError("workspace_z must be finite")
    rng = np.random.default_rng(seed)
    if config.level == 0:
        return StationaryProfile(np.asarray(config.anchor_position_xy), workspace_z)
    if config.level == 1:
        return _build_level1(config, rng, workspace_z)
    if config.level == 2:
        return _build_level2(config, rng, workspace_z)
    return _build_level3(config, rng, workspace_z)
