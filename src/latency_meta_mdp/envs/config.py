"""Validated runtime configuration for the synchronized RoboSuite backend."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RuntimeConfig:
    """The locked G0 timing/runtime contract.

    MuJoCo integrates internally at ``physics_dt_us``. All model-facing state,
    policy, client, camera, and recorder boundaries use ``formal_tick_us``.
    """

    runtime_version: str
    physics_dt_us: int
    formal_tick_us: int
    camera_stride_ticks: int
    compatibility_stride_ticks: int
    control_freq_hz: int
    lite_physics: bool

    def __post_init__(self) -> None:
        if not isinstance(self.runtime_version, str):
            raise ValueError("runtime_version must be a string")
        if not isinstance(self.lite_physics, bool):
            raise ValueError("lite_physics must be a boolean")
        positive_int_fields = {
            "physics_dt_us": self.physics_dt_us,
            "formal_tick_us": self.formal_tick_us,
            "camera_stride_ticks": self.camera_stride_ticks,
            "compatibility_stride_ticks": self.compatibility_stride_ticks,
            "control_freq_hz": self.control_freq_hz,
        }
        for name, value in positive_int_fields.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not self.runtime_version:
            raise ValueError("runtime_version must be non-empty")
        if self.formal_tick_us % self.physics_dt_us != 0:
            raise ValueError("formal_tick_us must be an integer multiple of physics_dt_us")
        expected_control_frequency = 1_000_000 // self.formal_tick_us
        if expected_control_frequency * self.formal_tick_us != 1_000_000:
            raise ValueError("formal_tick_us must divide one second exactly")
        if self.control_freq_hz != expected_control_frequency:
            raise ValueError(
                "control_freq_hz must equal the formal tick frequency "
                f"({expected_control_frequency})"
            )
        locked_values = {
            "runtime_version": (self.runtime_version, "robosuite_native_v1"),
            "physics_dt_us": (self.physics_dt_us, 2_000),
            "formal_tick_us": (self.formal_tick_us, 20_000),
            "camera_stride_ticks": (self.camera_stride_ticks, 1),
            "compatibility_stride_ticks": (self.compatibility_stride_ticks, 5),
            "control_freq_hz": (self.control_freq_hz, 50),
            "lite_physics": (self.lite_physics, True),
        }
        for name, (actual, expected) in locked_values.items():
            if actual != expected:
                raise ValueError(f"{name} must equal locked G0 value {expected!r}")

    @property
    def physics_steps_per_tick(self) -> int:
        return self.formal_tick_us // self.physics_dt_us

    @classmethod
    def default(cls) -> RuntimeConfig:
        return cls(
            runtime_version="robosuite_native_v1",
            physics_dt_us=2_000,
            formal_tick_us=20_000,
            camera_stride_ticks=1,
            compatibility_stride_ticks=5,
            control_freq_hz=50,
            lite_physics=True,
        )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> RuntimeConfig:
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"unknown runtime config keys: {unknown}")
        missing = sorted(allowed - set(values))
        if missing:
            raise ValueError(f"missing runtime config keys: {missing}")
        return cls(**dict(values))

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    """Load the exact G0 runtime contract from a YAML mapping."""
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, Mapping):
        raise ValueError("runtime config must contain a YAML mapping")
    return RuntimeConfig.from_mapping(values)
