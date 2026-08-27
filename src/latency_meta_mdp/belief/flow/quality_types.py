"""Immutable identities and arrays for Flow Belief quality samples."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

_SOURCE_PHASES = frozenset({"pregrasp", "approach", "close", "lift"})


def _readonly(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    copied = np.array(value, dtype=dtype, copy=True)
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True, order=True)
class QualityContextIdentity:
    level: int
    episode_id: str
    scene_seed: int
    validation_offset: int
    source_tick: int

    def __post_init__(self) -> None:
        integer_values = (
            self.level,
            self.scene_seed,
            self.validation_offset,
            self.source_tick,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_values):
            raise ValueError("quality context integer identity fields are invalid")
        if (
            self.level not in (1, 2, 3)
            or not self.episode_id
            or self.scene_seed < 0
            or self.validation_offset < 0
            or self.source_tick < 0
        ):
            raise ValueError("quality context identity is invalid")


@dataclass(frozen=True)
class QualitySelection:
    identity: QualityContextIdentity
    roles: tuple[str, ...]
    source_phase: str
    ranking_score: float
    short_delay_error: float
    long_delay_error: float
    handoff_distance_ticks: int
    motion_transition_score: float
    object_position_rmse_m: float
    robot_qpos_rmse_rad: float

    def __post_init__(self) -> None:
        if not isinstance(self.identity, QualityContextIdentity):
            raise TypeError("quality selection identity must be typed")
        if (
            not self.roles
            or any(not isinstance(role, str) or not role for role in self.roles)
            or len(set(self.roles)) != len(self.roles)
        ):
            raise ValueError("quality selection roles must be non-empty and unique")
        if self.source_phase not in _SOURCE_PHASES:
            raise ValueError("quality selection source phase is invalid")
        numeric_values = (
            self.ranking_score,
            self.short_delay_error,
            self.long_delay_error,
            self.motion_transition_score,
            self.object_position_rmse_m,
            self.robot_qpos_rmse_rad,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in numeric_values):
            raise ValueError("quality selection metrics must be finite and non-negative")
        if (
            isinstance(self.handoff_distance_ticks, bool)
            or not isinstance(self.handoff_distance_ticks, int)
            or self.handoff_distance_ticks < 0
        ):
            raise ValueError("quality selection handoff distance is invalid")


@dataclass(frozen=True)
class QualitySampleBundle:
    selections: tuple[QualitySelection, ...]
    display_delay_ticks: np.ndarray
    latency_probabilities: np.ndarray
    normalized_samples: np.ndarray
    normalized_targets: np.ndarray
    physical_samples: np.ndarray
    physical_targets: np.ndarray
    interaction_mode: np.ndarray
    absorbing: np.ndarray

    def __post_init__(self) -> None:
        selections = tuple(self.selections)
        identities = tuple(selection.identity for selection in selections)
        if (
            not selections
            or any(not isinstance(selection, QualitySelection) for selection in selections)
            or identities != tuple(sorted(set(identities)))
        ):
            raise ValueError("quality bundle selections must have sorted unique identities")
        context_count = len(selections)
        expected_shapes = {
            "display_delay_ticks": (5,),
            "latency_probabilities": (context_count, 5),
            "normalized_samples": (context_count, 5, 32, 22),
            "normalized_targets": (context_count, 5, 22),
            "physical_samples": (context_count, 5, 32, 22),
            "physical_targets": (context_count, 5, 22),
            "interaction_mode": (context_count, 5),
            "absorbing": (context_count, 5),
        }
        for name, shape in expected_shapes.items():
            if np.asarray(getattr(self, name)).shape != shape:
                raise ValueError(f"quality bundle {name} has invalid shape")
        delay_ticks = np.asarray(self.display_delay_ticks)
        probabilities = np.asarray(self.latency_probabilities)
        numeric_names = (
            "latency_probabilities",
            "normalized_samples",
            "normalized_targets",
            "physical_samples",
            "physical_targets",
        )
        if not np.array_equal(delay_ticks, np.asarray([1, 5, 10, 15, 20])):
            raise ValueError("quality bundle display delays are invalid")
        if any(not np.all(np.isfinite(getattr(self, name))) for name in numeric_names):
            raise ValueError("quality bundle arrays must be finite")
        if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
            raise ValueError("quality bundle latency probabilities are invalid")
        object.__setattr__(self, "selections", selections)
        for name, dtype in (
            ("display_delay_ticks", np.int64),
            ("latency_probabilities", np.float64),
            ("normalized_samples", np.float32),
            ("normalized_targets", np.float32),
            ("physical_samples", np.float32),
            ("physical_targets", np.float32),
            ("interaction_mode", np.int8),
            ("absorbing", np.bool_),
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name), dtype=dtype))
