"""Typed contracts for same-source-information control branches."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

_PHASES = ("pregrasp", "approach", "close", "lift")
_SPLITS = {"train", "validation"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _readonly_exact(
    value: np.ndarray,
    *,
    dtype: np.dtype,
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != dtype or array.shape != shape:
        raise ValueError(f"{name} must have shape {shape} and dtype {np.dtype(dtype)}")
    if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def _require_sha256(value: str, *, name: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256 digest")


@dataclass(frozen=True)
class BranchCorpusConfig:
    schema_version: int
    config_id: str
    contexts_per_phase_per_episode: int
    phases: tuple[str, ...]
    arm_scale_factors: tuple[float, ...]
    prefix_hold_ticks: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.config_id != "conditional_return_control_branch_corpus":
            raise ValueError("unsupported branch corpus configuration")
        if self.contexts_per_phase_per_episode != 1:
            raise ValueError("branch source selection requires exactly one context per phase")
        if self.phases != _PHASES:
            raise ValueError("branch phases must use the canonical semantic order")
        if (
            not self.arm_scale_factors
            or tuple(sorted(set(self.arm_scale_factors))) != self.arm_scale_factors
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0.0 < float(value) < 1.0
                for value in self.arm_scale_factors
            )
        ):
            raise ValueError("arm scale factors must be unique, ordered, and lie in (0, 1)")
        if (
            not self.prefix_hold_ticks
            or tuple(sorted(set(self.prefix_hold_ticks))) != self.prefix_hold_ticks
            or any(
                isinstance(value, bool) or not isinstance(value, int) or not 0 < value < 20
                for value in self.prefix_hold_ticks
            )
        ):
            raise ValueError("prefix-hold ticks must be unique ordered integers in [1, 19]")


def load_branch_corpus_config(path: Path) -> BranchCorpusConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != set(BranchCorpusConfig.__dataclass_fields__):
        raise ValueError("branch corpus config fields are invalid")
    values = dict(raw)
    for name in ("phases", "arm_scale_factors", "prefix_hold_ticks"):
        if not isinstance(values[name], list):
            raise ValueError(f"branch corpus {name} must be a YAML sequence")
        values[name] = tuple(values[name])
    return BranchCorpusConfig(**values)


@dataclass(frozen=True)
class SourceContextIdentity:
    source_context_id: str
    episode_id: str
    level: int
    scene_seed: int
    split: str
    source_tick: int
    source_phase: str
    history_start_tick: int
    source_episode_manifest_sha256: str
    source_arrays_sha256: str
    source_metadata_sha256: str
    source_observation_sha256: str
    motion_profile_sha256: str

    def __post_init__(self) -> None:
        if not self.source_context_id or not self.episode_id:
            raise ValueError("source context identity strings cannot be empty")
        if self.level not in (1, 2, 3):
            raise ValueError("source context level must be 1, 2, or 3")
        if (
            isinstance(self.scene_seed, bool)
            or not isinstance(self.scene_seed, int)
            or self.scene_seed < 0
        ):
            raise ValueError("source scene seed must be a non-negative integer")
        if self.split not in _SPLITS:
            raise ValueError("source split must be train or validation")
        if self.source_phase not in _PHASES:
            raise ValueError("source phase is invalid")
        if (
            isinstance(self.source_tick, bool)
            or not isinstance(self.source_tick, int)
            or self.source_tick < 5
            or self.history_start_tick != self.source_tick - 5
        ):
            raise ValueError("source history must be the exact K6 window ending at source tick")
        for name in (
            "source_episode_manifest_sha256",
            "source_arrays_sha256",
            "source_metadata_sha256",
            "source_observation_sha256",
            "motion_profile_sha256",
        ):
            _require_sha256(getattr(self, name), name=name)


@dataclass(frozen=True)
class SourceSelectionExclusion:
    episode_id: str
    level: int
    scene_seed: int
    split: str
    source_phase: str
    reason: str
    source_interval_minimum: int
    source_interval_maximum: int
    phase_tick_count: int
    interval_phase_tick_count: int
    running_interval_phase_tick_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise ValueError("source selection exclusion episode ID cannot be empty")
        if (
            isinstance(self.level, bool)
            or not isinstance(self.level, int)
            or self.level not in (1, 2, 3)
        ):
            raise ValueError("source selection exclusion level must be 1, 2, or 3")
        if (
            isinstance(self.scene_seed, bool)
            or not isinstance(self.scene_seed, int)
            or self.scene_seed < 0
        ):
            raise ValueError("source selection exclusion seed must be non-negative")
        if self.split not in _SPLITS or self.source_phase not in _PHASES:
            raise ValueError("source selection exclusion split or phase is invalid")
        if (
            isinstance(self.source_interval_minimum, bool)
            or not isinstance(self.source_interval_minimum, int)
            or isinstance(self.source_interval_maximum, bool)
            or not isinstance(self.source_interval_maximum, int)
            or self.source_interval_minimum < 0
            or self.source_interval_maximum < self.source_interval_minimum
        ):
            raise ValueError("source selection exclusion interval is invalid")
        counts = (
            self.phase_tick_count,
            self.interval_phase_tick_count,
            self.running_interval_phase_tick_count,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts
        ):
            raise ValueError("source selection exclusion counts must be non-negative integers")
        if not counts[2] <= counts[1] <= counts[0]:
            raise ValueError("source selection exclusion counts are inconsistent")
        expected_reason = (
            "phase_absent"
            if counts[0] == 0
            else "no_phase_tick_in_source_interval"
            if counts[1] == 0
            else "no_running_phase_tick_in_source_interval"
            if counts[2] == 0
            else None
        )
        if expected_reason is None or self.reason != expected_reason:
            raise ValueError("source selection exclusion reason does not match its counts")


@dataclass(frozen=True)
class ExecutablePrefix:
    controls: np.ndarray
    from_active_buffer_mask: np.ndarray
    last_executed_gripper_command: float

    def __post_init__(self) -> None:
        controls = _readonly_exact(
            self.controls,
            dtype=np.dtype(np.float32),
            shape=(20, 7),
            name="executable controls",
        )
        mask = _readonly_exact(
            self.from_active_buffer_mask,
            dtype=np.dtype(np.bool_),
            shape=(20,),
            name="active-buffer mask",
        )
        if not np.isfinite(self.last_executed_gripper_command) or not (
            -1.0 <= self.last_executed_gripper_command <= 1.0
        ):
            raise ValueError("last executed gripper command must be finite and lie in [-1, 1]")
        object.__setattr__(self, "controls", controls)
        object.__setattr__(self, "from_active_buffer_mask", mask)
        object.__setattr__(
            self,
            "last_executed_gripper_command",
            float(self.last_executed_gripper_command),
        )


@dataclass(frozen=True)
class ControlContinuationSpec:
    kind: str
    arm_scale: float | None
    prefix_real_ticks: int | None

    def __post_init__(self) -> None:
        if self.kind in {"nominal", "hold"}:
            if self.arm_scale is not None or self.prefix_real_ticks is not None:
                raise ValueError("nominal and hold branches cannot carry parameters")
            return
        if self.kind.startswith("arm_scale_"):
            if self.arm_scale is None or not 0.0 < self.arm_scale < 1.0:
                raise ValueError("arm-scale branch requires a scale in (0, 1)")
            if self.prefix_real_ticks is not None:
                raise ValueError("arm-scale branch cannot carry prefix ticks")
            return
        if self.kind.startswith("prefix_hold_"):
            if (
                isinstance(self.prefix_real_ticks, bool)
                or not isinstance(self.prefix_real_ticks, int)
                or not 0 < self.prefix_real_ticks < 20
            ):
                raise ValueError("prefix-hold branch requires ticks in [1, 19]")
            if self.arm_scale is not None:
                raise ValueError("prefix-hold branch cannot carry an arm scale")
            return
        raise ValueError("unsupported control continuation kind")

    @classmethod
    def nominal(cls) -> ControlContinuationSpec:
        return cls(kind="nominal", arm_scale=None, prefix_real_ticks=None)


@dataclass(frozen=True)
class BranchRollout:
    source: SourceContextIdentity
    branch: ControlContinuationSpec
    prefix: ExecutablePrefix
    target_states: np.ndarray
    target_absorbing: np.ndarray
    target_handoff_state: np.ndarray
    target_outcome_status: np.ndarray
    source_replay_max_abs: float
    source_fingerprint_match: bool
    nominal_future_valid: bool
    nominal_future_max_abs: float

    def __post_init__(self) -> None:
        states = _readonly_exact(
            self.target_states,
            dtype=np.dtype(np.float32),
            shape=(20, 22),
            name="branch target states",
        )
        absorbing = _readonly_exact(
            self.target_absorbing,
            dtype=np.dtype(np.bool_),
            shape=(20,),
            name="branch absorbing mask",
        )
        handoff = np.asarray(self.target_handoff_state)
        outcome = np.asarray(self.target_outcome_status)
        if handoff.shape != (20,) or outcome.shape != (20,):
            raise ValueError("branch status arrays must have shape (20,)")
        if handoff.dtype.kind not in {"U", "S"} or outcome.dtype.kind not in {"U", "S"}:
            raise ValueError("branch status arrays must contain strings")
        validity_flags = (self.source_fingerprint_match, self.nominal_future_valid)
        if any(not isinstance(value, bool) for value in validity_flags):
            raise TypeError("branch validity flags must be booleans")
        for name in ("source_replay_max_abs", "nominal_future_max_abs"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not self.nominal_future_valid and self.nominal_future_max_abs != 0.0:
            raise ValueError("non-nominal branch parity must use a finite zero sentinel")
        object.__setattr__(self, "target_states", states)
        object.__setattr__(self, "target_absorbing", absorbing)
        handoff_copy = np.array(handoff, copy=True)
        outcome_copy = np.array(outcome, copy=True)
        handoff_copy.setflags(write=False)
        outcome_copy.setflags(write=False)
        object.__setattr__(self, "target_handoff_state", handoff_copy)
        object.__setattr__(self, "target_outcome_status", outcome_copy)
