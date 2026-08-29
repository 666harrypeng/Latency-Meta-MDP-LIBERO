"""Pure contracts for the Flow Belief action-buffer causality gate."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import yaml

_PHASES = ("pregrasp", "approach", "close", "lift")
_BRANCH_IDS = ("expert", "hold", "half_speed", "delayed_prefix_5")


@dataclass(frozen=True)
class BufferCausalityConfig:
    schema_version: int
    audit_id: str
    contexts_per_phase: int
    phases: tuple[str, ...]
    branch_ids: tuple[str, ...]
    delay_ticks: tuple[int, ...]
    sample_count: int
    solver: str
    solver_step_count: int
    required_real_action_count: int
    half_speed_scale: float
    delayed_prefix_ticks: int
    sampling_seed: int

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.audit_id != "dinov3_flow_belief_buffer_causality_v1":
            raise ValueError("unsupported buffer-causality configuration")
        if self.contexts_per_phase <= 0 or self.phases != _PHASES:
            raise ValueError("buffer-causality phase configuration is invalid")
        if self.branch_ids != _BRANCH_IDS:
            raise ValueError("buffer-causality branch inventory is invalid")
        if self.delay_ticks != tuple(range(1, 21)):
            raise ValueError("buffer-causality delays must be ticks 1 through 20")
        if self.sample_count <= 0 or self.solver != "heun" or self.solver_step_count <= 0:
            raise ValueError("buffer-causality Flow sampling configuration is invalid")
        if self.required_real_action_count != 25:
            raise ValueError("buffer-causality requires a complete real 25-action buffer")
        if not 0.0 < self.half_speed_scale < 1.0:
            raise ValueError("buffer-causality half-speed scale must lie in (0, 1)")
        if not 0 < self.delayed_prefix_ticks < self.required_real_action_count:
            raise ValueError("buffer-causality delayed prefix is invalid")
        if isinstance(self.sampling_seed, bool) or not isinstance(self.sampling_seed, int):
            raise ValueError("buffer-causality sampling seed must be an integer")


@dataclass(frozen=True)
class BufferCausalityCandidate:
    episode_id: str
    level: int
    scene_seed: int
    source_tick: int
    phase: str
    real_transition_count: int

    def __post_init__(self) -> None:
        if (
            not self.episode_id
            or self.level not in (1, 2, 3)
            or self.phase not in _PHASES
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in (self.scene_seed, self.source_tick, self.real_transition_count)
            )
            or self.source_tick >= self.real_transition_count
        ):
            raise ValueError("buffer-causality candidate is invalid")


def load_buffer_causality_config(path: Path) -> BufferCausalityConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("buffer-causality config must be a mapping")
    values = dict(raw)
    for name in ("phases", "branch_ids", "delay_ticks"):
        values[name] = tuple(values[name])
    return BufferCausalityConfig(**values)


def _evenly_spaced(rows: list[BufferCausalityCandidate], count: int) -> list:
    if count == 1:
        return [rows[len(rows) // 2]]
    indices = [round(index * (len(rows) - 1) / (count - 1)) for index in range(count)]
    if len(set(indices)) != count:
        raise ValueError("buffer-causality selection indices are not unique")
    return [rows[index] for index in indices]


def select_buffer_causality_contexts(
    *,
    candidates: tuple[BufferCausalityCandidate, ...],
    levels: tuple[int, ...],
    phases: tuple[str, ...],
    contexts_per_phase: int,
    required_real_action_count: int,
) -> tuple[BufferCausalityCandidate, ...]:
    if levels != tuple(sorted(set(levels))) or any(level not in (1, 2, 3) for level in levels):
        raise ValueError("buffer-causality levels must be sorted and unique")
    if not phases or any(phase not in _PHASES for phase in phases):
        raise ValueError("buffer-causality phases are invalid")
    if contexts_per_phase <= 0 or required_real_action_count <= 0:
        raise ValueError("buffer-causality selection counts must be positive")
    selected = []
    for level in levels:
        for phase in phases:
            eligible = sorted(
                (
                    row
                    for row in candidates
                    if row.level == level
                    and row.phase == phase
                    and row.source_tick + required_real_action_count
                    <= row.real_transition_count
                ),
                key=lambda row: (row.scene_seed, row.source_tick, row.episode_id),
            )
            episode_ids = [row.episode_id for row in eligible]
            if len(episode_ids) != len(set(episode_ids)):
                raise ValueError(
                    f"buffer-causality candidates duplicate an episode for L{level} {phase}"
                )
            if len(eligible) < contexts_per_phase:
                raise ValueError(
                    f"buffer-causality needs {contexts_per_phase} contexts for L{level} {phase}"
                )
            selected.extend(_evenly_spaced(eligible, contexts_per_phase))
    return tuple(sorted(selected, key=lambda row: (row.level, row.phase, row.scene_seed)))


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=np.float32, copy=True)
    result.setflags(write=False)
    return result


def build_buffer_branches(
    expert_actions: np.ndarray,
    *,
    half_speed_scale: float = 0.5,
    delayed_prefix_ticks: int = 5,
) -> Mapping[str, np.ndarray]:
    expert = np.asarray(expert_actions)
    if expert.shape != (25, 7) or not np.all(np.isfinite(expert)):
        raise ValueError("expert action buffer must be finite with shape [25, 7]")
    if not 0.0 < half_speed_scale < 1.0 or not 0 < delayed_prefix_ticks < 25:
        raise ValueError("buffer-causality branch parameters are invalid")
    expert = np.array(expert, dtype=np.float32, copy=True)
    hold_action = np.zeros(7, dtype=np.float32)
    hold_action[6] = expert[0, 6]
    hold = np.repeat(hold_action[None], 25, axis=0)
    half_speed = np.array(expert, copy=True)
    half_speed[:, :6] *= half_speed_scale
    delayed = np.concatenate(
        (
            np.repeat(hold_action[None], delayed_prefix_ticks, axis=0),
            expert[: 25 - delayed_prefix_ticks],
        ),
        axis=0,
    )
    return MappingProxyType(
        {
            "expert": _readonly(expert),
            "hold": _readonly(hold),
            "half_speed": _readonly(half_speed),
            "delayed_prefix_5": _readonly(delayed),
        }
    )
