"""Minimal same-source control interventions for the final JEPA J4 gate."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from latency_meta_mdp.belief.jepa.diagnostics.history_signal import (
    TemporalSignalEpisode,
)

_CATEGORIES = (
    "smooth_approach",
    "grasp_funnel",
    "close_stabilize",
    "post_handoff_lift",
)


def build_j4_control_branches(
    *,
    nominal_controls: np.ndarray,
    last_executed_control: np.ndarray,
) -> Mapping[str, np.ndarray]:
    nominal = np.asarray(nominal_controls)
    last = np.asarray(last_executed_control)
    if (
        nominal.shape != (20, 7)
        or nominal.dtype != np.float32
        or last.shape != (7,)
        or last.dtype != np.float32
        or not np.all(np.isfinite(nominal))
        or not np.all(np.isfinite(last))
        or np.any(np.abs(nominal) > 1.0)
        or np.any(np.abs(last) > 1.0)
    ):
        raise ValueError("J4 controls must use finite controller-native float32 D20 arrays")
    hold = np.zeros_like(nominal)
    hold[:, 6] = last[6]
    scale = np.array(nominal, copy=True)
    scale[:, :6] *= np.float32(0.5)
    prefix = np.array(nominal, copy=True)
    prefix[4:, :6] = 0.0
    prefix[4:, 6] = prefix[3, 6]
    branches = {
        "nominal": np.array(nominal, copy=True),
        "hold": hold,
        "scale_0.5": scale,
        "prefix4_then_hold": prefix,
    }
    for value in branches.values():
        value.setflags(write=False)
    return MappingProxyType(branches)


@dataclass(frozen=True, order=True)
class J4SourceContext:
    logical_master_task_index: int
    episode_id: str
    category: str
    source_tick: int

    def __post_init__(self) -> None:
        if type(self.logical_master_task_index) is not int or self.logical_master_task_index < 0:
            raise ValueError("J4 master index is invalid")
        if type(self.episode_id) is not str or not self.episode_id:
            raise ValueError("J4 episode ID cannot be empty")
        if self.category not in _CATEGORIES:
            raise ValueError("J4 context category is invalid")
        if type(self.source_tick) is not int or self.source_tick < 8:
            raise ValueError("J4 source tick lacks stride-4 history")


def select_j4_source_contexts(
    episodes: tuple[TemporalSignalEpisode, ...],
) -> tuple[J4SourceContext, ...]:
    if (
        type(episodes) is not tuple
        or not episodes
        or any(not isinstance(value, TemporalSignalEpisode) for value in episodes)
        or len({value.record.logical_master_task_index for value in episodes}) != len(episodes)
    ):
        raise ValueError("J4 selection requires one signal episode per master")
    selected = []
    for episode in sorted(episodes, key=lambda value: value.record.logical_master_task_index):
        record = episode.record
        eligible = range(8, record.terminal_tick - 20 + 1)
        pools = {
            "smooth_approach": [
                tick
                for tick in eligible
                if record.phases[tick] == "smooth_approach"
                and episode.handoff_state[tick] == "driven"
            ],
            "grasp_funnel": [tick for tick in eligible if record.phases[tick] == "grasp_funnel"],
            "close_stabilize": [
                tick for tick in eligible if record.phases[tick] == "close_stabilize"
            ],
            "post_handoff_lift": [
                tick
                for tick in eligible
                if record.phases[tick] == "lift" and episode.handoff_state[tick] == "physical"
            ],
        }
        if any(not values for values in pools.values()):
            raise ValueError(f"J4 episode lacks a D20-ready phase context: {record.episode_id}")
        selected.extend(
            J4SourceContext(
                logical_master_task_index=record.logical_master_task_index,
                episode_id=record.episode_id,
                category=category,
                source_tick=pools[category][len(pools[category]) // 2],
            )
            for category in _CATEGORIES
        )
    return tuple(selected)
