"""Deterministic critical-phase source selection for rolling belief reports."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from latency_meta_mdp.belief.common.feature_corpus import FeatureBeliefCorpus
from latency_meta_mdp.belief.flow.rolling_config import FlowBeliefRollingConfig
from latency_meta_mdp.vision_probe_data import ProbeSplit


def _integer(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int)


@dataclass(frozen=True, order=True)
class RollingWindowIdentity:
    level: int
    episode_id: str
    scene_seed: int
    validation_offset: int
    source_tick: int
    history_start_tick: int
    source_phase: str
    critical_end_tick: int
    handoff_tick: int
    handoff_distance_ticks: int

    def __post_init__(self) -> None:
        integers = (
            self.level,
            self.scene_seed,
            self.validation_offset,
            self.source_tick,
            self.history_start_tick,
            self.critical_end_tick,
            self.handoff_tick,
            self.handoff_distance_ticks,
        )
        if any(not _integer(value) for value in integers):
            raise ValueError("rolling window integer fields are invalid")
        if (
            self.level not in (1, 2, 3)
            or not self.episode_id
            or self.scene_seed < 0
            or self.validation_offset < 0
            or self.history_start_tick < 0
            or self.source_tick < self.history_start_tick
            or self.source_phase not in ("pregrasp", "approach")
            or self.critical_end_tick < self.source_tick
            or self.handoff_tick < self.critical_end_tick
            or self.handoff_distance_ticks != self.handoff_tick - self.source_tick
        ):
            raise ValueError("rolling window identity is invalid")


@dataclass(frozen=True)
class RollingSeedSelection:
    level: int
    episode_id: str
    scene_seed: int
    critical_start_tick: int
    critical_end_tick: int
    approach_start_tick: int
    handoff_tick: int
    windows: tuple[RollingWindowIdentity, ...]

    def __post_init__(self) -> None:
        integers = (
            self.level,
            self.scene_seed,
            self.critical_start_tick,
            self.critical_end_tick,
            self.approach_start_tick,
            self.handoff_tick,
        )
        windows = tuple(self.windows)
        if any(not _integer(value) for value in integers):
            raise ValueError("rolling seed integer fields are invalid")
        if (
            self.level not in (1, 2, 3)
            or not self.episode_id
            or self.scene_seed < 0
            or not windows
            or self.critical_start_tick != windows[0].source_tick
            or not self.critical_start_tick <= self.approach_start_tick <= self.critical_end_tick
            or self.handoff_tick < self.critical_end_tick
            or windows != tuple(sorted(set(windows)))
            or any(
                row.level != self.level
                or row.episode_id != self.episode_id
                or row.scene_seed != self.scene_seed
                or row.critical_end_tick != self.critical_end_tick
                or row.handoff_tick != self.handoff_tick
                for row in windows
            )
        ):
            raise ValueError("rolling seed selection is invalid")
        object.__setattr__(self, "windows", windows)


def select_rolling_source_ticks(
    *,
    legal_source_ticks: tuple[int, ...],
    source_phase_by_tick: dict[int, str],
    critical_end_tick: int,
    config: FlowBeliefRollingConfig,
) -> tuple[int, ...]:
    ticks = tuple(legal_source_ticks)
    if (
        not isinstance(config, FlowBeliefRollingConfig)
        or not _integer(critical_end_tick)
        or critical_end_tick < 0
        or not ticks
        or any(not _integer(tick) or tick < 0 for tick in ticks)
        or ticks != tuple(sorted(set(ticks)))
    ):
        raise ValueError("rolling legal source ticks are invalid")
    if any(tick not in source_phase_by_tick for tick in ticks):
        raise ValueError("rolling source phase mapping is incomplete")
    if any(source_phase_by_tick[tick] not in config.source_phases for tick in ticks):
        raise ValueError("rolling source ticks must use configured critical phases")
    maximum_delay = max(config.display_delay_ticks)
    eligible = tuple(
        tick
        for tick in ticks
        if tick >= config.history_sample_count - 1 and tick + maximum_delay <= critical_end_tick
    )
    if not eligible:
        raise ValueError("rolling inspection requires a full pregrasp/approach window")
    first = eligible[0]
    last = critical_end_tick - maximum_delay
    eligible_set = set(eligible)
    if last not in eligible_set:
        raise ValueError("rolling inspection final full-window source tick is unavailable")
    selected = set(range(first, last + 1, config.inspection_stride_ticks))
    if not selected <= eligible_set:
        raise ValueError("rolling inspection uniform source tick is unavailable")
    approach_ticks = tuple(tick for tick in eligible if source_phase_by_tick[tick] == "approach")
    if config.include_approach_anchor:
        if not approach_ticks:
            raise ValueError("rolling inspection approach anchor is unavailable")
        selected.add(approach_ticks[0])
    if config.include_last_full_window_anchor:
        selected.add(last)
    return tuple(sorted(selected))


def _first_tick(values: np.ndarray, expected: str, *, field: str) -> int:
    matches = np.flatnonzero(np.asarray(values) == expected)
    if len(matches) == 0:
        raise ValueError(f"rolling inspection {field} is unavailable")
    return int(matches[0])


def select_rolling_seed_windows(
    *,
    corpus: FeatureBeliefCorpus,
    scene_seed: int,
    config: FlowBeliefRollingConfig,
) -> RollingSeedSelection:
    if not isinstance(corpus, FeatureBeliefCorpus) or not isinstance(
        config, FlowBeliefRollingConfig
    ):
        raise TypeError("rolling seed selection requires typed corpus and config")
    if not _integer(scene_seed) or scene_seed < 0:
        raise ValueError("rolling scene seed is invalid")
    if corpus.temporal_contract.history_sample_count != config.history_sample_count:
        raise ValueError("rolling history config and corpus disagree")
    matches = [
        (record_offset, record)
        for record_offset, record in enumerate(corpus.records)
        if record.split is ProbeSplit.VALIDATION and record.scene_seed == scene_seed
    ]
    if len(matches) != 1:
        raise ValueError("rolling inspection requires one matching validation episode")
    record_offset, record = matches[0]
    critical_end_tick = _first_tick(record.tail.expert_phase, "close", field="close phase")
    approach_start_tick = _first_tick(
        record.tail.expert_phase,
        "approach",
        field="approach phase",
    )
    handoff_tick = _first_tick(record.handoff, "physical", field="physical handoff")
    offset_by_source_tick: dict[int, int] = {}
    for validation_offset, (reference_record, index_offset) in enumerate(
        corpus.sample_references[ProbeSplit.VALIDATION]
    ):
        if reference_record == record_offset:
            source_tick = int(record.indices[index_offset].source_tick)
            if source_tick in offset_by_source_tick:
                raise ValueError("rolling validation source tick is duplicated")
            offset_by_source_tick[source_tick] = validation_offset
    critical_ticks = tuple(
        sorted(
            tick
            for tick in offset_by_source_tick
            if tick < len(record.tail.expert_phase)
            and str(record.tail.expert_phase[tick]) in config.source_phases
            and tick <= critical_end_tick
        )
    )
    phases = {tick: str(record.tail.expert_phase[tick]) for tick in critical_ticks}
    selected_ticks = select_rolling_source_ticks(
        legal_source_ticks=critical_ticks,
        source_phase_by_tick=phases,
        critical_end_tick=critical_end_tick,
        config=config,
    )
    windows = tuple(
        RollingWindowIdentity(
            level=corpus.level,
            episode_id=record.episode_id,
            scene_seed=record.scene_seed,
            validation_offset=offset_by_source_tick[source_tick],
            source_tick=source_tick,
            history_start_tick=source_tick - config.history_sample_count + 1,
            source_phase=phases[source_tick],
            critical_end_tick=critical_end_tick,
            handoff_tick=handoff_tick,
            handoff_distance_ticks=handoff_tick - source_tick,
        )
        for source_tick in selected_ticks
    )
    return RollingSeedSelection(
        level=corpus.level,
        episode_id=record.episode_id,
        scene_seed=record.scene_seed,
        critical_start_tick=windows[0].source_tick,
        critical_end_tick=critical_end_tick,
        approach_start_tick=approach_start_tick,
        handoff_tick=handoff_tick,
        windows=windows,
    )
