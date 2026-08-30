"""Leakage-free K6 corpus for motion-aware causal-return history encoding."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from latency_meta_mdp.belief.causal_return.motion_aware_contracts import (
    MotionAwareHistorySample,
)
from latency_meta_mdp.temporal_contract import TemporalContract, load_temporal_contract
from latency_meta_mdp.vision_encoder import VisionEncoderSpec
from latency_meta_mdp.vision_probe_corpus import (
    VisionProbeCorpus,
    VisionProbeEpisodeRecord,
    load_level_probe_corpus,
)
from latency_meta_mdp.vision_probe_data import ProbeSplit, VisionProbeSampleIndex


@dataclass(frozen=True)
class MotionAwareHistoryEpisodeRecord:
    """One verified episode/cache pair and its shared causal-return indices."""

    source: VisionProbeEpisodeRecord
    indices: tuple[VisionProbeSampleIndex, ...]


@dataclass(frozen=True)
class MotionAwareHistoryCorpus:
    """Level-specific histories with episode-level train/validation isolation."""

    level: int
    history_sample_count: int
    records: tuple[MotionAwareHistoryEpisodeRecord, ...]
    sample_references: dict[ProbeSplit, tuple[tuple[int, int], ...]]

    @property
    def sample_counts(self) -> dict[ProbeSplit, int]:
        return {split: len(self.sample_references[split]) for split in ProbeSplit}

    @property
    def episode_counts(self) -> dict[ProbeSplit, int]:
        counts = Counter(
            record.indices[0].split for record in self.records if record.indices
        )
        return {split: counts[split] for split in ProbeSplit}

    @property
    def seeds(self) -> dict[ProbeSplit, frozenset[int]]:
        values: dict[ProbeSplit, set[int]] = defaultdict(set)
        for record in self.records:
            if record.indices:
                values[record.indices[0].split].add(record.source.episode.scene_seed)
        return {split: frozenset(values[split]) for split in ProbeSplit}

    def materialize(self, split: ProbeSplit, offset: int) -> MotionAwareHistorySample:
        record_offset, index_offset = self.sample_references[split][offset]
        record = self.records[record_offset]
        index = record.indices[index_offset]
        episode = record.source.episode
        history = slice(index.history_start_tick, index.source_tick + 1)
        transition_offset, transition_valid = _signed_transition_offset(
            episode=episode,
            source_tick=index.source_tick,
        )
        robot_history, position, velocity = _physical_state_values(
            episode=episode,
            history=history,
            source_tick=index.source_tick,
        )
        return MotionAwareHistorySample(
            episode_id=episode.episode_id,
            level=episode.level,
            scene_seed=episode.scene_seed,
            source_tick=index.source_tick,
            source_phase=str(episode.expert_phase[index.source_tick]),
            transition_offset_ticks=transition_offset,
            transition_offset_valid=transition_valid,
            vision_history=record.source.cache.features[history],
            robot_history=robot_history,
            history_valid_mask=np.ones(self.history_sample_count, dtype=np.bool_),
            object_position_target=position,
            object_velocity_target=velocity,
        )

    def state_values(
        self,
        split: ProbeSplit,
        offset: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return normalization values without copying visual features."""

        record_offset, index_offset = self.sample_references[split][offset]
        record = self.records[record_offset]
        index = record.indices[index_offset]
        return _physical_state_values(
            episode=record.source.episode,
            history=slice(index.history_start_tick, index.source_tick + 1),
            source_tick=index.source_tick,
        )

    def transition_metadata(self, split: ProbeSplit, offset: int) -> tuple[int, bool]:
        """Return signed transition metadata without copying visual features."""

        record_offset, index_offset = self.sample_references[split][offset]
        record = self.records[record_offset]
        index = record.indices[index_offset]
        return _signed_transition_offset(
            episode=record.source.episode,
            source_tick=index.source_tick,
        )


def _robot_history(*, episode, history: slice) -> np.ndarray:
    deployment = episode.deployment
    width = deployment.gripper_qpos[:, 0] - deployment.gripper_qpos[:, 1]
    width_velocity = deployment.gripper_qvel[:, 0] - deployment.gripper_qvel[:, 1]
    return np.concatenate(
        (
            deployment.robot_qpos[history],
            deployment.robot_qvel[history],
            width[history, None],
            width_velocity[history, None],
        ),
        axis=1,
    )


def _physical_state_values(
    *,
    episode,
    history: slice,
    source_tick: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(_robot_history(episode=episode, history=history), dtype=np.float32),
        np.asarray(
            episode.supervision.object_pose[source_tick, :3], dtype=np.float32
        ),
        np.asarray(
            episode.supervision.object_velocity[source_tick, :3], dtype=np.float32
        ),
    )


def _signed_transition_offset(*, episode, source_tick: int) -> tuple[int, bool]:
    if episode.level != 3:
        return 0, False
    segment_index = np.asarray(episode.supervision.commanded_motion_segment_index)
    transition_ticks = np.flatnonzero(segment_index[1:] != segment_index[:-1]) + 1
    if len(transition_ticks) == 0:
        raise ValueError("L3 motion-aware episode has no segment transition")
    nearest = min(
        (int(tick) for tick in transition_ticks),
        key=lambda tick: (abs(tick - source_tick), tick),
    )
    return source_tick - nearest, True


def build_motion_aware_history_corpus(
    *,
    source: VisionProbeCorpus,
    temporal_contract: TemporalContract,
) -> MotionAwareHistoryCorpus:
    """Restrict a neutral feature corpus to sources shared with later dynamics."""

    if (
        source.level not in (1, 2, 3)
        or source.history_sample_count != 6
        or temporal_contract.history_sample_count != source.history_sample_count
        or temporal_contract.formal_tick_us != 20_000
    ):
        raise ValueError("motion-aware corpus and temporal contract disagree")
    records = []
    references: dict[ProbeSplit, list[tuple[int, int]]] = defaultdict(list)
    seed_splits: dict[int, ProbeSplit] = {}
    for source_record in source.records:
        episode = source_record.episode
        record_splits = {index.split for index in source_record.indices}
        if len(record_splits) != 1:
            raise ValueError("motion-aware episode indices must belong to one split")
        record_split = next(iter(record_splits))
        previous_split = seed_splits.setdefault(episode.scene_seed, record_split)
        if previous_split is not record_split:
            raise ValueError("motion-aware scene seed cannot cross episode splits")
        interval = temporal_contract.belief_source_interval(
            episode_action_count=episode.transition_count
        )
        indices = tuple(
            index
            for index in source_record.indices
            if interval.minimum <= index.source_tick <= interval.maximum
        )
        if not indices:
            raise ValueError("motion-aware episode has no shared causal-return source")
        record_offset = len(records)
        records.append(
            MotionAwareHistoryEpisodeRecord(source=source_record, indices=indices)
        )
        for index_offset, index in enumerate(indices):
            references[index.split].append((record_offset, index_offset))
    return MotionAwareHistoryCorpus(
        level=source.level,
        history_sample_count=source.history_sample_count,
        records=tuple(records),
        sample_references={split: tuple(references[split]) for split in ProbeSplit},
    )


def load_motion_aware_history_corpus(
    *,
    source_bulk_manifest: Path,
    cache_run_manifest: Path,
    expected_spec: VisionEncoderSpec,
    temporal_config_path: Path,
    split_plan_path: Path,
    level: int,
) -> MotionAwareHistoryCorpus:
    """Load formal synchronized inputs and construct one level-specific corpus."""

    temporal_contract = load_temporal_contract(temporal_config_path)
    source = load_level_probe_corpus(
        source_bulk_manifest=source_bulk_manifest,
        cache_run_manifest=cache_run_manifest,
        expected_spec=expected_spec,
        level=level,
        history_sample_count=temporal_contract.history_sample_count,
        split_plan_path=split_plan_path,
    )
    return build_motion_aware_history_corpus(
        source=source,
        temporal_contract=temporal_contract,
    )
