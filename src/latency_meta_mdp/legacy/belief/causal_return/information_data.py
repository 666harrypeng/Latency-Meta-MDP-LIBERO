"""Leakage-free K6 data for causal-return information-state estimation."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from latency_meta_mdp.data.vision.contracts import VisionEncoderSpec
from latency_meta_mdp.legacy.belief.causal_return.contracts import InformationStateSample
from latency_meta_mdp.legacy.vision_probe_corpus import (
    VisionProbeCorpus,
    VisionProbeEpisodeRecord,
    load_level_probe_corpus,
)
from latency_meta_mdp.legacy.vision_probe_data import ProbeSplit, VisionProbeSampleIndex
from latency_meta_mdp.runtime.temporal_contract import TemporalContract, load_temporal_contract


@dataclass(frozen=True)
class InformationStateEpisodeRecord:
    source: VisionProbeEpisodeRecord
    indices: tuple[VisionProbeSampleIndex, ...]


@dataclass(frozen=True)
class InformationStateCorpus:
    level: int
    history_sample_count: int
    records: tuple[InformationStateEpisodeRecord, ...]
    sample_references: dict[ProbeSplit, tuple[tuple[int, int], ...]]

    @property
    def sample_counts(self) -> dict[ProbeSplit, int]:
        return {split: len(self.sample_references[split]) for split in ProbeSplit}

    @property
    def episode_counts(self) -> dict[ProbeSplit, int]:
        counts = Counter(record.indices[0].split for record in self.records if record.indices)
        return {split: counts[split] for split in ProbeSplit}

    def materialize(self, split: ProbeSplit, offset: int) -> InformationStateSample:
        record_offset, index_offset = self.sample_references[split][offset]
        record = self.records[record_offset]
        index = record.indices[index_offset]
        episode = record.source.episode
        history = slice(index.history_start_tick, index.source_tick + 1)
        robot_history, target = _state_values(
            episode=episode,
            history=history,
            source_tick=index.source_tick,
        )
        relative_ticks = (
            np.arange(index.history_start_tick, index.source_tick + 1) - index.source_tick
        )
        velocity = np.asarray(
            episode.supervision.commanded_motion_velocity[index.source_tick, :3],
            dtype=np.float64,
        )
        acceleration = np.asarray(
            episode.supervision.commanded_motion_acceleration[index.source_tick, :3],
            dtype=np.float64,
        )
        speed = float(np.linalg.norm(velocity))
        motion_curvature = float(
            np.linalg.norm(np.cross(velocity, acceleration)) / max(speed**3, 1e-12)
        )
        segments = np.asarray(episode.supervision.commanded_motion_segment_index)
        transition_ticks = np.flatnonzero(segments[1:] != segments[:-1]) + 1
        transition_distance = (
            min(abs(int(tick) - index.source_tick) for tick in transition_ticks)
            if len(transition_ticks)
            else -1
        )
        return InformationStateSample(
            episode_id=episode.episode_id,
            level=episode.level,
            scene_seed=episode.scene_seed,
            source_tick=index.source_tick,
            source_phase=str(episode.expert_phase[index.source_tick]),
            motion_curvature=motion_curvature,
            motion_transition_distance_ticks=transition_distance,
            vision_history=record.source.cache.features[history],
            robot_history=robot_history,
            history_time_ms=relative_ticks * 20,
            history_valid_mask=np.ones(self.history_sample_count, dtype=np.bool_),
            object_state_target=target,
        )

    def state_values(
        self,
        split: ProbeSplit,
        offset: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return normalization values without loading or copying visual features."""

        record_offset, index_offset = self.sample_references[split][offset]
        record = self.records[record_offset]
        index = record.indices[index_offset]
        history = slice(index.history_start_tick, index.source_tick + 1)
        return _state_values(
            episode=record.source.episode,
            history=history,
            source_tick=index.source_tick,
        )


def _state_values(*, episode, history: slice, source_tick: int) -> tuple[np.ndarray, np.ndarray]:
    deployment = episode.deployment
    width = deployment.gripper_qpos[:, 0] - deployment.gripper_qpos[:, 1]
    width_velocity = deployment.gripper_qvel[:, 0] - deployment.gripper_qvel[:, 1]
    robot_history = np.concatenate(
        (
            deployment.robot_qpos[history],
            deployment.robot_qvel[history],
            width[history, None],
            width_velocity[history, None],
        ),
        axis=1,
    )
    target = np.concatenate(
        (
            episode.supervision.object_pose[source_tick, :3],
            episode.supervision.object_velocity[source_tick, :3],
        )
    )
    return robot_history, target


def build_information_state_corpus(
    *,
    source: VisionProbeCorpus,
    temporal_contract: TemporalContract,
) -> InformationStateCorpus:
    """Restrict neutral cached records to source ticks shared by later causal dynamics."""

    if (
        source.level not in (1, 2, 3)
        or source.history_sample_count != 6
        or temporal_contract.history_sample_count != source.history_sample_count
    ):
        raise ValueError("information-state corpus and temporal contract disagree")
    records = []
    references: dict[ProbeSplit, list[tuple[int, int]]] = defaultdict(list)
    for source_record in source.records:
        episode = source_record.episode
        interval = temporal_contract.belief_source_interval(
            episode_action_count=episode.transition_count
        )
        indices = tuple(
            index
            for index in source_record.indices
            if interval.minimum <= index.source_tick <= interval.maximum
        )
        if not indices:
            raise ValueError("information-state episode has no shared causal-return source")
        record_offset = len(records)
        records.append(InformationStateEpisodeRecord(source=source_record, indices=indices))
        for index_offset, index in enumerate(indices):
            references[index.split].append((record_offset, index_offset))
    return InformationStateCorpus(
        level=source.level,
        history_sample_count=source.history_sample_count,
        records=tuple(records),
        sample_references={split: tuple(references[split]) for split in ProbeSplit},
    )


def load_information_state_corpus(
    *,
    source_bulk_manifest: Path,
    cache_run_manifest: Path,
    expected_spec: VisionEncoderSpec,
    temporal_config_path: Path,
    split_plan_path: Path,
    level: int,
) -> InformationStateCorpus:
    temporal_contract = load_temporal_contract(temporal_config_path)
    source = load_level_probe_corpus(
        source_bulk_manifest=source_bulk_manifest,
        cache_run_manifest=cache_run_manifest,
        expected_spec=expected_spec,
        level=level,
        history_sample_count=temporal_contract.history_sample_count,
        split_plan_path=split_plan_path,
    )
    return build_information_state_corpus(
        source=source,
        temporal_contract=temporal_contract,
    )
