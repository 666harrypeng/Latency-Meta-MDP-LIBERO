"""Verified per-level corpus for feature-backed return-belief training."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from latency_meta_mdp.belief.common.feature_data import FeatureBeliefSample
from latency_meta_mdp.belief_training_data import (
    InteractionMode,
    build_belief_training_indices,
)
from latency_meta_mdp.control import load_action_contract
from latency_meta_mdp.latency_law import TruncatedBetaLatencyLaw, load_latency_law
from latency_meta_mdp.latency_law_family import load_episode_latency_law_family
from latency_meta_mdp.return_belief_geometry import build_absorbing_return_state_stream
from latency_meta_mdp.temporal_contract import TemporalContract, load_temporal_contract
from latency_meta_mdp.terminal_absorbing_tail import (
    TerminalAbsorbingTailView,
    build_terminal_absorbing_tail,
)
from latency_meta_mdp.vision_encoder import VisionEncoderSpec
from latency_meta_mdp.vision_probe_corpus import (
    VisionProbeCorpus,
    load_level_probe_corpus,
)
from latency_meta_mdp.vision_probe_data import ProbeSplit


@dataclass(frozen=True)
class FeatureBeliefEpisodeRecord:
    episode_id: str
    level: int
    scene_seed: int
    split: ProbeSplit
    features: np.ndarray
    tail: TerminalAbsorbingTailView
    indices: tuple
    proprio_stream: np.ndarray
    target_state_stream: np.ndarray
    left_contact: np.ndarray
    right_contact: np.ndarray
    handoff: np.ndarray
    latency_probabilities: np.ndarray


@dataclass(frozen=True)
class FeatureBeliefCorpus:
    level: int
    temporal_contract: TemporalContract
    latency_law: TruncatedBetaLatencyLaw
    latency_law_family_id: str | None
    records: tuple[FeatureBeliefEpisodeRecord, ...]
    sample_references: dict[ProbeSplit, tuple[tuple[int, int], ...]]

    @property
    def episode_counts(self) -> dict[ProbeSplit, int]:
        counts = Counter(record.split for record in self.records)
        return {split: counts[split] for split in ProbeSplit}

    @property
    def sample_counts(self) -> dict[ProbeSplit, int]:
        return {split: len(self.sample_references[split]) for split in ProbeSplit}

    def materialize(self, split: ProbeSplit, offset: int) -> FeatureBeliefSample:
        record_offset, index_offset = self.sample_references[split][offset]
        record = self.records[record_offset]
        index = record.indices[index_offset]
        history = slice(index.history_start_tick, index.source_tick + 1)
        delays = np.asarray(self.latency_law.delay_ticks, dtype=np.int64)
        targets = index.source_tick + delays
        absorbing = record.tail.boundary_is_absorbing[targets]
        contact = record.left_contact[targets] | record.right_contact[targets]
        modes = np.full(len(delays), InteractionMode.FREE, dtype=np.int8)
        modes[contact] = InteractionMode.CONTACT
        modes[record.handoff[targets] == "physical"] = InteractionMode.GRASPED
        modes[absorbing] = InteractionMode.TERMINAL
        remaining_stop = index.source_tick + self.temporal_contract.remaining_buffer_coverage
        return FeatureBeliefSample(
            episode_id=record.episode_id,
            level=record.level,
            scene_seed=record.scene_seed,
            source_tick=index.source_tick,
            vision_history=record.features[history],
            robot_proprio_history=record.proprio_stream[history],
            remaining_actions=record.tail.expert_actions[index.source_tick : remaining_stop],
            latency_probabilities=record.latency_probabilities,
            target_delay_ticks=delays,
            target_states=record.target_state_stream[targets],
            target_interaction_mode=modes,
            target_absorbing=absorbing,
        )


def _proprio_stream(tail: TerminalAbsorbingTailView) -> np.ndarray:
    deployment = tail.episode.deployment
    width = deployment.gripper_qpos[:, 0] - deployment.gripper_qpos[:, 1]
    width_velocity = deployment.gripper_qvel[:, 0] - deployment.gripper_qvel[:, 1]
    return np.concatenate(
        (
            deployment.robot_qpos,
            deployment.robot_qvel,
            width[:, None],
            width_velocity[:, None],
        ),
        axis=1,
    ).astype(np.float32)


def _build_record(
    *,
    source_record,
    temporal_contract: TemporalContract,
    action_contract,
    latency_probabilities: np.ndarray,
) -> FeatureBeliefEpisodeRecord:
    episode = source_record.episode
    tail = build_terminal_absorbing_tail(
        episode=episode,
        temporal_contract=temporal_contract,
        action_contract=action_contract,
    )
    indices = build_belief_training_indices(
        tail_view=tail,
        temporal_contract=temporal_contract,
    )
    return FeatureBeliefEpisodeRecord(
        episode_id=episode.episode_id,
        level=episode.level,
        scene_seed=episode.scene_seed,
        split=source_record.indices[0].split,
        features=source_record.cache.features,
        tail=tail,
        indices=indices,
        proprio_stream=_proprio_stream(tail),
        target_state_stream=build_absorbing_return_state_stream(tail).astype(np.float32),
        left_contact=tail.extend_boundary_array(episode.supervision.left_pad_contact),
        right_contact=tail.extend_boundary_array(episode.supervision.right_pad_contact),
        handoff=tail.extend_boundary_array(episode.supervision.handoff_state),
        latency_probabilities=np.asarray(latency_probabilities, dtype=np.float64),
    )


def load_level_feature_belief_corpus(
    *,
    project_root: Path,
    source_bulk_manifest: Path,
    cache_run_manifest: Path,
    expected_spec: VisionEncoderSpec,
    temporal_config_path: Path,
    latency_law_path: Path,
    latency_law_family_path: Path | None = None,
    level: int,
    split_plan_path: Path | None = None,
) -> FeatureBeliefCorpus:
    temporal_contract = load_temporal_contract(temporal_config_path)
    latency_law = load_latency_law(latency_law_path)
    latency_law_family = (
        load_episode_latency_law_family(latency_law_family_path)
        if latency_law_family_path is not None
        else None
    )
    if latency_law_family is not None and (
        latency_law_family.base_law_id != latency_law.law_id
        or latency_law_family.latency_deadline_seconds != latency_law.latency_deadline_seconds
        or latency_law_family.control_tick_seconds != latency_law.control_tick_seconds
    ):
        raise ValueError("feature belief latency-law family and nominal law disagree")
    if latency_law.delay_ticks != tuple(range(1, temporal_contract.maximum_delay_ticks + 1)):
        raise ValueError("feature belief latency law and temporal support disagree")
    source: VisionProbeCorpus = load_level_probe_corpus(
        source_bulk_manifest=source_bulk_manifest,
        cache_run_manifest=cache_run_manifest,
        expected_spec=expected_spec,
        level=level,
        history_sample_count=temporal_contract.history_sample_count,
        split_plan_path=split_plan_path,
    )
    action_contract = load_action_contract(
        project_root.resolve() / "configs/control/panda_osc_pose_delta_v1.yaml"
    )
    records = tuple(
        _build_record(
            source_record=source_record,
            temporal_contract=temporal_contract,
            action_contract=action_contract,
            latency_probabilities=(
                latency_law_family.sample_for_episode(
                    level=source_record.episode.level,
                    episode_id=source_record.episode.episode_id,
                    scene_seed=source_record.episode.scene_seed,
                ).probabilities
                if latency_law_family is not None
                else latency_law.probabilities
            ),
        )
        for source_record in source.records
    )
    references: dict[ProbeSplit, list[tuple[int, int]]] = defaultdict(list)
    for record_offset, record in enumerate(records):
        for index_offset in range(len(record.indices)):
            references[record.split].append((record_offset, index_offset))
    return FeatureBeliefCorpus(
        level=level,
        temporal_contract=temporal_contract,
        latency_law=latency_law,
        latency_law_family_id=(
            latency_law_family.family_id if latency_law_family is not None else None
        ),
        records=records,
        sample_references={split: tuple(references[split]) for split in ProbeSplit},
    )
