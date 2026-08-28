"""Flow-specific normalization, delay queries, and deterministic path samples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from latency_meta_mdp.belief.common.feature_corpus import FeatureBeliefCorpus
from latency_meta_mdp.belief.common.feature_data import (
    exhaustive_feature_delay_queries,
    sample_feature_delay_queries,
)
from latency_meta_mdp.belief.flow.config import FlowBeliefConfig
from latency_meta_mdp.vision_probe_data import ProbeSplit


@dataclass(frozen=True)
class FlowBeliefNormalization:
    proprio_mean: np.ndarray
    proprio_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray


@dataclass(frozen=True)
class FlowBeliefItem:
    vision_history: np.ndarray
    proprio_history: np.ndarray
    remaining_actions: np.ndarray
    latency_probabilities: np.ndarray
    delay_ticks: np.ndarray
    query_probabilities: np.ndarray
    target_states: np.ndarray
    noise: np.ndarray
    flow_time: np.ndarray
    interaction_mode: np.ndarray
    absorbing: np.ndarray


def build_flow_belief_normalization(
    corpus: FeatureBeliefCorpus,
) -> FlowBeliefNormalization:
    proprio_values = []
    action_values = []
    weighted_target_sum = np.zeros(22, dtype=np.float64)
    weighted_target_square_sum = np.zeros(22, dtype=np.float64)
    context_count = 0
    delays = np.asarray(corpus.latency_law.delay_ticks, dtype=np.int64)
    for record_offset, index_offset in corpus.sample_references[ProbeSplit.TRAIN]:
        record = corpus.records[record_offset]
        probabilities = np.asarray(record.latency_probabilities, dtype=np.float64)
        index = record.indices[index_offset]
        history = slice(index.history_start_tick, index.source_tick + 1)
        proprio_values.append(record.proprio_stream[history])
        action_stop = index.source_tick + corpus.temporal_contract.remaining_buffer_coverage
        action_values.append(record.tail.expert_actions[index.source_tick : action_stop])
        targets = record.target_state_stream[index.source_tick + delays]
        weighted_target_sum += np.sum(probabilities[:, None] * targets, axis=0)
        weighted_target_square_sum += np.sum(
            probabilities[:, None] * np.square(targets),
            axis=0,
        )
        context_count += 1
    proprio = np.concatenate(proprio_values, axis=0).astype(np.float64)
    actions = np.concatenate(action_values, axis=0).astype(np.float64)
    target_mean = weighted_target_sum / context_count
    target_variance = np.maximum(
        weighted_target_square_sum / context_count - np.square(target_mean),
        0.0,
    )
    return FlowBeliefNormalization(
        proprio_mean=proprio.mean(axis=0).astype(np.float32),
        proprio_std=np.maximum(proprio.std(axis=0), 1e-6).astype(np.float32),
        action_mean=actions.mean(axis=0).astype(np.float32),
        action_std=np.maximum(actions.std(axis=0), 1e-6).astype(np.float32),
        target_mean=target_mean.astype(np.float32),
        target_std=np.maximum(np.sqrt(target_variance), 1e-6).astype(np.float32),
    )


class FlowBeliefDataset:
    def __init__(
        self,
        *,
        corpus: FeatureBeliefCorpus,
        split: ProbeSplit,
        normalization: FlowBeliefNormalization,
        config: FlowBeliefConfig,
        exhaustive_queries: bool,
        delay_query_uniform_mix: float = 0.0,
    ) -> None:
        self.corpus = corpus
        self.split = split
        self.normalization = normalization
        self.config = config
        self.exhaustive_queries = exhaustive_queries
        if not np.isfinite(delay_query_uniform_mix) or not 0.0 <= delay_query_uniform_mix < 1.0:
            raise ValueError("Flow delay-query uniform mix must lie in [0, 1)")
        self.delay_query_uniform_mix = float(delay_query_uniform_mix)
        self.references = corpus.sample_references[split]
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.references)

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("Flow dataset epoch must be a non-negative integer")
        self.epoch = epoch

    def _rng(self, offset: int) -> np.random.Generator:
        if self.exhaustive_queries:
            seed = self.config.evaluation_seed + self.corpus.level * 1_000_000 + offset
        else:
            seed = (
                self.config.random_seed
                + self.corpus.level * 1_000_000
                + self.epoch * 100_000
                + offset
            )
        return np.random.default_rng(seed)

    def __getitem__(self, offset: int) -> FlowBeliefItem:
        sample = self.corpus.materialize(self.split, offset)
        rng = self._rng(offset)
        if self.exhaustive_queries:
            base = exhaustive_feature_delay_queries(sample)
            draw_count = self.config.validation_flow_draw_count
            delay_ticks = np.repeat(base.delay_ticks, draw_count)
            target_states = np.repeat(base.target_states, draw_count, axis=0)
            query_probabilities = np.repeat(
                base.probabilities / draw_count,
                draw_count,
            )
            interaction_mode = np.repeat(base.interaction_mode, draw_count)
            absorbing = np.repeat(base.absorbing, draw_count)
        else:
            base = sample_feature_delay_queries(
                sample=sample,
                rng=rng,
                query_count=self.config.sampled_delay_query_count,
                uniform_mix=self.delay_query_uniform_mix,
            )
            delay_ticks = base.delay_ticks
            target_states = base.target_states
            query_probabilities = np.full(
                len(delay_ticks),
                1.0 / len(delay_ticks),
                dtype=np.float32,
            )
            interaction_mode = base.interaction_mode
            absorbing = base.absorbing
        query_count = len(delay_ticks)
        noise = rng.standard_normal((query_count, 22), dtype=np.float32)
        flow_time = rng.random(query_count, dtype=np.float32)
        return FlowBeliefItem(
            vision_history=np.array(sample.vision_history, copy=True),
            proprio_history=(
                (sample.robot_proprio_history - self.normalization.proprio_mean)
                / self.normalization.proprio_std
            ).astype(np.float32),
            remaining_actions=(
                (sample.remaining_actions - self.normalization.action_mean)
                / self.normalization.action_std
            ).astype(np.float32),
            latency_probabilities=np.asarray(
                sample.latency_probabilities,
                dtype=np.float32,
            ),
            delay_ticks=np.asarray(delay_ticks, dtype=np.int64),
            query_probabilities=np.asarray(query_probabilities, dtype=np.float32),
            target_states=(
                (target_states - self.normalization.target_mean) / self.normalization.target_std
            ).astype(np.float32),
            noise=noise,
            flow_time=flow_time,
            interaction_mode=np.asarray(interaction_mode, dtype=np.int8),
            absorbing=np.asarray(absorbing, dtype=np.bool_),
        )
