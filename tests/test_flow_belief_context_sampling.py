# ruff: noqa: E402

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from latency_meta_mdp.belief.common.feature_data import FeatureBeliefSample
from latency_meta_mdp.belief.flow.config import load_flow_belief_config
from latency_meta_mdp.belief.flow.context_sampling import (
    LoadedFlowQualityLevel,
    sample_flow_validation_contexts,
)
from latency_meta_mdp.belief.flow.training_data import FlowBeliefNormalization
from latency_meta_mdp.vision_probe_data import ProbeSplit


class _ZeroEncoder(torch.nn.Module):
    def forward(
        self,
        *,
        vision_history,
        proprio_history,
        remaining_actions,
        latency_probabilities,
    ):
        del proprio_history, remaining_actions, latency_probabilities
        return torch.zeros(
            (len(vision_history), 2, 4),
            device=vision_history.device,
            dtype=torch.float32,
        )


class _ZeroVectorField(torch.nn.Module):
    def forward(self, *, noisy_state, flow_time, belief_tokens, delay_ticks):
        del flow_time, belief_tokens, delay_ticks
        return torch.zeros_like(noisy_state)


class _FakeCorpus:
    def __init__(self, contexts: tuple[FeatureBeliefSample, ...]) -> None:
        self.level = 2
        self.contexts = contexts
        self.sample_references = {
            ProbeSplit.TRAIN: (),
            ProbeSplit.VALIDATION: tuple((0, index) for index in range(len(contexts))),
            ProbeSplit.HOLDOUT: (),
        }

    def materialize(self, split: ProbeSplit, offset: int) -> FeatureBeliefSample:
        assert split is ProbeSplit.VALIDATION
        return self.contexts[offset]


def _context(offset: int) -> FeatureBeliefSample:
    delays = np.arange(1, 21, dtype=np.int64)
    targets = np.repeat(delays[:, None], 22, axis=1).astype(np.float32) + offset
    return FeatureBeliefSample(
        episode_id=f"episode-{offset}",
        level=2,
        scene_seed=1000 + offset,
        source_tick=25 + offset,
        vision_history=np.full((6, 2, 196, 384), offset, dtype=np.float16),
        robot_proprio_history=np.full((6, 16), offset, dtype=np.float32),
        remaining_actions=np.full((25, 7), offset, dtype=np.float32),
        latency_probabilities=np.full(20, 0.05, dtype=np.float64),
        target_delay_ticks=delays,
        target_states=targets,
        target_interaction_mode=np.zeros(20, dtype=np.int8),
        target_absorbing=np.zeros(20, dtype=np.bool_),
    )


def _normalization() -> FlowBeliefNormalization:
    return FlowBeliefNormalization(
        proprio_mean=np.zeros(16, dtype=np.float32),
        proprio_std=np.ones(16, dtype=np.float32),
        action_mean=np.zeros(7, dtype=np.float32),
        action_std=np.ones(7, dtype=np.float32),
        target_mean=np.zeros(22, dtype=np.float32),
        target_std=np.ones(22, dtype=np.float32),
    )


def _loaded() -> LoadedFlowQualityLevel:
    model = SimpleNamespace(
        encoder=_ZeroEncoder(),
        vector_field=_ZeroVectorField(),
    )
    return LoadedFlowQualityLevel(
        model=model,
        normalization=_normalization(),
        level_manifest={"level": 2},
    )


@pytest.mark.parametrize("offsets", [(0,), (0, 3)])
def test_context_sampler_preserves_requested_offset_and_delay_order(offsets) -> None:
    corpus = _FakeCorpus(tuple(_context(index) for index in range(8)))
    config = load_flow_belief_config(Path("configs/belief/dinov3_flow_belief_v1.yaml"))

    sampled = sample_flow_validation_contexts(
        corpus=corpus,
        loaded=_loaded(),
        flow_config=config,
        validation_offsets=offsets,
        delay_ticks=(1, 5, 20),
        sample_count=4,
        solver="heun",
        solver_step_count=2,
        device="cpu",
    )
    repeated = sample_flow_validation_contexts(
        corpus=corpus,
        loaded=_loaded(),
        flow_config=config,
        validation_offsets=offsets,
        delay_ticks=(1, 5, 20),
        sample_count=4,
        solver="heun",
        solver_step_count=2,
        device="cpu",
    )

    assert sampled.validation_offsets == offsets
    np.testing.assert_array_equal(sampled.delay_ticks, [1, 5, 20])
    assert sampled.normalized_samples.shape == (len(offsets), 3, 4, 22)
    assert sampled.normalized_targets.shape == (len(offsets), 3, 22)
    np.testing.assert_array_equal(sampled.normalized_samples, repeated.normalized_samples)
    np.testing.assert_array_equal(sampled.normalized_targets, sampled.physical_targets)
    assert sampled.normalized_samples.flags.writeable is False
    assert set(sampled.noise_sha256_by_offset) == set(offsets)


def test_context_sampler_rejects_unordered_offsets_and_delay_queries() -> None:
    corpus = _FakeCorpus(tuple(_context(index) for index in range(8)))
    config = load_flow_belief_config(Path("configs/belief/dinov3_flow_belief_v1.yaml"))
    kwargs = {
        "corpus": corpus,
        "loaded": _loaded(),
        "flow_config": config,
        "sample_count": 4,
        "solver": "heun",
        "solver_step_count": 2,
        "device": "cpu",
    }
    with pytest.raises(ValueError, match="offsets"):
        sample_flow_validation_contexts(
            validation_offsets=(3, 0),
            delay_ticks=(1, 5, 20),
            **kwargs,
        )
    with pytest.raises(ValueError, match="delay ticks"):
        sample_flow_validation_contexts(
            validation_offsets=(0, 3),
            delay_ticks=(1, 20, 5),
            **kwargs,
        )
