from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from latency_meta_mdp.belief.common.feature_data import (
    FeatureBeliefSample,
    sample_feature_delay_queries,
)
from latency_meta_mdp.belief.flow.config import load_flow_belief_config
from latency_meta_mdp.belief.flow.multilaw_config import (
    load_multilaw_flow_training_config,
)
from latency_meta_mdp.belief.flow.training_data import (
    FlowBeliefDataset,
    FlowBeliefNormalization,
)
from latency_meta_mdp.vision_probe_data import ProbeSplit


def _sample(probabilities: np.ndarray) -> FeatureBeliefSample:
    return FeatureBeliefSample(
        episode_id="l1-seed-001000-attempt-000",
        level=1,
        scene_seed=1000,
        source_tick=10,
        vision_history=np.zeros((6, 2, 196, 384), dtype=np.float16),
        robot_proprio_history=np.zeros((6, 16), dtype=np.float32),
        remaining_actions=np.zeros((25, 7), dtype=np.float32),
        latency_probabilities=probabilities,
        target_delay_ticks=np.arange(1, 21, dtype=np.int64),
        target_states=np.zeros((20, 22), dtype=np.float32),
        target_interaction_mode=np.zeros(20, dtype=np.int8),
        target_absorbing=np.zeros(20, dtype=np.bool_),
    )


def test_tail_balanced_queries_preserve_law_but_cover_low_probability_bins() -> None:
    probabilities = np.full(20, 1e-6, dtype=np.float64)
    probabilities[6:9] = 1.0
    probabilities /= probabilities.sum()
    sample = _sample(probabilities)

    nominal = sample_feature_delay_queries(
        sample=sample,
        rng=np.random.default_rng(7),
        query_count=20_000,
        uniform_mix=0.0,
    )
    balanced = sample_feature_delay_queries(
        sample=sample,
        rng=np.random.default_rng(7),
        query_count=20_000,
        uniform_mix=0.10,
    )

    nominal_tail = np.count_nonzero(np.isin(nominal.delay_ticks, (1, 20)))
    balanced_tail = np.count_nonzero(np.isin(balanced.delay_ticks, (1, 20)))
    assert balanced_tail > nominal_tail + 150
    np.testing.assert_array_equal(sample.latency_probabilities, probabilities)


def test_multilaw_corpus_changes_only_episode_law_condition() -> None:
    source = Path("outputs/bulk/expert/panda-ball-formal-train-1000-1199-7571a4c/manifest.json")
    cache = Path(
        "outputs/derived/vision_features/dinov3-vits16-formal-1000-1199-408dbe3/manifest.json"
    )
    if not source.is_file() or not cache.is_file():
        pytest.skip("multi-law corpus test requires formal local artifacts")
    from latency_meta_mdp.belief.common.feature_corpus import (
        load_level_feature_belief_corpus,
    )
    from latency_meta_mdp.vision_encoder import load_vision_encoder_spec
    from latency_meta_mdp.vision_probe_data import ProbeSplit

    common = dict(
        project_root=Path.cwd(),
        source_bulk_manifest=source,
        cache_run_manifest=cache,
        expected_spec=load_vision_encoder_spec(
            Path("configs/vision/dinov3_vits16_lvd1689m_224_v1.yaml")
        ),
        temporal_config_path=Path("configs/temporal/h50_e25_d20_k6_v1.yaml"),
        latency_law_path=Path("configs/latency/truncated_beta_5_26_400ms_v1.yaml"),
        split_plan_path=Path("configs/data/formal_belief_train_val_v1.yaml"),
        level=1,
    )
    nominal = load_level_feature_belief_corpus(**common)
    multi = load_level_feature_belief_corpus(
        **common,
        latency_law_family_path=Path("configs/latency/truncated_beta_family_5_26_400ms_v1.yaml"),
    )
    nominal_sample = nominal.materialize(ProbeSplit.TRAIN, 0)
    multi_sample = multi.materialize(ProbeSplit.TRAIN, 0)

    assert multi.latency_law_family_id == "truncated_beta_family_5_26_400ms_v1"
    assert not np.array_equal(
        nominal_sample.latency_probabilities,
        multi_sample.latency_probabilities,
    )
    np.testing.assert_array_equal(nominal_sample.vision_history, multi_sample.vision_history)
    np.testing.assert_array_equal(
        nominal_sample.robot_proprio_history,
        multi_sample.robot_proprio_history,
    )
    np.testing.assert_array_equal(nominal_sample.remaining_actions, multi_sample.remaining_actions)
    np.testing.assert_array_equal(nominal_sample.target_states, multi_sample.target_states)
    assert not hasattr(multi_sample, "realized_delay_tick")


def test_multilaw_is_stable_within_episode_and_shared_across_levels() -> None:
    source = Path("outputs/bulk/expert/panda-ball-formal-train-1000-1199-7571a4c/manifest.json")
    cache = Path(
        "outputs/derived/vision_features/dinov3-vits16-formal-1000-1199-408dbe3/manifest.json"
    )
    if not source.is_file() or not cache.is_file():
        pytest.skip("multi-law corpus test requires formal local artifacts")
    from latency_meta_mdp.belief.common.feature_corpus import (
        load_level_feature_belief_corpus,
    )
    from latency_meta_mdp.vision_encoder import load_vision_encoder_spec
    from latency_meta_mdp.vision_probe_data import ProbeSplit

    spec = load_vision_encoder_spec(Path("configs/vision/dinov3_vits16_lvd1689m_224_v1.yaml"))
    corpora = [
        load_level_feature_belief_corpus(
            project_root=Path.cwd(),
            source_bulk_manifest=source,
            cache_run_manifest=cache,
            expected_spec=spec,
            temporal_config_path=Path("configs/temporal/h50_e25_d20_k6_v1.yaml"),
            latency_law_path=Path("configs/latency/truncated_beta_5_26_400ms_v1.yaml"),
            latency_law_family_path=Path(
                "configs/latency/truncated_beta_family_5_26_400ms_v1.yaml"
            ),
            split_plan_path=Path("configs/data/formal_belief_train_val_v1.yaml"),
            level=level,
        )
        for level in (1, 2, 3)
    ]
    first_l1 = corpora[0].materialize(ProbeSplit.TRAIN, 0)
    second_l1 = corpora[0].materialize(ProbeSplit.TRAIN, 1)
    first_l2 = corpora[1].materialize(ProbeSplit.TRAIN, 0)
    first_l3 = corpora[2].materialize(ProbeSplit.TRAIN, 0)

    assert first_l1.episode_id == second_l1.episode_id
    np.testing.assert_array_equal(
        first_l1.latency_probabilities,
        second_l1.latency_probabilities,
    )
    np.testing.assert_array_equal(
        first_l1.latency_probabilities,
        first_l2.latency_probabilities,
    )
    np.testing.assert_array_equal(
        first_l2.latency_probabilities,
        first_l3.latency_probabilities,
    )


def test_multilaw_training_config_locks_tail_query_floor() -> None:
    config = load_multilaw_flow_training_config(
        Path("configs/belief/dinov3_flow_belief_multilaw_v2.yaml")
    )

    assert config.training_id == "dinov3_flow_belief_multilaw_v2"
    assert config.latency_law_family_id == "truncated_beta_family_5_26_400ms_v1"
    assert config.tail_query_uniform_mix == 0.10
    assert config.early_stopping_patience == 30


def test_flow_dataset_applies_tail_mix_only_to_training_query_draws() -> None:
    probabilities = np.full(20, 1e-6, dtype=np.float64)
    probabilities[6:9] = 1.0
    probabilities /= probabilities.sum()
    sample = _sample(probabilities)
    corpus = SimpleNamespace(
        level=1,
        sample_references={split: ((0, 0),) for split in ProbeSplit},
        materialize=lambda split, offset: sample,
    )
    normalization = FlowBeliefNormalization(
        proprio_mean=np.zeros(16, dtype=np.float32),
        proprio_std=np.ones(16, dtype=np.float32),
        action_mean=np.zeros(7, dtype=np.float32),
        action_std=np.ones(7, dtype=np.float32),
        target_mean=np.zeros(22, dtype=np.float32),
        target_std=np.ones(22, dtype=np.float32),
    )
    config = replace(
        load_flow_belief_config(Path("configs/belief/dinov3_flow_belief_v1.yaml")),
        sampled_delay_query_count=20_000,
    )
    nominal = FlowBeliefDataset(
        corpus=corpus,
        split=ProbeSplit.TRAIN,
        normalization=normalization,
        config=config,
        exhaustive_queries=False,
        delay_query_uniform_mix=0.0,
    )[0]
    balanced = FlowBeliefDataset(
        corpus=corpus,
        split=ProbeSplit.TRAIN,
        normalization=normalization,
        config=config,
        exhaustive_queries=False,
        delay_query_uniform_mix=0.10,
    )[0]

    nominal_tail = np.count_nonzero(np.isin(nominal.delay_ticks, (1, 20)))
    balanced_tail = np.count_nonzero(np.isin(balanced.delay_ticks, (1, 20)))
    assert balanced_tail > nominal_tail + 150
    np.testing.assert_allclose(
        nominal.latency_probabilities,
        probabilities,
        atol=0.0,
        rtol=1e-7,
    )
    np.testing.assert_allclose(
        balanced.latency_probabilities,
        probabilities,
        atol=0.0,
        rtol=1e-7,
    )
