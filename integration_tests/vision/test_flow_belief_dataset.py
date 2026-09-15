from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from latency_meta_mdp.data.vision.contracts import load_vision_encoder_spec
from latency_meta_mdp.legacy.vision_probe_data import ProbeSplit


def _corpus_and_config():
    from latency_meta_mdp.legacy.belief.common.feature_corpus import (
        load_level_feature_belief_corpus,
    )
    from latency_meta_mdp.legacy.belief.flow.config import load_flow_belief_config

    source = Path("outputs/bulk/expert/panda-ball-first-tranche-e58eee5/manifest.json")
    cache = Path(
        "outputs/derived/vision_features/dinov3-vits16-first-tranche-54b9562/manifest.json"
    )
    if not source.is_file() or not cache.is_file():
        pytest.skip("Flow Belief dataset requires the local first tranche")
    corpus = load_level_feature_belief_corpus(
        project_root=Path.cwd(),
        source_bulk_manifest=source,
        cache_run_manifest=cache,
        expected_spec=load_vision_encoder_spec(
            Path("configs/models/vision/dinov3_vits16_lvd1689m_224_v1.yaml")
        ),
        temporal_config_path=Path("configs/contracts/temporal/h50_e25_d20_k6_v1.yaml"),
        latency_law_path=Path("configs/runtime/latency/truncated_beta_5_26_400ms_v1.yaml"),
        level=1,
    )
    config = load_flow_belief_config(Path("configs/legacy/belief/dinov3_flow_belief_v1.yaml"))
    return corpus, config


def test_flow_training_dataset_owns_deterministic_epoch_noise_and_time() -> None:
    from latency_meta_mdp.legacy.belief.flow.training_data import (
        FlowBeliefDataset,
        build_flow_belief_normalization,
    )

    corpus, config = _corpus_and_config()
    dataset = FlowBeliefDataset(
        corpus=corpus,
        split=ProbeSplit.TRAIN,
        normalization=build_flow_belief_normalization(corpus),
        config=config,
        exhaustive_queries=False,
    )
    dataset.set_epoch(0)
    first = dataset[0]
    repeated = dataset[0]
    dataset.set_epoch(1)
    next_epoch = dataset[0]

    assert first.vision_history.shape == (6, 2, 196, 384)
    assert first.delay_ticks.shape == (4,)
    assert first.target_states.shape == (4, 22)
    assert first.noise.shape == (4, 22)
    assert first.flow_time.shape == (4,)
    np.testing.assert_array_equal(first.noise, repeated.noise)
    np.testing.assert_array_equal(first.flow_time, repeated.flow_time)
    assert not np.array_equal(first.noise, next_epoch.noise)
    assert not np.array_equal(first.flow_time, next_epoch.flow_time)
    assert not hasattr(first, "realized_delay_tick")
    assert first.__class__.__module__ == "latency_meta_mdp.legacy.belief.flow.training_data"


def test_flow_validation_dataset_enumerates_twenty_delays_with_fixed_draws() -> None:
    from latency_meta_mdp.legacy.belief.flow.training_data import (
        FlowBeliefDataset,
        build_flow_belief_normalization,
    )

    corpus, config = _corpus_and_config()
    dataset = FlowBeliefDataset(
        corpus=corpus,
        split=ProbeSplit.VALIDATION,
        normalization=build_flow_belief_normalization(corpus),
        config=config,
        exhaustive_queries=True,
    )

    item = dataset[0]

    assert item.delay_ticks.shape == (80,)
    assert item.target_states.shape == (80, 22)
    assert item.noise.shape == (80, 22)
    assert item.flow_time.shape == (80,)
    assert item.query_probabilities.sum() == pytest.approx(1.0)
    np.testing.assert_array_equal(
        np.unique(item.delay_ticks, return_counts=True)[1],
        np.full(20, 4),
    )


def test_formal_flow_dataset_uses_explicit_180_20_split() -> None:
    from latency_meta_mdp.legacy.belief.common.feature_corpus import (
        load_level_feature_belief_corpus,
    )
    from latency_meta_mdp.legacy.belief.flow.config import load_flow_belief_config
    from latency_meta_mdp.legacy.belief.flow.training_data import (
        FlowBeliefDataset,
        build_flow_belief_normalization,
    )

    source = Path("outputs/bulk/expert/panda-ball-formal-train-1000-1199-7571a4c/manifest.json")
    cache = Path(
        "outputs/derived/vision_features/dinov3-vits16-formal-1000-1199-408dbe3/manifest.json"
    )
    if not source.is_file() or not cache.is_file():
        pytest.skip("formal Flow dataset requires local artifacts")
    corpus = load_level_feature_belief_corpus(
        project_root=Path.cwd(),
        source_bulk_manifest=source,
        cache_run_manifest=cache,
        expected_spec=load_vision_encoder_spec(
            Path("configs/models/vision/dinov3_vits16_lvd1689m_224_v1.yaml")
        ),
        temporal_config_path=Path("configs/contracts/temporal/h50_e25_d20_k6_v1.yaml"),
        latency_law_path=Path("configs/runtime/latency/truncated_beta_5_26_400ms_v1.yaml"),
        split_plan_path=Path("configs/legacy/data/formal_belief_train_val_v1.yaml"),
        level=1,
    )
    config = load_flow_belief_config(Path("configs/legacy/belief/dinov3_flow_belief_v1.yaml"))

    assert corpus.episode_counts == {
        ProbeSplit.TRAIN: 180,
        ProbeSplit.VALIDATION: 20,
        ProbeSplit.HOLDOUT: 0,
    }
    dataset = FlowBeliefDataset(
        corpus=corpus,
        split=ProbeSplit.VALIDATION,
        normalization=build_flow_belief_normalization(corpus),
        config=config,
        exhaustive_queries=True,
    )
    assert len(dataset) == corpus.sample_counts[ProbeSplit.VALIDATION]
    assert dataset[0].delay_ticks.shape == (80,)
