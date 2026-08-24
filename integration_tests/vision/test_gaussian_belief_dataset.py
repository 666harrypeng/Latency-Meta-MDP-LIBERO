from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from latency_meta_mdp.vision_encoder import load_vision_encoder_spec
from latency_meta_mdp.vision_probe_data import ProbeSplit


def _corpus_and_config():
    from latency_meta_mdp.belief_feature_corpus import load_level_feature_belief_corpus
    from latency_meta_mdp.gaussian_belief_config import load_gaussian_belief_config

    source = Path(
        "outputs/bulk/expert/panda-ball-first-tranche-e58eee5/manifest.json"
    )
    cache = Path(
        "outputs/derived/vision_features/"
        "dinov3-vits16-first-tranche-54b9562/manifest.json"
    )
    if not source.is_file() or not cache.is_file():
        pytest.skip("Gaussian belief dataset requires the local first tranche")
    return (
        load_level_feature_belief_corpus(
            project_root=Path.cwd(),
            source_bulk_manifest=source,
            cache_run_manifest=cache,
            expected_spec=load_vision_encoder_spec(
                Path("configs/vision/dinov3_vits16_lvd1689m_224_v1.yaml")
            ),
            temporal_config_path=Path("configs/temporal/h50_e25_d20_k6_v1.yaml"),
            latency_law_path=Path("configs/latency/truncated_beta_5_26_400ms_v1.yaml"),
            level=1,
        ),
        load_gaussian_belief_config(
            Path("configs/belief/dinov3_gaussian_belief_v1.yaml")
        ),
    )


def test_gaussian_dataset_separates_encoder_inputs_from_decoder_queries() -> None:
    from latency_meta_mdp.gaussian_belief_training_data import (
        GaussianBeliefDataset,
        build_gaussian_belief_normalization,
    )

    corpus, config = _corpus_and_config()
    normalization = build_gaussian_belief_normalization(corpus)
    dataset = GaussianBeliefDataset(
        corpus=corpus,
        split=ProbeSplit.TRAIN,
        normalization=normalization,
        config=config,
        exhaustive_queries=False,
    )
    dataset.set_epoch(0)

    item = dataset[0]

    assert item.vision_history.shape == (6, 2, 196, 384)
    assert item.proprio_history.shape == (6, 16)
    assert item.remaining_actions.shape == (25, 7)
    assert item.latency_probabilities.shape == (20,)
    assert item.delay_ticks.shape == (4,)
    assert item.target_states.shape == (4, 22)
    assert item.query_probabilities.shape == (4,)
    assert not hasattr(item, "realized_delay_tick")
    assert np.all(np.isfinite(item.target_states))


def test_gaussian_validation_dataset_enumerates_all_delay_bins() -> None:
    from latency_meta_mdp.gaussian_belief_training_data import (
        GaussianBeliefDataset,
        build_gaussian_belief_normalization,
    )

    corpus, config = _corpus_and_config()
    dataset = GaussianBeliefDataset(
        corpus=corpus,
        split=ProbeSplit.VALIDATION,
        normalization=build_gaussian_belief_normalization(corpus),
        config=config,
        exhaustive_queries=True,
    )

    item = dataset[0]

    np.testing.assert_array_equal(item.delay_ticks, np.arange(1, 21))
    assert item.target_states.shape == (20, 22)
    assert item.query_probabilities.shape == (20,)
    assert item.query_probabilities.sum() == pytest.approx(1.0)
