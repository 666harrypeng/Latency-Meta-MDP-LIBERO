from __future__ import annotations

from pathlib import Path

import pytest

from latency_meta_mdp.vision_encoder import load_vision_encoder_spec
from latency_meta_mdp.vision_probe_data import ProbeSplit


def test_l1_feature_belief_corpus_uses_level_specific_episode_splits() -> None:
    source = Path(
        "outputs/bulk/expert/panda-ball-first-tranche-e58eee5/manifest.json"
    )
    cache = Path(
        "outputs/derived/vision_features/"
        "dinov3-vits16-first-tranche-54b9562/manifest.json"
    )
    if not source.is_file() or not cache.is_file():
        pytest.skip("feature belief corpus requires the local first tranche")
    from latency_meta_mdp.belief_feature_corpus import load_level_feature_belief_corpus

    corpus = load_level_feature_belief_corpus(
        project_root=Path.cwd(),
        source_bulk_manifest=source,
        cache_run_manifest=cache,
        expected_spec=load_vision_encoder_spec(
            Path("configs/vision/dinov3_vits16_lvd1689m_224_v1.yaml")
        ),
        temporal_config_path=Path("configs/temporal/h50_e25_d20_k6_v1.yaml"),
        latency_law_path=Path("configs/latency/truncated_beta_5_26_400ms_v1.yaml"),
        level=1,
    )

    assert corpus.episode_counts == {
        ProbeSplit.TRAIN: 20,
        ProbeSplit.VALIDATION: 2,
        ProbeSplit.HOLDOUT: 3,
    }
    assert sum(corpus.sample_counts.values()) == 2407
    sample = corpus.materialize(ProbeSplit.TRAIN, 0)
    assert sample.vision_history.shape == (6, 2, 196, 384)
    assert sample.robot_proprio_history.shape == (6, 16)
    assert sample.remaining_actions.shape == (25, 7)
    assert sample.latency_probabilities.shape == (20,)
    assert sample.target_states.shape == (20, 22)
    assert not hasattr(sample, "agentview_history")
    assert not hasattr(sample, "realized_delay_tick")
