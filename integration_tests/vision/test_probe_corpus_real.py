from __future__ import annotations

from pathlib import Path

from latency_meta_mdp.vision_encoder import load_vision_encoder_spec
from latency_meta_mdp.vision_probe_data import ProbeSplit


def test_probe_corpus_uses_disjoint_episode_splits_and_k6_histories() -> None:
    from latency_meta_mdp.vision_probe_corpus import load_level_probe_corpus

    source = Path(
        "outputs/bulk/expert/panda-ball-first-tranche-e58eee5/manifest.json"
    )
    cache = Path(
        "outputs/derived/vision_features/"
        "dinov3-vits16-first-tranche-54b9562/manifest.json"
    )
    if not source.is_file() or not cache.is_file():
        return
    spec = load_vision_encoder_spec(
        Path("configs/vision/dinov3_vits16_lvd1689m_224_v1.yaml")
    )

    corpus = load_level_probe_corpus(
        source_bulk_manifest=source,
        cache_run_manifest=cache,
        expected_spec=spec,
        level=1,
        history_sample_count=6,
    )

    assert corpus.episode_counts == {
        ProbeSplit.TRAIN: 20,
        ProbeSplit.VALIDATION: 2,
        ProbeSplit.HOLDOUT: 3,
    }
    assert corpus.sample_counts[ProbeSplit.TRAIN] > 2000
    assert corpus.sample_counts[ProbeSplit.VALIDATION] > 200
    assert corpus.sample_counts[ProbeSplit.HOLDOUT] > 300
    assert corpus.seeds[ProbeSplit.TRAIN] == frozenset(range(1000, 1020))
    assert corpus.seeds[ProbeSplit.VALIDATION] == frozenset((1020, 1021))
    assert corpus.seeds[ProbeSplit.HOLDOUT] == frozenset((1022, 1023, 1024))
    assert not (
        corpus.seeds[ProbeSplit.TRAIN]
        & corpus.seeds[ProbeSplit.VALIDATION]
        | corpus.seeds[ProbeSplit.TRAIN]
        & corpus.seeds[ProbeSplit.HOLDOUT]
        | corpus.seeds[ProbeSplit.VALIDATION]
        & corpus.seeds[ProbeSplit.HOLDOUT]
    )
    first = corpus.materialize(ProbeSplit.TRAIN, 0)
    assert first.vision_history.shape == (6, 2, 196, 384)
    assert first.robot_proprio_history.shape == (6, 16)
    assert first.target_state.shape == (9,)


def test_formal_probe_corpus_uses_160_20_20_splits() -> None:
    from latency_meta_mdp.vision_probe_corpus import load_level_probe_corpus

    source = Path(
        "outputs/bulk/expert/"
        "panda-ball-formal-train-1000-1199-7571a4c/manifest.json"
    )
    cache = Path(
        "outputs/derived/vision_features/"
        "dinov3-vits16-formal-1000-1199-408dbe3/manifest.json"
    )
    if not source.is_file() or not cache.is_file():
        return
    corpus = load_level_probe_corpus(
        source_bulk_manifest=source,
        cache_run_manifest=cache,
        expected_spec=load_vision_encoder_spec(
            Path("configs/vision/dinov3_vits16_lvd1689m_224_v1.yaml")
        ),
        level=1,
        history_sample_count=6,
    )

    assert corpus.episode_counts == {
        ProbeSplit.TRAIN: 160,
        ProbeSplit.VALIDATION: 20,
        ProbeSplit.HOLDOUT: 20,
    }
    assert corpus.seeds[ProbeSplit.TRAIN] == frozenset(range(1000, 1160))
    assert corpus.seeds[ProbeSplit.VALIDATION] == frozenset(range(1160, 1180))
    assert corpus.seeds[ProbeSplit.HOLDOUT] == frozenset(range(1180, 1200))
