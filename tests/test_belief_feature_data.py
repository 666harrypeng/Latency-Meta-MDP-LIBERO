from __future__ import annotations

import numpy as np


def _sample():
    from latency_meta_mdp.belief_feature_data import FeatureBeliefSample

    probabilities = np.arange(1, 21, dtype=np.float64)
    probabilities /= probabilities.sum()
    targets = np.arange(20 * 22, dtype=np.float32).reshape(20, 22)
    return FeatureBeliefSample(
        episode_id="l2-seed-001000-attempt-000",
        level=2,
        scene_seed=1000,
        source_tick=25,
        vision_history=np.zeros((6, 2, 196, 384), dtype=np.float16),
        robot_proprio_history=np.zeros((6, 16), dtype=np.float32),
        remaining_actions=np.zeros((25, 7), dtype=np.float32),
        latency_probabilities=probabilities,
        target_delay_ticks=np.arange(1, 21, dtype=np.int64),
        target_states=targets,
        target_interaction_mode=np.arange(20, dtype=np.int8) % 4,
        target_absorbing=np.arange(20) >= 17,
    )


def test_feature_belief_sample_exposes_only_causal_encoder_inputs() -> None:
    sample = _sample()

    assert sample.vision_history.shape == (6, 2, 196, 384)
    assert sample.robot_proprio_history.shape == (6, 16)
    assert sample.remaining_actions.shape == (25, 7)
    assert sample.latency_probabilities.shape == (20,)
    assert sample.target_states.shape == (20, 22)
    assert not hasattr(sample, "realized_delay_tick")
    assert not hasattr(sample, "query_delay_tick")
    assert not sample.vision_history.flags.writeable
    assert not sample.target_states.flags.writeable


def test_sampled_decoder_queries_are_seeded_weighted_and_target_aligned() -> None:
    from latency_meta_mdp.belief_feature_data import sample_feature_delay_queries

    sample = _sample()
    left = sample_feature_delay_queries(
        sample=sample,
        rng=np.random.default_rng(7),
        query_count=4,
    )
    right = sample_feature_delay_queries(
        sample=sample,
        rng=np.random.default_rng(7),
        query_count=4,
    )

    np.testing.assert_array_equal(left.delay_ticks, right.delay_ticks)
    np.testing.assert_array_equal(left.target_states, right.target_states)
    assert left.delay_ticks.shape == (4,)
    assert left.target_states.shape == (4, 22)
    for row, delay in enumerate(left.delay_ticks):
        np.testing.assert_array_equal(
            left.target_states[row],
            sample.target_states[int(delay) - 1],
        )


def test_exhaustive_decoder_queries_preserve_all_twenty_delay_branches() -> None:
    from latency_meta_mdp.belief_feature_data import exhaustive_feature_delay_queries

    sample = _sample()
    queries = exhaustive_feature_delay_queries(sample)

    np.testing.assert_array_equal(queries.delay_ticks, np.arange(1, 21))
    np.testing.assert_array_equal(queries.target_states, sample.target_states)
    np.testing.assert_array_equal(queries.probabilities, sample.latency_probabilities)
