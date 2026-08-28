from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from latency_meta_mdp.latency_law_family import load_episode_latency_law_family


def _family():
    return load_episode_latency_law_family(
        Path("configs/latency/truncated_beta_family_5_26_400ms_v1.yaml")
    )


def _earlier_family():
    return load_episode_latency_law_family(
        Path("configs/latency/truncated_beta_family_8_65_400ms_v1.yaml")
    )


def test_episode_law_is_deterministic_normalized_and_stable() -> None:
    family = _family()

    first = family.sample_for_episode(
        level=2,
        episode_id="l2-seed-001000-attempt-000",
        scene_seed=1000,
    )
    repeat = family.sample_for_episode(
        level=2,
        episode_id="l2-seed-001000-attempt-000",
        scene_seed=1000,
    )

    np.testing.assert_array_equal(first.probabilities, repeat.probabilities)
    assert first.delay_ticks == tuple(range(1, 21))
    assert first.probabilities.shape == (20,)
    assert first.probabilities.flags.writeable is False
    assert np.all(first.probabilities > 0.0)
    assert np.isclose(first.probabilities.sum(), 1.0, atol=1e-12, rtol=0)
    assert first.shape_alpha > 1.0
    assert first.shape_beta > 1.0
    assert 0.0 <= first.uniform_floor <= 0.04


def test_different_episode_seeds_receive_different_smooth_laws() -> None:
    family = _family()
    first = family.sample_for_episode(
        level=1,
        episode_id="l1-seed-001000-attempt-000",
        scene_seed=1000,
    )
    second = family.sample_for_episode(
        level=1,
        episode_id="l1-seed-001001-attempt-000",
        scene_seed=1001,
    )

    assert first.generation_seed != second.generation_seed
    assert not np.array_equal(first.probabilities, second.probabilities)
    assert max(abs(np.diff(first.probabilities, n=2))) < 0.08


def test_same_master_seed_uses_same_system_law_across_levels() -> None:
    family = _family()
    laws = [
        family.sample_for_episode(
            level=level,
            episode_id=f"l{level}-seed-001013-attempt-000",
            scene_seed=1013,
        )
        for level in (1, 2, 3)
    ]

    np.testing.assert_array_equal(laws[0].probabilities, laws[1].probabilities)
    np.testing.assert_array_equal(laws[1].probabilities, laws[2].probabilities)


def test_family_stays_near_nominal_physical_latency_regime() -> None:
    family = _family()
    laws = [
        family.sample_for_episode(
            level=1,
            episode_id=f"l1-seed-{seed:06d}-attempt-000",
            scene_seed=seed,
        )
        for seed in range(1000, 1200)
    ]
    means = np.asarray([law.effective_mean_seconds for law in laws])
    central_mass = np.asarray([law.probabilities[5:10].sum() for law in laws])

    assert np.quantile(means, 0.1) > 0.145
    assert np.quantile(means, 0.9) < 0.20
    assert np.quantile(central_mass, 0.1) > 0.45
    assert len({law.probability_sha256 for law in laws}) == 200


def test_family_rejects_episode_identity_mismatch() -> None:
    with pytest.raises(ValueError, match="identity"):
        _family().sample_for_episode(
            level=3,
            episode_id="l2-seed-001000-attempt-000",
            scene_seed=1000,
        )


def test_beta_8_65_family_is_earlier_but_keeps_a_small_tail() -> None:
    family = _earlier_family()
    laws = [
        family.sample_for_episode(
            level=1,
            episode_id=f"l1-seed-{seed:06d}-attempt-000",
            scene_seed=seed,
        )
        for seed in range(1000, 1200)
    ]
    means = np.asarray([law.effective_mean_seconds for law in laws])
    aggregate = np.mean([law.probabilities for law in laws], axis=0)

    assert family.family_id == "truncated_beta_family_8_65_400ms_v1"
    assert family.base_law_id == "truncated_beta_8_65_400ms_v1"
    assert family.uniform_floor_max == 0.02
    assert 0.10 < np.quantile(means, 0.1) < 0.115
    assert 0.115 < np.quantile(means, 0.5) < 0.125
    assert 0.125 < np.quantile(means, 0.9) < 0.14
    assert aggregate[3:8].sum() > 0.80
    assert 0.035 < aggregate[9:].sum() < 0.07
