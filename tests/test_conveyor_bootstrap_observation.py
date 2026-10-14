"""Conveyor bootstrap keys bind the pixels actually consumed by frozen pi05."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from latency_meta_mdp.runtime.bootstrap_cache import CachedBootstrapPolicy
from latency_meta_mdp.runtime.policy_execution import PolicyObservation


@pytest.fixture
def canonicalize(monkeypatch):
    # Use the pinned OpenPI client transform, without loading a model.
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "third_party/openpi/packages/openpi-client/src"))
    from latency_meta_mdp.meta.information.collection import bootstrap_observation

    return bootstrap_observation


def test_bootstrap_accepts_only_identical_model_inputs(tmp_path, canonicalize):
    image = np.full((256, 256, 3), 142, dtype=np.uint8)
    # Captured A100 image neighborhood: its center varied by one raw level.
    image[117:122, 231:236, 1] = [
        [134, 132, 136, 141, 136],
        [137, 138, 140, 143, 139],
        [139, 140, 142, 143, 143],
        [139, 140, 142, 145, 146],
        [139, 143, 145, 147, 147],
    ]
    observation = PolicyObservation(0, image, image, np.zeros(16))
    variant = image.copy()
    variant[119, 233, 1] -= 1
    equivalent = replace(observation, image=variant)
    canonical = canonicalize(observation)
    assert canonical.image.shape == (224, 224, 3)
    assert not np.array_equal(observation.image, equivalent.image)
    np.testing.assert_array_equal(canonical.image, canonicalize(equivalent).image)
    np.testing.assert_array_equal(canonicalize(canonical).image, canonical.image)

    path = tmp_path / "bootstrap.npz"
    expected = np.full((50, 7), 0.25)
    native_observations = []

    def native(obs):
        native_observations.append(obs)
        return {"actions": expected}

    cache = CachedBootstrapPolicy(
        path,
        identity={"encoding": "model-input"},
        noise_rng=np.random.default_rng(0),
        native_policy=native,
    )
    cache(canonical)
    np.testing.assert_array_equal(cache(canonicalize(equivalent))["actions"], expected)
    assert len(native_observations) == 1
    assert native_observations[0].image.shape == (224, 224, 3)
    for changed in (
        replace(observation, image=np.zeros_like(image)),
        replace(observation, wrist_image=np.zeros_like(image)),
        replace(observation, state=np.ones(16)),
        replace(observation, prompt="different task"),
    ):
        with pytest.raises(ValueError, match="observation"):
            cache(canonicalize(changed))
    changed_pixel = canonical.image.copy()
    changed_pixel[0, 0, 0] += 1
    with pytest.raises(ValueError, match="observation"):
        cache(canonicalize(replace(canonical, image=changed_pixel)))
