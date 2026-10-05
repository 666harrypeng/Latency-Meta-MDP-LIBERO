import numpy as np
import pytest

from latency_meta_mdp.runtime.bootstrap_cache import CachedBootstrapPolicy
from latency_meta_mdp.runtime.policy_execution import PolicyObservation


def test_cached_bootstrap_preserves_actions_noise_sequence_and_observation_binding(tmp_path):
    obs = PolicyObservation(
        0, np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4, 3), np.uint8), np.zeros(16)
    )
    rng = np.random.default_rng(7)
    calls = []

    def native(observation):
        calls.append(observation)
        return {"actions": rng.standard_normal((50, 32), dtype=np.float32)[:, :7]}

    path = tmp_path / "bootstrap.npz"
    first = CachedBootstrapPolicy(
        path, identity={"revision": "a"}, noise_rng=rng, native_policy=native
    )
    expected = first(obs)["actions"]
    after = rng.standard_normal(10)
    fresh = np.random.default_rng(7)
    cached = CachedBootstrapPolicy(path, identity={"revision": "a"}, noise_rng=fresh)
    np.testing.assert_array_equal(cached(obs)["actions"], expected)
    np.testing.assert_array_equal(fresh.standard_normal(10), after)
    assert len(calls) == 1
    assert cached.last_record["cache_hit"] is True
    assert cached.last_record["native_wall_ns"] >= 0
    from dataclasses import replace

    obs = replace(obs, image=np.ones_like(obs.image))
    with pytest.raises(ValueError, match="observation"):
        cached(obs)


def test_bootstrap_cache_rejects_wrong_model_and_missing_file(tmp_path):
    obs = PolicyObservation(
        0, np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4, 3), np.uint8), np.zeros(16)
    )
    rng = np.random.default_rng(0)
    path = tmp_path / "bootstrap.npz"
    with pytest.raises(FileNotFoundError):
        CachedBootstrapPolicy(path, identity={}, noise_rng=rng)(obs)
    CachedBootstrapPolicy(
        path,
        identity={"revision": "a"},
        noise_rng=rng,
        native_policy=lambda obs: {"actions": np.zeros((50, 7))},
    )(obs)
    with pytest.raises(ValueError, match="identity"):
        CachedBootstrapPolicy(path, identity={"revision": "b"}, noise_rng=rng)(obs)
