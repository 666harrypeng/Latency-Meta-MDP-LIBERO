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


def test_partial_preparation_resumes_after_render_change(tmp_path, monkeypatch):
    import sys
    from contextlib import contextmanager
    from types import SimpleNamespace

    from latency_meta_mdp.envs.conveyor import config as scene_config
    from latency_meta_mdp.envs.conveyor import task
    from latency_meta_mdp.meta.information import cohort, collection, services
    from latency_meta_mdp.runtime import policy_execution

    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(random=SimpleNamespace(key=int)))
    manifest = {"episodes": [{"scene": 1}]}
    monkeypatch.setattr(
        cohort, "prepare_cohort", lambda *a, **k: ({"task_config": "test"}, manifest)
    )
    monkeypatch.setattr(scene_config, "load_conveyor_spec", lambda *a: None)
    monkeypatch.setattr(collection, "bootstrap_binding", lambda *a: {"model": "clean"})
    monkeypatch.setattr(collection, "bootstrap_observation", lambda obs: obs)
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    observation = PolicyObservation(0, image, image.copy(), np.zeros(16))
    monkeypatch.setattr(
        task,
        "make_conveyor_runtime",
        lambda *a, **k: SimpleNamespace(
            executor=SimpleNamespace(initialize=lambda: observation),
            env=SimpleNamespace(close=lambda: None),
        ),
    )

    @contextmanager
    def frozen(*a, **k):
        yield SimpleNamespace(policy=SimpleNamespace())

    monkeypatch.setattr(services, "frozen_services", frozen)
    inferred = []

    def infer(obs):
        inferred.append(obs.image.copy())
        return {"actions": np.zeros((50, 7))}

    monkeypatch.setattr(policy_execution, "InProcessOpenpiPolicy", lambda *a, **k: infer)
    cohort.prepare_bootstraps(None, tmp_path, root=tmp_path, audit_path=None)
    cached = (tmp_path / "bootstrap/seed-1.npz").read_bytes()
    image[0, 0, 0] = 1  # A later renderer can change the actual model input.
    observation = replace(observation, image=image)
    manifest["episodes"].append({"scene": 2})
    cohort.prepare_bootstraps(None, tmp_path, root=tmp_path, audit_path=None)
    assert len(inferred) == 3
    assert (tmp_path / "bootstrap/seed-1.npz").read_bytes() == cached
    assert (tmp_path / "bootstrap/seed-2.npz").is_file()
