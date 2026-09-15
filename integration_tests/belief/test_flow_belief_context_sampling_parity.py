# ruff: noqa: E402

from __future__ import annotations

import numpy as np
import pytest

from latency_meta_mdp.io.paths import repository_root

torch = pytest.importorskip("torch")

from latency_meta_mdp.data.vision.contracts import load_vision_encoder_spec
from latency_meta_mdp.legacy.belief.common.feature_corpus import (
    load_level_feature_belief_corpus,
)
from latency_meta_mdp.legacy.belief.flow.config import load_flow_belief_config
from latency_meta_mdp.legacy.belief.flow.context_sampling import (
    load_verified_flow_quality_level,
    sample_flow_validation_contexts,
)

_PROJECT_ROOT = repository_root()
_SOURCE = (
    _PROJECT_ROOT / "outputs/bulk/expert/panda-ball-formal-train-1000-1199-7571a4c/manifest.json"
)
_CACHE = (
    _PROJECT_ROOT
    / "outputs/derived/vision_features/dinov3-vits16-formal-1000-1199-408dbe3/manifest.json"
)
_FLOW = _PROJECT_ROOT / "outputs/analysis/flow_belief/dinov3-formal-180-20-f6caf55"
_QUALITY = (
    _PROJECT_ROOT
    / "outputs/analysis/flow_belief_quality_samples/dinov3-formal-v1-6a85f90/L1/samples.npz"
)


def test_reusable_context_sampler_matches_canonical_l1_quality_samples() -> None:
    required = (_SOURCE, _CACHE, _FLOW / "L1/manifest.json", _QUALITY)
    if any(not path.is_file() for path in required):
        pytest.skip("context-sampling parity requires formal Flow artifacts")
    flow_config = load_flow_belief_config(
        _PROJECT_ROOT / "configs/legacy/belief/dinov3_flow_belief_v1.yaml"
    )
    corpus = load_level_feature_belief_corpus(
        project_root=_PROJECT_ROOT,
        source_bulk_manifest=_SOURCE,
        cache_run_manifest=_CACHE,
        expected_spec=load_vision_encoder_spec(
            _PROJECT_ROOT / "configs/models/vision/dinov3_vits16_lvd1689m_224_v1.yaml"
        ),
        temporal_config_path=_PROJECT_ROOT / "configs/contracts/temporal/h50_e25_d20_k6_v1.yaml",
        latency_law_path=_PROJECT_ROOT
        / "configs/runtime/latency/truncated_beta_5_26_400ms_v1.yaml",
        split_plan_path=_PROJECT_ROOT / "configs/legacy/data/formal_belief_train_val_v1.yaml",
        level=1,
    )
    with np.load(_QUALITY, allow_pickle=False) as expected_source:
        expected = {
            name: np.array(expected_source[name], copy=True) for name in expected_source.files
        }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded = load_verified_flow_quality_level(
        checkpoint_dir=_FLOW / "L1",
        flow_config=flow_config,
        expected_level=1,
        device=device,
    )

    sampled = sample_flow_validation_contexts(
        corpus=corpus,
        loaded=loaded,
        flow_config=flow_config,
        validation_offsets=tuple(int(value) for value in expected["validation_offsets"]),
        delay_ticks=tuple(int(value) for value in expected["display_delay_ticks"]),
        sample_count=32,
        solver="heun",
        solver_step_count=16,
        device=device,
    )

    np.testing.assert_array_equal(sampled.delay_ticks, expected["display_delay_ticks"])
    np.testing.assert_array_equal(sampled.latency_probabilities, expected["latency_probabilities"])
    np.testing.assert_array_equal(sampled.interaction_mode, expected["interaction_mode"])
    np.testing.assert_array_equal(sampled.absorbing, expected["absorbing"])
    np.testing.assert_allclose(
        sampled.normalized_samples, expected["normalized_samples"], atol=1e-4, rtol=1e-4
    )
    np.testing.assert_allclose(
        sampled.normalized_targets, expected["normalized_targets"], atol=1e-6, rtol=1e-6
    )
    np.testing.assert_allclose(
        sampled.physical_samples, expected["physical_samples"], atol=1e-5, rtol=1e-5
    )
    np.testing.assert_allclose(
        sampled.physical_targets, expected["physical_targets"], atol=1e-6, rtol=1e-6
    )
