from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
import torch

from latency_meta_mdp.belief.causal_return.contracts import (
    CurrentStateEstimate,
    InformationStateSample,
    load_information_state_config,
)


def _sample() -> InformationStateSample:
    return InformationStateSample(
        episode_id="l1-seed-001180-attempt-000",
        level=1,
        scene_seed=1180,
        source_tick=25,
        source_phase="pregrasp",
        motion_curvature=0.0,
        motion_transition_distance_ticks=-1,
        vision_history=np.zeros((6, 2, 196, 384), dtype=np.float16),
        robot_history=np.zeros((6, 16), dtype=np.float32),
        history_time_ms=np.asarray([-100, -80, -60, -40, -20, 0], dtype=np.int64),
        history_valid_mask=np.ones(6, dtype=np.bool_),
        object_state_target=np.zeros(6, dtype=np.float32),
    )


def test_information_state_config_has_semantic_identity_and_k6_contract() -> None:
    config = load_information_state_config(
        Path("configs/belief/causal_return/information_state.yaml")
    )

    assert config.config_id == "causal_return_information_state"
    assert config.history_sample_count == 6
    assert config.formal_tick_ms == 20
    assert config.model_dim == 192
    assert config.state_token_count == 4
    assert config.spatial_pool_query_count == 4
    assert config.temporal_fusion_layer_count == 3
    assert config.attention_head_count == 6
    assert config.object_state_dim == 6
    assert "v2" not in repr(config).lower()


def test_information_state_sample_is_typed_readonly_and_leakage_free() -> None:
    sample = _sample()

    assert sample.vision_history.shape == (6, 2, 196, 384)
    assert sample.robot_history.shape == (6, 16)
    assert sample.model_inputs().keys() == {
        "vision_history",
        "robot_history",
        "history_time_ms",
        "history_valid_mask",
    }
    assert all(not value.flags.writeable for value in sample.model_inputs().values())
    assert "remaining_actions" not in {field.name for field in fields(sample)}
    assert "latency_probabilities" not in {field.name for field in fields(sample)}
    assert "motion_curvature" not in sample.model_inputs()
    assert "motion_transition_distance_ticks" not in sample.model_inputs()
    with pytest.raises(TypeError):
        InformationStateSample(
            **sample.__dict__,
            remaining_actions=np.zeros((25, 7), dtype=np.float32),
        )


def test_information_state_sample_rejects_bad_time_or_padding() -> None:
    sample = _sample()
    with pytest.raises(ValueError, match="timestamps"):
        InformationStateSample(
            **{**sample.__dict__, "history_time_ms": np.asarray([-80, -60, -40, -20, 0, 20])}
        )
    with pytest.raises(ValueError, match="valid"):
        InformationStateSample(
            **{
                **sample.__dict__,
                "history_valid_mask": np.asarray([False, True, True, True, True, True]),
            }
        )


def test_current_state_estimate_has_explicit_semantics() -> None:
    estimate = CurrentStateEstimate(
        robot_state=torch.zeros(2, 16),
        object_state_mean=torch.zeros(2, 6),
        object_state_log_scale=torch.zeros(2, 6),
        state_tokens=torch.zeros(2, 4, 192),
    )

    assert estimate.robot_state.shape == (2, 16)
    assert estimate.object_state_mean.shape == (2, 6)
    assert estimate.object_state_log_scale.shape == (2, 6)
    assert estimate.state_tokens.shape == (2, 4, 192)
