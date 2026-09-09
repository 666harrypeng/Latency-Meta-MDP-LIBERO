from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from latency_meta_mdp.belief.causal_return.motion_aware_contracts import (
    MotionAwareHistoryConfig,
    MotionAwareHistoryEstimate,
    MotionAwareHistorySample,
    load_motion_aware_history_config,
)

VALID_CONFIG = """schema_version: 1
config_id: causal_return_motion_aware_history
history_sample_count: 6
formal_tick_us: 20000
camera_count: 2
patch_token_count: 196
vision_feature_dim: 384
robot_state_dim: 16
state_schema_id: franka_ball_position_velocity
object_position_dim: 3
object_velocity_dim: 3
history_dim: 192
spatiotemporal_block_count: 3
attention_head_count: 8
typed_query_count: 2
robot_gru_layer_count: 2
dropout: 0.1
batch_size: 4
learning_rate: 0.0003
weight_decay: 0.0001
gradient_clip_norm: 5.0
max_epochs: 100
early_stopping_patience: 20
early_stopping_min_delta: 0.0001
huber_delta: 1.0
random_seed: 20260830
"""


def _write_config(tmp_path: Path, text: str = VALID_CONFIG) -> Path:
    path = tmp_path / "motion_aware_history.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _sample(**overrides) -> MotionAwareHistorySample:
    values = {
        "episode_id": "l3-seed-001180-attempt-000",
        "level": 3,
        "scene_seed": 1180,
        "source_tick": 25,
        "source_phase": "pregrasp",
        "transition_offset_ticks": -3,
        "transition_offset_valid": True,
        "vision_history": np.zeros((6, 2, 196, 384), dtype=np.float16),
        "robot_history": np.zeros((6, 16), dtype=np.float32),
        "history_valid_mask": np.ones(6, dtype=np.bool_),
        "object_position_target": np.zeros(3, dtype=np.float32),
        "object_velocity_target": np.zeros(3, dtype=np.float32),
    }
    values.update(overrides)
    return MotionAwareHistorySample(**values)


def test_motion_aware_config_locks_semantic_contract(tmp_path: Path) -> None:
    config = load_motion_aware_history_config(_write_config(tmp_path))
    assert config.config_id == "causal_return_motion_aware_history"
    assert config.state_schema_id == "franka_ball_position_velocity"
    assert config.history_dim == 192
    assert config.spatiotemporal_block_count == 3
    assert config.history_dim % config.attention_head_count == 0
    assert config.typed_query_count == 2


def test_motion_aware_config_rejects_unknown_missing_and_drifted_fields(
    tmp_path: Path,
) -> None:
    unknown = VALID_CONFIG + "unexpected: 1\n"
    with pytest.raises(ValueError, match="fields"):
        load_motion_aware_history_config(_write_config(tmp_path, unknown))

    missing = VALID_CONFIG.replace("history_dim: 192\n", "")
    with pytest.raises(ValueError, match="fields"):
        load_motion_aware_history_config(_write_config(tmp_path, missing))

    drifted = VALID_CONFIG.replace("patch_token_count: 196", "patch_token_count: 4")
    with pytest.raises(ValueError, match="architecture"):
        load_motion_aware_history_config(_write_config(tmp_path, drifted))


def test_motion_aware_sample_is_immutable_and_exposes_only_deployment_inputs() -> None:
    sample = _sample()
    assert sample.vision_history.shape == (6, 2, 196, 384)
    assert sample.robot_history.shape == (6, 16)
    assert not sample.vision_history.flags.writeable
    assert not sample.robot_history.flags.writeable
    assert not sample.object_position_target.flags.writeable
    assert set(sample.deployment_inputs()) == {
        "vision_history",
        "robot_history",
        "history_valid_mask",
    }
    assert "transition_offset_ticks" not in sample.deployment_inputs()
    assert "object_position_target" not in sample.deployment_inputs()


def test_motion_aware_sample_requires_exact_physical_dtypes() -> None:
    with pytest.raises(ValueError, match="dtype"):
        _sample(robot_history=np.zeros((6, 16), dtype=np.float64))
    with pytest.raises(ValueError, match="dtype"):
        _sample(object_position_target=np.zeros(3, dtype=np.float64))


def test_motion_aware_sample_rejects_invalid_transition_and_forbidden_fields() -> None:
    with pytest.raises(ValueError, match="transition"):
        _sample(transition_offset_valid=False, transition_offset_ticks=1)

    values = asdict(_sample())
    values["remaining_actions"] = np.zeros((20, 7), dtype=np.float32)
    with pytest.raises(TypeError, match="remaining_actions"):
        MotionAwareHistorySample(**values)


def test_transition_metadata_is_present_only_for_l3() -> None:
    with pytest.raises(ValueError, match="transition"):
        _sample(level=1, transition_offset_valid=True, transition_offset_ticks=-1)
    with pytest.raises(ValueError, match="transition"):
        _sample(level=3, transition_offset_valid=False, transition_offset_ticks=0)


def test_motion_aware_estimate_requires_exact_finite_shapes() -> None:
    estimate = MotionAwareHistoryEstimate(
        history_context_tokens=torch.zeros(2, 2, 192),
        object_position_normalized=torch.zeros(2, 3),
        object_velocity_normalized=torch.zeros(2, 3),
    )
    assert estimate.history_context_tokens.shape == (2, 2, 192)

    with pytest.raises(ValueError, match="shape"):
        MotionAwareHistoryEstimate(
            history_context_tokens=torch.zeros(2, 4, 192),
            object_position_normalized=torch.zeros(2, 3),
            object_velocity_normalized=torch.zeros(2, 3),
        )
    with pytest.raises(ValueError, match="finite"):
        MotionAwareHistoryEstimate(
            history_context_tokens=torch.zeros(2, 2, 192),
            object_position_normalized=torch.full((2, 3), float("nan")),
            object_velocity_normalized=torch.zeros(2, 3),
        )


def test_motion_aware_config_dataclass_rejects_bool_integer_fields(tmp_path: Path) -> None:
    config = load_motion_aware_history_config(_write_config(tmp_path))
    values = asdict(config)
    values["history_sample_count"] = True
    with pytest.raises(ValueError, match="integer"):
        MotionAwareHistoryConfig(**values)
