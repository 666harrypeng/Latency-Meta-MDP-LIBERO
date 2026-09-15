from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from latency_meta_mdp.legacy.belief.causal_return.motion_aware_contracts import (
    load_motion_aware_history_config,
)
from latency_meta_mdp.legacy.belief.causal_return.spatiotemporal_history import (
    FactorizedSpatiotemporalBlock,
    MotionAwareHistoryEncoder,
)


def _config():
    return load_motion_aware_history_config(
        Path("configs/legacy/belief/causal_return/motion_aware_history.yaml")
    )


def test_factorized_block_preserves_patch_grid() -> None:
    block = FactorizedSpatiotemporalBlock(
        history_dim=192,
        attention_head_count=8,
        dropout=0.0,
    ).eval()
    vision = torch.randn(2, 6, 2, 196, 192)
    valid = torch.ones(2, 6, dtype=torch.bool)
    output = block(vision, history_valid_mask=valid)
    assert output.shape == vision.shape
    assert not any("pool" in name for name, _ in block.named_parameters())


def test_factorized_block_is_time_causal() -> None:
    torch.manual_seed(20260830)
    block = FactorizedSpatiotemporalBlock(
        history_dim=192,
        attention_head_count=8,
        dropout=0.0,
    ).eval()
    vision = torch.randn(1, 6, 2, 196, 192)
    valid = torch.ones(1, 6, dtype=torch.bool)
    original = block(vision, history_valid_mask=valid)

    future_changed = vision.clone()
    future_changed[:, 5] += 10.0
    future = block(future_changed, history_valid_mask=valid)
    torch.testing.assert_close(original[:, :5], future[:, :5], atol=1e-6, rtol=1e-6)

    past_changed = vision.clone()
    past_changed[:, 0] += 10.0
    past = block(past_changed, history_valid_mask=valid)
    assert not torch.allclose(original[:, 5], past[:, 5])


def test_motion_aware_encoder_has_exact_typed_outputs_and_no_early_pooling() -> None:
    torch.manual_seed(20260830)
    config = _config()
    encoder = MotionAwareHistoryEncoder(config).eval()
    estimate = encoder(
        vision_history=torch.zeros(1, 6, 2, 196, 384, dtype=torch.float16),
        robot_history_normalized=torch.zeros(1, 6, 16),
        history_valid_mask=torch.ones(1, 6, dtype=torch.bool),
    )
    assert estimate.history_context_tokens.shape == (1, 2, 192)
    assert estimate.object_position_normalized.shape == (1, 3)
    assert estimate.object_velocity_normalized.shape == (1, 3)
    assert len(encoder.spatiotemporal_blocks) == 3
    assert encoder.pose_query.shape == (1, 192)
    assert encoder.motion_query.shape == (1, 192)
    assert not hasattr(encoder, "spatial_pool_queries")
    assert not hasattr(encoder, "object_log_scale_head")


def test_motion_aware_encoder_uses_latest_history_and_rejects_invalid_inputs() -> None:
    torch.manual_seed(20260830)
    encoder = MotionAwareHistoryEncoder(_config()).eval()
    vision = torch.randn(1, 6, 2, 196, 384, dtype=torch.float16)
    robot = torch.randn(1, 6, 16)
    valid = torch.ones(1, 6, dtype=torch.bool)
    original = encoder(
        vision_history=vision,
        robot_history_normalized=robot,
        history_valid_mask=valid,
    )
    changed = vision.clone()
    changed[:, -1] += 1.0
    updated = encoder(
        vision_history=changed,
        robot_history_normalized=robot,
        history_valid_mask=valid,
    )
    assert not torch.allclose(
        original.object_velocity_normalized,
        updated.object_velocity_normalized,
    )

    invalid = valid.clone()
    invalid[:, 0] = False
    with pytest.raises(ValueError, match="valid"):
        encoder(
            vision_history=vision,
            robot_history_normalized=robot,
            history_valid_mask=invalid,
        )


def test_motion_aware_encoder_source_and_state_dict_preserve_isolation() -> None:
    encoder = MotionAwareHistoryEncoder(_config())
    forbidden = ("buffer", "latency", "delay", "v2")
    assert all(
        all(token not in name.lower() for token in forbidden) for name in encoder.state_dict()
    )
    source = Path(inspect.getfile(MotionAwareHistoryEncoder)).read_text(encoding="utf-8")
    assert "information_state" not in source
    assert "InformationStateEstimator" not in source
    assert "mean(dim=1)" not in source
    assert "log_scale" not in source
