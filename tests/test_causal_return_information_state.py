import inspect
from pathlib import Path

import pytest
import torch

from latency_meta_mdp.belief.causal_return.contracts import (
    load_information_state_config,
)
from latency_meta_mdp.belief.causal_return.information_state import (
    InformationStateEstimator,
)


def _model() -> InformationStateEstimator:
    config = load_information_state_config(
        Path("configs/belief/causal_return/information_state.yaml")
    )
    torch.manual_seed(7)
    model = InformationStateEstimator(config)
    model.eval()
    return model


def _inputs(batch: int = 2):
    generator = torch.Generator().manual_seed(11)
    return {
        "vision_history": torch.randn(batch, 6, 2, 196, 384, generator=generator),
        "robot_history": torch.randn(batch, 6, 16, generator=generator),
        "history_time_ms": torch.tensor([[-100, -80, -60, -40, -20, 0]] * batch),
        "history_valid_mask": torch.ones(batch, 6, dtype=torch.bool),
    }


def test_estimator_api_and_explicit_outputs_are_shape_safe() -> None:
    model = _model()
    values = _inputs()

    with torch.inference_mode():
        estimate = model(**values)

    assert tuple(inspect.signature(model.forward).parameters) == (
        "vision_history",
        "robot_history",
        "history_time_ms",
        "history_valid_mask",
    )
    assert estimate.robot_state.shape == (2, 16)
    assert estimate.object_state_mean.shape == (2, 6)
    assert estimate.object_state_log_scale.shape == (2, 6)
    assert estimate.state_tokens.shape == (2, 4, 192)
    torch.testing.assert_close(estimate.robot_state, values["robot_history"][:, -1], rtol=0, atol=0)
    assert torch.all(estimate.object_state_log_scale >= model.config.log_scale_min)
    assert torch.all(estimate.object_state_log_scale <= model.config.log_scale_max)


def test_estimator_uses_temporal_order_and_rejects_bad_timestamps() -> None:
    model = _model()
    values = _inputs(batch=1)
    reversed_frames = dict(values)
    reversed_frames["vision_history"] = torch.flip(values["vision_history"], dims=(1,))
    reversed_frames["robot_history"] = torch.flip(values["robot_history"], dims=(1,))

    with torch.inference_mode():
        original = model(**values).object_state_mean
        reordered = model(**reversed_frames).object_state_mean

    assert not torch.allclose(original, reordered)
    bad = dict(values)
    bad["history_time_ms"] = torch.tensor([[-100, -80, -40, -60, -20, 0]])
    with pytest.raises(ValueError, match="timestamps"):
        model(**bad)


def test_estimator_rejects_padded_history_before_encoding() -> None:
    model = _model()
    values = _inputs(batch=1)
    values["history_valid_mask"][:, :2] = False

    with pytest.raises(ValueError, match="validity mask"):
        model(**values)


def test_estimator_inventory_has_no_legacy_or_future_control_path() -> None:
    model = _model()
    forbidden = ("buffer", "latency", "delay", "v2")

    assert all(not any(word in name.lower() for word in forbidden) for name in model.state_dict())
    source = Path(inspect.getfile(InformationStateEstimator)).read_text(encoding="utf-8")
    assert "belief.flow" not in source
    assert "belief.gaussian" not in source
