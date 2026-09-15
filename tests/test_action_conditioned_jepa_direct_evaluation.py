from dataclasses import replace
from pathlib import Path
from types import MethodType

import pytest
import torch
from test_action_conditioned_jepa_direct_prediction import _query
from test_action_conditioned_jepa_rollout import _normalization

from latency_meta_mdp.belief.jepa.backbone import ActionConditionedJepaPredictor
from latency_meta_mdp.belief.jepa.config import (
    load_action_conditioned_jepa_config,
)
from latency_meta_mdp.belief.jepa.contracts import FutureLatentPrediction
from latency_meta_mdp.belief.jepa.data import (
    DirectPredictionSample,
)


@pytest.fixture(scope="module")
def legacy_model():
    config = load_action_conditioned_jepa_config(
        model_path=Path("configs/models/jepa/model.yaml"),
        level_path=Path("configs/models/jepa/l3.yaml"),
        temporal_sampling_path=Path("configs/models/jepa/stride4_80ms_history_160ms.yaml"),
    )
    return ActionConditionedJepaPredictor(
        config=config, proprio_normalization=_normalization(), project_root=Path.cwd()
    )


@pytest.mark.parametrize("q", [4, 8, 12, 16, 20])
def test_legacy_endpoint_stops_after_requested_native_steps(legacy_model, monkeypatch, q):
    from latency_meta_mdp.belief.jepa.metrics import (
        predict_legacy_endpoint,
    )

    calls = []

    def step(self, *, vision_history, proprio_history, executed_controls, outgoing_control):
        calls.append(outgoing_control.clone())
        increment = outgoing_control[:, :, 0].sum(dim=1)
        return (
            vision_history[:, -1].float() + increment[:, None, None, None],
            proprio_history[:, -1] + increment[:, None],
        )

    monkeypatch.setattr(legacy_model, "predict_next", MethodType(step, legacy_model))
    query = _query(q=q)
    controls = torch.ones_like(query.executable_controls)
    controls[:, q:] = float("nan")
    query = replace(
        query,
        executable_controls=controls,
        vision_history=torch.zeros_like(query.vision_history),
        proprio_history=torch.zeros_like(query.proprio_history),
    )
    result = predict_legacy_endpoint(legacy_model, query)
    assert len(calls) == q // 4
    assert torch.all(result.visual_latents == q) and torch.all(result.proprio == q)
    assert result.target_ticks.tolist() == [10 + q]
    assert not result.visual_latents.requires_grad


def test_legacy_endpoint_rejects_non_native_queries(legacy_model):
    from latency_meta_mdp.belief.jepa.metrics import (
        predict_legacy_endpoint,
    )

    with pytest.raises(ValueError, match="native"):
        predict_legacy_endpoint(legacy_model, _query(q=7))


def test_metrics_are_coordinate_mse_and_require_paired_timestamps():
    from latency_meta_mdp.belief.jepa.metrics import (
        prediction_mse_by_example,
    )

    query = _query(q=4)
    target = DirectPredictionSample(query, torch.zeros(1, 2, 196, 384).half(), torch.zeros(1, 16))
    proprio = torch.zeros(1, 16)
    proprio[:, 0] = 3
    proprio[:, 14] = 0.01
    predicted = FutureLatentPrediction(
        query.source_ticks, query.source_ticks + 4, torch.ones_like(target.target_visual), proprio
    )
    metrics = prediction_mse_by_example(predicted, target)
    assert metrics["visual"][0] == pytest.approx(1)
    assert metrics["qpos"][0] == pytest.approx(9 / 7)
    assert metrics["qvel"][0] == 0
    assert metrics["gripper_width"][0] == pytest.approx(0.0001)
    with pytest.raises(ValueError, match="timestamp"):
        prediction_mse_by_example(replace(predicted, target_ticks=query.source_ticks + 8), target)


def test_copy_current_baseline_uses_correct_proprio_units():
    from latency_meta_mdp.belief.jepa.metrics import (
        copy_current_prediction,
    )

    query = _query(q=12)
    result = copy_current_prediction(
        query, proprio_mean=torch.arange(16), proprio_scale=torch.full((16,), 2.0)
    )
    torch.testing.assert_close(result.visual_latents, query.vision_history[:, -1], rtol=0, atol=0)
    torch.testing.assert_close(result.proprio, query.proprio_history[:, -1] * 2 + torch.arange(16))
    assert result.target_ticks.tolist() == [22]
