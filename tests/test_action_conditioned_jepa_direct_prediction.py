from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from test_action_conditioned_jepa_rollout import _normalization

from latency_meta_mdp.belief.jepa.config import (
    load_action_conditioned_jepa_config,
)


def _query(q=7, batch=1):
    from latency_meta_mdp.belief.jepa.contracts import ForecastQuery

    return ForecastQuery(
        vision_history=torch.randn(batch, 3, 2, 196, 384).half(),
        proprio_history=torch.randn(batch, 3, 16),
        executed_controls=torch.randn(batch, 2, 4, 7),
        executable_controls=torch.randn(batch, 20, 7),
        control_mask=torch.arange(20)[None].expand(batch, -1) < q,
        query_ticks=torch.full((batch,), q, dtype=torch.int64),
        source_ticks=torch.full((batch,), 10, dtype=torch.int64),
    )


@pytest.fixture(scope="module")
def model():
    from latency_meta_mdp.belief.jepa.model import (
        DirectJepaPredictor,
    )

    config = load_action_conditioned_jepa_config(
        model_path=Path("configs/models/jepa/model.yaml"),
        level_path=Path("configs/models/jepa/l3.yaml"),
        temporal_sampling_path=Path("configs/models/jepa/stride4_80ms_history_160ms.yaml"),
    )
    normalization = replace(
        _normalization(),
        mean=np.arange(16, dtype=np.float32),
        scale=np.full(16, 2, dtype=np.float32),
    )
    return DirectJepaPredictor(
        backbone_config=config,
        proprio_normalization=normalization,
        project_root=Path.cwd(),
    ).eval()


def test_query_rejects_nonprefix_masks_and_out_of_support_horizons():
    query = _query()
    mask = query.control_mask.clone()
    mask[:, 0] = False
    with pytest.raises(ValueError, match="mask"):
        replace(query, control_mask=mask)
    with pytest.raises(ValueError, match="query"):
        replace(query, query_ticks=torch.tensor([21]))
    with pytest.raises(ValueError, match="source"):
        replace(query, source_ticks=torch.tensor([7]))


def test_q0_is_exact_identity_without_running_backbone(model):
    query = _query(q=0)

    def fail(*args):
        raise AssertionError("q0 must not run the predictor")

    handle = model.trunk.backbone.predictor_blocks[0].register_forward_pre_hook(fail)
    try:
        result = model.predict_at(query)
    finally:
        handle.remove()
    torch.testing.assert_close(result.visual_latents, query.vision_history[:, -1], rtol=0, atol=0)
    torch.testing.assert_close(
        result.proprio,
        query.proprio_history[:, -1] * 2 + torch.arange(16),
        rtol=0,
        atol=0,
    )
    assert result.target_ticks.tolist() == [10]


def test_query_masks_poison_and_gradient_beyond_horizon(model):
    query = _query(q=7)
    controls = query.executable_controls.clone().requires_grad_()
    query = replace(query, executable_controls=controls)
    visual, proprio = model(query)
    (visual.square().mean() + proprio.square().mean()).backward()
    assert controls.grad[:, :7].abs().sum() > 0
    assert torch.count_nonzero(controls.grad[:, 7:]) == 0
    poison = controls.detach().clone()
    poison[:, 7:] = float("nan")
    with torch.no_grad():
        poisoned = model(replace(query, executable_controls=poison))
    torch.testing.assert_close(poisoned[0], visual, rtol=0, atol=0)
    torch.testing.assert_close(poisoned[1], proprio, rtol=0, atol=0)
    model.zero_grad(set_to_none=True)


def test_ordered_controls_and_horizon_change_prediction_in_one_pass(model):
    query = _query(q=20)
    calls = []
    hook = model.trunk.backbone.predictor_blocks[0].register_forward_hook(
        lambda *args: calls.append(1)
    )
    try:
        result = model.predict_at(query)
    finally:
        hook.remove()
    assert len(calls) == 1
    assert result.visual_latents.shape == (1, 2, 196, 384)
    assert result.visual_latents.dtype == torch.float16
    assert result.proprio.shape == (1, 16)
    assert result.target_ticks.tolist() == [30]
    reversed_result = model.predict_at(
        replace(
            query,
            executable_controls=query.executable_controls.flip(1),
        )
    )
    assert not torch.equal(result.visual_latents, reversed_result.visual_latents)
    shorter = replace(
        query, query_ticks=torch.tensor([19]), control_mask=torch.arange(20)[None] < 19
    )
    assert not torch.equal(result.proprio, model.predict_at(shorter).proprio)


def test_loss_normalizes_proprio_and_detaches_supervision(model):
    from latency_meta_mdp.belief.jepa.optimization import (
        direct_prediction_loss,
    )

    pred_visual = torch.zeros(2, 2, 196, 384, requires_grad=True)
    pred_proprio = torch.zeros(2, 16, requires_grad=True)
    target_visual = torch.ones_like(pred_visual, requires_grad=True)
    target_proprio = (torch.arange(16).expand(2, -1) + 2.0).requires_grad_()
    loss, metrics = direct_prediction_loss(
        (pred_visual, pred_proprio),
        target_visual,
        target_proprio,
        proprio_mean=model.trunk.proprio_mean,
        proprio_scale=model.trunk.proprio_scale,
    )
    assert loss.item() == pytest.approx(2)
    loss.backward()
    assert target_visual.grad is None and target_proprio.grad is None
    assert pred_visual.grad.abs().sum() > 0 and pred_proprio.grad.abs().sum() > 0
    assert metrics["visual_mse"].item() == 1


def test_old_ar_checkpoint_cannot_load_as_direct(model, tmp_path):
    from safetensors.torch import save_file

    from latency_meta_mdp.belief.jepa.model import (
        load_direct_prediction_weights,
        save_direct_prediction_weights,
    )

    old = tmp_path / "old.safetensors"
    save_file(model.trunk.state_dict(), str(old))
    with pytest.raises(ValueError, match="architecture"):
        load_direct_prediction_weights(model, old)
    new = tmp_path / "direct.safetensors"
    save_direct_prediction_weights(model, new)
    load_direct_prediction_weights(model, new)
    with pytest.raises(FileExistsError):
        save_direct_prediction_weights(model, new)


def test_training_preflight_config_requires_exact_balanced_exposure():
    from latency_meta_mdp.belief.jepa.optimization import (
        DirectTrainingConfig,
    )

    config = DirectTrainingConfig()
    assert config.examples_per_epoch == 44800
    assert config.optimizer_steps_per_epoch == 175
    assert config.max_epochs * config.examples_per_epoch == 3360000
    with pytest.raises(ValueError, match="exposure"):
        replace(config, examples_per_epoch=44801)
    with pytest.raises(ValueError, match="architecture"):
        replace(config, architecture_id="legacy_ar")


def test_training_accumulation_matches_full_logical_batch():
    import copy

    from latency_meta_mdp.belief.jepa.data import (
        DirectPredictionSample,
    )
    from latency_meta_mdp.belief.jepa.optimization import (
        DirectTrainingConfig,
        train_direct_epoch,
    )

    class TinyPredictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = torch.nn.Module()
            self.trunk.register_buffer("proprio_mean", torch.zeros(16))
            self.trunk.register_buffer("proprio_scale", torch.ones(16))
            self.linear = torch.nn.Linear(16, 16)

        def forward(self, query):
            return query.vision_history[:, -1].float(), self.linear(query.proprio_history[:, -1])

    full = TinyPredictor()
    accumulated = copy.deepcopy(full)
    query = _query(batch=4)
    batch = DirectPredictionSample(query, torch.zeros(4, 2, 196, 384).half(), torch.ones(4, 16))
    chunks = [
        DirectPredictionSample(
            replace(
                query, **{name: value[start : start + 2] for name, value in vars(query).items()}
            ),
            batch.target_visual[start : start + 2],
            batch.target_proprio[start : start + 2],
        )
        for start in (0, 2)
    ]
    config = DirectTrainingConfig(global_batch_size=4, examples_per_epoch=20)
    for model, loader in ((full, [batch]), (accumulated, chunks)):
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        result = train_direct_epoch(
            model, loader, optimizer, config=config, device=torch.device("cpu")
        )
        assert result["examples_seen"] == 4
        assert len(result["updates"]) == 1
        assert result["seen_by_query"][7] == 4
    for name, parameter in full.named_parameters():
        torch.testing.assert_close(
            parameter, dict(accumulated.named_parameters())[name], rtol=1e-6, atol=1e-7
        )
