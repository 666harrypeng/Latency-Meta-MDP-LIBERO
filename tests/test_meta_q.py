import pytest
import torch


def test_fitted_q_uses_duration_discount_terminal_mask_and_legal_action():
    from latency_meta_mdp.meta.q import fitted_q_targets

    target = fitted_q_targets(
        reward=torch.tensor([0.0, 1.0, 0.0]),
        action=torch.tensor([0, 1, 1]),
        discount=torch.tensor([0.9**4, 0.0, 0.9**7]),
        next_online=torch.tensor([[0.4, 0.2], [99.0, 99.0], [9.0, 0.3]]),
        next_target=torch.tensor([[0.5, 0.1], [99.0, 99.0], [8.0, 0.4]]),
        next_legal=torch.tensor([[True, True], [False, False], [False, True]]),
        call_cost=0.01,
        forecast_cost=0.002,
    )
    torch.testing.assert_close(
        target,
        torch.tensor(
            [
                -0.002 + 0.9**4 * 0.5,
                1 - 0.01 - 0.002,
                -0.01 - 0.002 + 0.9**7 * 0.4,
            ]
        ),
    )


def test_no_future_meta_ablation_cannot_read_future_visual_or_proprio():
    from latency_meta_mdp.meta.q import MetaQNetwork

    torch.manual_seed(27)
    model = MetaQNetwork(
        vector_mean=torch.zeros(501), vector_scale=torch.ones(501), use_future=False
    )
    visual = torch.randn(2, 2, 2, 196, 384)
    vector = torch.randn(2, 501)
    a = model(visual, vector)
    changed = visual.clone()
    changed[:, 1] = 1000
    other = vector.clone()
    other[:, 16:32] = -1000
    torch.testing.assert_close(a, model(changed, other), rtol=0, atol=0)
    assert a.shape == (2, 2) and sum(p.numel() for p in model.parameters()) < 2_000_000


def test_meta_checkpoint_cannot_silently_change_policy_or_decision_clock():
    import pytest

    from latency_meta_mdp.meta.q import validate_policy_binding

    config = {"policy_binding": {"checkpoint_step": 7560, "decision_interval_ticks": 4}}
    validate_policy_binding(config, {"checkpoint_step": 7560, "decision_interval_ticks": 4})
    with pytest.raises(ValueError, match="decision_interval_ticks"):
        validate_policy_binding(config, {"checkpoint_step": 7560, "decision_interval_ticks": 1})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP gradient regression")
def test_amp_target_forward_does_not_detach_online_training_weights():
    import copy

    from latency_meta_mdp.meta.q import MetaQNetwork
    from latency_meta_mdp.meta.train import meta_td_loss

    device = torch.device("cuda:0")
    model = MetaQNetwork(vector_mean=torch.zeros(501), vector_scale=torch.ones(501)).to(device)
    target = copy.deepcopy(model).eval().requires_grad_(False)
    visual = torch.randn(2, 2, 2, 196, 384, device=device, dtype=torch.float16)
    vector = torch.randn(2, 501, device=device)
    records = {
        k: torch.tensor(v, device=device)
        for k, v in {
            "state": [0, 1],
            "next": [1, 0],
            "action": [0, 1],
            "reward": [0.0, 1.0],
            "discount": [0.99, 0.0],
        }.items()
    }
    loss = meta_td_loss(
        model,
        target,
        visual,
        vector,
        records,
        torch.ones((2, 2), device=device, dtype=torch.bool),
        torch.arange(2, device=device),
        {"call_cost": 0.01, "forecast_cost": 0.002},
    )
    assert loss.requires_grad
    loss.backward()
    assert model.visual_projection.weight.grad.abs().sum() > 0
    assert model.head[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in target.parameters())


def test_fixed_probe_labels_are_detached_and_do_not_follow_model_updates():
    import copy

    from latency_meta_mdp.meta.q import MetaQNetwork
    from latency_meta_mdp.meta.train import meta_batch_predictions

    model = MetaQNetwork(vector_mean=torch.zeros(501), vector_scale=torch.ones(501))
    target = copy.deepcopy(model).requires_grad_(False)
    x = torch.zeros(2, 2, 2, 196, 384, dtype=torch.float16)
    v = torch.zeros(2, 501)
    records = {
        k: torch.tensor(a)
        for k, a in {
            "state": [0, 1],
            "next": [1, 0],
            "action": [0, 1],
            "reward": [0.0, 1.0],
            "discount": [1.0, 0.0],
        }.items()
    }
    args = (
        model,
        target,
        x,
        v,
        records,
        torch.ones(2, 2, dtype=torch.bool),
        torch.arange(2),
        {"q_precision": "float32", "call_cost": 0.0, "forecast_cost": 0.0},
    )
    prediction, labels, q = meta_batch_predictions(*args)
    fixed = labels.clone()
    assert not labels.requires_grad and prediction.dtype == torch.float32
    torch.nn.functional.smooth_l1_loss(prediction, labels).backward()
    assert model.head[-1].weight.grad is not None
    with torch.no_grad():
        model.head[-1].bias.add_(10)
    _, changed, _ = meta_batch_predictions(*args)
    torch.testing.assert_close(labels, fixed)
    assert q.shape == (2, 2)
