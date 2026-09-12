import torch


def test_fitted_q_uses_duration_discount_terminal_mask_and_legal_action():
    from latency_meta_mdp.meta_q import fitted_q_targets

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
    from latency_meta_mdp.meta_q import MetaQNetwork

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

    from latency_meta_mdp.meta_q import validate_policy_binding

    config = {"policy_binding": {"checkpoint_step": 7560, "decision_interval_ticks": 4}}
    validate_policy_binding(config, {"checkpoint_step": 7560, "decision_interval_ticks": 4})
    with pytest.raises(ValueError, match="decision_interval_ticks"):
        validate_policy_binding(config, {"checkpoint_step": 7560, "decision_interval_ticks": 1})
