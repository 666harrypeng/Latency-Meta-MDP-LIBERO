import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned host CUDA transfer")
def test_host_visual_batch_preserves_duplicates_order_and_dtype():
    from latency_meta_mdp.meta.returns import model_visual_batch

    visual = torch.arange(12, dtype=torch.float16).reshape(4, 3)
    ids = torch.tensor([3, 0, 3, 1])
    got = model_visual_batch(visual, ids, torch.device("cuda:0"))
    assert got.dtype == visual.dtype
    torch.testing.assert_close(got.cpu(), visual[ids], rtol=0, atol=0)


def test_sequences_stop_at_episode_end_without_using_state_zero_placeholder():
    from latency_meta_mdp.meta.returns import sequence_indices

    ids, valid = sequence_indices(torch.tensor([1, -1, 3, -1]), torch.tensor([0, 1, 2]), 3)
    assert ids.tolist() == [[0, 1, -1], [1, -1, -1], [2, 3, -1]]
    assert valid.tolist() == [[True, True, False], [True, False, False], [True, True, False]]


@pytest.mark.parametrize("link", [[0], [5], [-2]])
def test_sequence_rejects_nonforward_or_out_of_range_links(link):
    from latency_meta_mdp.meta.returns import sequence_indices

    with pytest.raises(ValueError):
        sequence_indices(torch.tensor(link), torch.tensor([0]), 2)


def test_trace_propagates_real_terminal_reward_and_cuts_off_policy_suffix():
    from latency_meta_mdp.meta.returns import trace_targets

    r = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    d = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    v = torch.tensor([[0.2, 0.0], [0.2, 0.0]])
    c = torch.tensor([[0.8, 0.0], [0.0, 0.0]])
    y = trace_targets(r, d, v, c, torch.ones_like(r, dtype=torch.bool))
    torch.testing.assert_close(y, torch.tensor([0.84, 0.2]))


def test_trace_keeps_duration_and_negative_cost_but_ignores_padding():
    from latency_meta_mdp.meta.returns import trace_targets

    y = trace_targets(
        torch.tensor([[-0.1, -0.3, float("nan")]]),
        torch.tensor([[0.9**4, 0.0, float("nan")]]),
        torch.tensor([[0.2, 99.0, float("nan")]]),
        torch.tensor([[0.8, 0.0, 0.0]]),
        torch.tensor([[True, True, False]]),
    )
    torch.testing.assert_close(y, torch.tensor([-0.1 + 0.9**4 * (0.2 * 0.2 - 0.8 * 0.3)]))


def test_single_item_trace_is_one_step_and_terminal_never_bootstraps():
    from latency_meta_mdp.meta.returns import trace_targets

    y = trace_targets(
        torch.tensor([[0.0], [1.0]]),
        torch.tensor([[0.9**7], [0.0]]),
        torch.tensor([[0.4], [99.0]]),
        torch.zeros(2, 1),
        torch.ones(2, 1, dtype=torch.bool),
    )
    torch.testing.assert_close(y, torch.tensor([0.9**7 * 0.4, 1.0]))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_trace_training_uses_legal_double_q_and_gradients_only_at_start(device):
    from latency_meta_mdp.meta.q import MetaQNetwork
    from latency_meta_mdp.meta.train import meta_batch_predictions

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    model = MetaQNetwork(vector_mean=torch.zeros(501), vector_scale=torch.ones(501)).to(device)
    import copy

    target = copy.deepcopy(model).requires_grad_(False)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        for p in target.parameters():
            p.zero_()
        model.head[-1].bias.copy_(torch.tensor([9.0, 1.0]))
        target.head[-1].bias.copy_(torch.tensor([8.0, 0.2]))
    records = {
        k: torch.tensor(v)
        for k, v in {
            "state": [0, 1],
            "next": [1, 0],
            "action": [0, 1],
            "reward": [0.0, 1.0],
            "task_reward": [0.0, 1.0],
            "discount": [1.0, 0.0],
            "cost": [0.1, 1.0],
            "next_transition": [1, -1],
        }.items()
    }
    cfg = {
        "target_method": "greedy_trace",
        "max_trace_steps": 8,
        "trace_decay": 0.8,
        "budget": 10.0,
        "cost_multiplier": 1.0,
        "q_precision": "float32",
        "call_cost": 0.0,
        "forecast_cost": 0.0,
    }
    p, y, _ = meta_batch_predictions(
        model,
        target,
        torch.zeros(2, 2, 2, 196, 384),
        torch.zeros(2, 501),
        records,
        torch.tensor([[True, True], [False, True]]),
        torch.tensor([0]),
        cfg,
    )
    torch.testing.assert_close(y, torch.tensor([-0.01 + 0.2 * 0.2 + 0.8 * 0.9], device=device))
    assert p.requires_grad and not y.requires_grad
    (p - y).square().mean().backward()
    assert model.head[-1].bias.grad is not None
    assert all(p.grad is None for p in target.parameters())
