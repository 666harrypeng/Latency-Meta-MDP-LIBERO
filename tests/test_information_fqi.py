import numpy as np
import pytest
import torch

from latency_meta_mdp.meta.information.fqi import fit_iteration, targets
from latency_meta_mdp.meta.information.model import StageQ
from latency_meta_mdp.meta.information.schema import ActorPacket


def test_failed_intervals_bootstrap_future_reward_and_mask_illegal_actions():
    q = torch.tensor([[0.3, 100.0], [10.0, 20.0]])
    y = targets(
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.1, 0.2]),
        torch.tensor([False, True]),
        q,
        torch.tensor([[True, False], [False, False]]),
        budget_lambda=0,
    )
    torch.testing.assert_close(y, torch.tensor([0.3, 0.0]))
    y = targets(
        torch.tensor([0.0, 0.1]),
        torch.tensor([0.1, 0.2]),
        torch.tensor([False, True]),
        q,
        torch.tensor([[True, False], [False, False]]),
        budget_lambda=0.5,
    )
    torch.testing.assert_close(y, torch.tensor([0.25, 0.0]))


def test_fitted_iteration_freezes_targets_and_learns_all_failure_terminal():
    torch.manual_seed(4)
    model = StageQ()
    p = ActorPacket(10, "G")
    sequence = [
        dict(
            packet=p,
            action=5,
            task_reward=0.0,
            cost=0.0,
            duration=0,
            terminated=True,
            next_packet=None,
        )
    ]
    before = {k: v.clone() for k, v in model.state_dict().items()}
    report = fit_iteration(model, [sequence], mode="bound", epochs=2, learning_rate=0.001)
    assert np.isfinite(report["loss"])
    assert report["transitions"] == 1 and report["target_max"] == 0
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    fit_iteration(model, [[{**sequence[0], "task_reward": 0.1}]], mode="bound")
    assert any(not torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    assert report["target_hash_before"] == report["target_hash_after"]
    with pytest.raises(ValueError, match="terminal"):
        fit_iteration(model, [[{**sequence[0], "terminated": False}]], mode="bound")


def test_balanced_source_loss_preserves_transition_weight_with_unequal_lengths():
    from latency_meta_mdp.meta.information.fqi import replay_weights

    weights = replay_weights([1, 3, 2], ["offline", "offline", "online"])
    # Three episode optimizer contributions averaged: each source has half mass.
    np.testing.assert_allclose(weights, [0.375, 0.375, 0.75])
    assert sum(w * n for w, n in zip(weights[:2], [1, 3])) / 3 == 0.5
    assert weights[2] * 2 / 3 == 0.5
    with pytest.raises(ValueError):
        replay_weights([1], ["online"])


def test_mixed_fqi_emits_separate_loss_components_and_freezes_target():
    model = StageQ()
    ep = [
        dict(
            packet=ActorPacket(10, "G"),
            action=5,
            task_reward=0.1,
            cost=0.0,
            duration=1,
            terminated=True,
            next_packet=None,
        )
    ]
    progress = []
    report = fit_iteration(
        model,
        [ep, ep],
        mode="bound",
        epochs=1,
        conservative_weight=0.01,
        sources=["offline", "online"],
        progress=progress.append,
    )
    assert report["target_hash_before"] == report["target_hash_after"]
    fits = [v for v in progress if v["phase"] == "fit"]
    assert {v["source"] for v in fits} == {"offline", "online"}
    for v in fits:
        assert v["loss"] == pytest.approx(v["huber_loss"] + 0.01 * v["conservative_loss"])
