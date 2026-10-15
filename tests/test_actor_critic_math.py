import math

import pytest
import torch

from latency_meta_mdp.meta.information.actor_critic import (
    advantage_weights,
    categorical,
    expectile_loss,
    ppo_loss,
    returns_to_go,
)


def test_expectile_weights_positive_residual_and_detaches_teacher():
    teacher = torch.tensor([2.0, 0.0], requires_grad=True)
    value = torch.tensor([0.0, 2.0], requires_grad=True)
    loss = expectile_loss(value, teacher, 0.7).mean()
    assert loss.item() == pytest.approx(2.0)
    loss.backward()
    assert teacher.grad is None
    torch.testing.assert_close(value.grad, torch.tensor([-1.4, 0.6]))


def test_weight_clips_before_exponentiation_and_has_no_teacher_gradient():
    weights = advantage_weights(torch.tensor([-1.0, 0.0, 1.0e6], requires_grad=True), 0.1, 20.0)
    torch.testing.assert_close(weights, torch.tensor([math.exp(-10), 1.0, 20.0]))
    assert not weights.requires_grad


def test_masked_probabilities_and_exact_ppo_clipping_both_signs():
    old = torch.zeros(2, 3)
    legal = torch.tensor([[True, True, False]] * 2)
    new = torch.tensor([[math.log(4), 0.0, 1000.0]] * 2, requires_grad=True)
    loss, metrics = ppo_loss(
        new,
        old,
        legal,
        torch.tensor([0, 0]),
        torch.tensor([1.0, -1.0]),
        clip=0.2,
        entropy_coefficient=0.0,
    )
    # ratio1.6: positive advantage capped1.2; negative advantage retains1.6.
    torch.testing.assert_close(loss, torch.tensor([-1.2, 1.6]))
    assert metrics["kl"].mean().item() == pytest.approx(math.log(1.25))
    loss.mean().backward()
    assert new.grad[0].abs().sum() == 0
    assert new.grad[:, 2].abs().sum() == 0
    assert categorical(new, legal).probs[:, 2].sum() == 0


def test_empty_legal_mask_rejected_and_complete_returns_preserve_sparse_reward():
    with pytest.raises(ValueError, match="legal"):
        categorical(torch.zeros(1, 2), torch.zeros(1, 2, dtype=torch.bool))
    assert returns_to_go([0.0, 0.1, 0.0, 0.2], [False, False, False, True]) == pytest.approx(
        [0.3, 0.3, 0.2, 0.2]
    )
    with pytest.raises(ValueError, match="complete"):
        returns_to_go([0.1], [False])
    with pytest.raises(ValueError, match="complete"):
        returns_to_go([0.1, 0.2], [True, True])
