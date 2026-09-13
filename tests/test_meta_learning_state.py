import copy

import pytest
import torch


def setup_model():
    torch.manual_seed(27)
    model = torch.nn.Linear(3, 2)
    target = copy.deepcopy(model).requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    return model, target, optimizer


def update(model, target, optimizer, step):
    x = torch.rand(4, 3)
    y = torch.rand(4, 2) + target(x).detach()
    optimizer.zero_grad()
    (model(x) - y).square().mean().backward()
    optimizer.step()
    if step % 3 == 0:
        target.load_state_dict(model.state_dict())


def test_training_resume_restores_optimizer_target_and_rng_exactly(tmp_path):
    from latency_meta_mdp.meta_learning_state import load_learning_state, save_learning_state

    model, target, optimizer = setup_model()
    config = {"use_future": True, "policy_binding": {"policy": "frozen"}}
    for step in range(1, 5):
        update(model, target, optimizer, step)
    path = tmp_path / "resume.pt"
    save_learning_state(path, model, target, optimizer, 4, config)
    for step in range(5, 9):
        update(model, target, optimizer, step)
    restored, restored_target, restored_optimizer = setup_model()
    assert load_learning_state(path, restored, restored_target, restored_optimizer, config) == 4
    for step in range(5, 9):
        update(restored, restored_target, restored_optimizer, step)
    for a, b in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for a, b in zip(target.parameters(), restored_target.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(ValueError, match="use_future"):
        load_learning_state(path, restored, restored_target, restored_optimizer,
                            {**config, "use_future": False})


def test_learning_rate_has_a_settling_phase_and_is_extendable():
    from latency_meta_mdp.meta_learning_state import learning_rate_at

    cfg = {"learning_rate": 3e-4, "minimum_learning_rate": 3e-5,
           "lr_decay_start": 6000, "lr_decay_end": 24000}
    assert learning_rate_at(6000, cfg) == 3e-4
    assert 3e-5 < learning_rate_at(12000, cfg) < 3e-4
    assert learning_rate_at(24000, cfg) == learning_rate_at(36000, cfg) == 3e-5
