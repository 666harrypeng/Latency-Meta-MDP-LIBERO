from __future__ import annotations

import pytest


def test_euler_and_heun_solve_constant_velocity_exactly() -> None:
    torch = pytest.importorskip("torch")
    from latency_meta_mdp.legacy.belief.flow.sampler import integrate_flow_ode

    initial = torch.zeros(2, 3, 22)

    def velocity(state, flow_time):
        return torch.full_like(state, 2.0)

    euler = integrate_flow_ode(
        initial_state=initial,
        velocity_fn=velocity,
        solver="euler",
        step_count=4,
    )
    heun = integrate_flow_ode(
        initial_state=initial,
        velocity_fn=velocity,
        solver="heun",
        step_count=4,
    )

    assert torch.allclose(euler, torch.full_like(initial, 2.0))
    assert torch.allclose(heun, torch.full_like(initial, 2.0))


def test_heun_beats_euler_on_linear_time_velocity() -> None:
    torch = pytest.importorskip("torch")
    from latency_meta_mdp.legacy.belief.flow.sampler import integrate_flow_ode

    initial = torch.zeros(1, 1, 22)

    def velocity(state, flow_time):
        return flow_time[..., None].expand_as(state)

    euler = integrate_flow_ode(
        initial_state=initial,
        velocity_fn=velocity,
        solver="euler",
        step_count=4,
    )
    heun = integrate_flow_ode(
        initial_state=initial,
        velocity_fn=velocity,
        solver="heun",
        step_count=4,
    )
    target = torch.full_like(initial, 0.5)

    assert torch.max(torch.abs(heun - target)) < torch.max(torch.abs(euler - target))
    assert torch.allclose(heun, target)


def test_conditional_sampler_preserves_delay_and_sample_axes_deterministically() -> None:
    torch = pytest.importorskip("torch")
    from torch import nn

    from latency_meta_mdp.legacy.belief.flow.sampler import sample_flow_belief

    class ConstantField(nn.Module):
        def forward(self, *, noisy_state, flow_time, belief_tokens, delay_ticks):
            del flow_time, belief_tokens
            return torch.ones_like(noisy_state) * delay_ticks[..., None].float()

    noise = torch.zeros(2, 3, 4, 22)
    delay = torch.tensor([[1, 2, 3], [4, 5, 6]])
    belief = torch.zeros(2, 8, 192)

    first = sample_flow_belief(
        vector_field=ConstantField(),
        belief_tokens=belief,
        delay_ticks=delay,
        noise=noise,
        solver="heun",
        step_count=4,
    )
    second = sample_flow_belief(
        vector_field=ConstantField(),
        belief_tokens=belief,
        delay_ticks=delay,
        noise=noise,
        solver="heun",
        step_count=4,
    )

    assert first.shape == (2, 3, 4, 22)
    assert torch.equal(first, second)
    assert torch.allclose(first[:, :, 0, 0], delay.float())
