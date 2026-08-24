from __future__ import annotations

import pytest


def test_straight_flow_path_matches_hand_derived_state_and_velocity() -> None:
    torch = pytest.importorskip("torch")
    from latency_meta_mdp.belief.flow.model import build_flow_matching_batch

    target = torch.full((1, 1, 22), 2.0)
    noise = torch.full((1, 1, 22), -1.0)
    flow_time = torch.full((1, 1), 0.25)

    batch = build_flow_matching_batch(
        target_state=target,
        noise=noise,
        flow_time=flow_time,
    )

    assert torch.allclose(batch.noisy_state, torch.full_like(target, -0.25))
    assert torch.allclose(batch.target_velocity, torch.full_like(target, 3.0))
    assert torch.equal(batch.noise, noise)
    assert torch.equal(batch.flow_time, flow_time)


def test_flow_path_rejects_malformed_time_or_noise() -> None:
    torch = pytest.importorskip("torch")
    from latency_meta_mdp.belief.flow.model import build_flow_matching_batch

    target = torch.zeros(2, 4, 22)
    with pytest.raises(ValueError, match="Flow time"):
        build_flow_matching_batch(
            target_state=target,
            noise=torch.zeros_like(target),
            flow_time=torch.full((2, 4), 1.1),
        )
    with pytest.raises(ValueError, match="noise"):
        build_flow_matching_batch(
            target_state=target,
            noise=torch.zeros(2, 4, 21),
            flow_time=torch.full((2, 4), 0.5),
        )
