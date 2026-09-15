from __future__ import annotations

from pathlib import Path

import pytest


def test_flow_belief_model_jointly_trains_encoder_and_vector_field() -> None:
    torch = pytest.importorskip("torch")
    from latency_meta_mdp.legacy.belief.flow.config import load_flow_belief_config
    from latency_meta_mdp.legacy.belief.flow.model import (
        FlowBeliefModel,
        build_flow_matching_batch,
        conditional_flow_matching_loss,
    )

    config = load_flow_belief_config(Path("configs/legacy/belief/dinov3_flow_belief_v1.yaml"))
    model = FlowBeliefModel(config).cuda()
    target = torch.randn(2, 4, 22, device="cuda")
    flow_batch = build_flow_matching_batch(
        target_state=target,
        noise=torch.randn_like(target),
        flow_time=torch.rand(2, 4, device="cuda"),
    )

    velocity, belief = model(
        vision_history=torch.randn(2, 6, 2, 196, 384, device="cuda", dtype=torch.float16),
        proprio_history=torch.randn(2, 6, 16, device="cuda"),
        remaining_actions=torch.randn(2, 25, 7, device="cuda"),
        latency_probabilities=torch.full((2, 20), 0.05, device="cuda"),
        noisy_state=flow_batch.noisy_state,
        flow_time=flow_batch.flow_time,
        delay_ticks=torch.tensor([[1, 4, 8, 20], [2, 6, 12, 19]], device="cuda"),
    )
    loss = conditional_flow_matching_loss(
        predicted_velocity=velocity,
        target_velocity=flow_batch.target_velocity,
    )
    loss.backward()

    assert velocity.shape == (2, 4, 22)
    assert belief.shape == (2, 8, 192)
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.encoder.parameters())
    assert all(parameter.grad is not None for parameter in model.vector_field.parameters())
    assert sum(parameter.numel() for parameter in model.parameters()) < 8_000_000
