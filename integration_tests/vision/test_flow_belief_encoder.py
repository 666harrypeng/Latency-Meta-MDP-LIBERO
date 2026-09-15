from __future__ import annotations

import inspect
from pathlib import Path

import pytest


def test_flow_encoder_is_causal_delta_free_and_fully_trainable() -> None:
    torch = pytest.importorskip("torch")
    from latency_meta_mdp.legacy.belief.flow.config import load_flow_belief_config
    from latency_meta_mdp.legacy.belief.flow.encoder import FlowBeliefEncoder

    config = load_flow_belief_config(Path("configs/legacy/belief/dinov3_flow_belief_v1.yaml"))
    encoder = FlowBeliefEncoder(config).cuda()
    signature = inspect.signature(encoder.forward)

    belief = encoder(
        vision_history=torch.randn(2, 6, 2, 196, 384, device="cuda", dtype=torch.float16),
        proprio_history=torch.randn(2, 6, 16, device="cuda"),
        remaining_actions=torch.randn(2, 25, 7, device="cuda"),
        latency_probabilities=torch.full((2, 20), 0.05, device="cuda"),
    )
    belief.square().mean().backward()

    assert "delay_ticks" not in signature.parameters
    assert belief.shape == (2, 8, 192)
    assert sum(parameter.numel() for parameter in encoder.parameters()) < 5_000_000
    assert all(parameter.grad is not None for parameter in encoder.parameters())
    assert encoder.__class__.__module__ == "latency_meta_mdp.legacy.belief.flow.encoder"
