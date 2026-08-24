from __future__ import annotations

from pathlib import Path

import pytest


def test_gaussian_belief_encoder_is_delta_free_and_decoder_is_query_conditioned() -> None:
    torch = pytest.importorskip("torch")
    from latency_meta_mdp.gaussian_belief_config import load_gaussian_belief_config
    from latency_meta_mdp.gaussian_belief_model import GaussianBeliefModel

    config = load_gaussian_belief_config(
        Path("configs/belief/dinov3_gaussian_belief_v1.yaml")
    )
    model = GaussianBeliefModel(config).cuda().eval()
    vision = torch.randn(2, 6, 2, 196, 384, device="cuda", dtype=torch.float16)
    proprio = torch.randn(2, 6, 16, device="cuda")
    actions = torch.randn(2, 25, 7, device="cuda")
    law = torch.softmax(torch.randn(2, 20, device="cuda"), dim=-1)

    belief = model.encoder(
        vision_history=vision,
        proprio_history=proprio,
        remaining_actions=actions,
        latency_probabilities=law,
    )
    mean, log_std = model.decoder(
        belief_tokens=belief,
        delay_ticks=torch.tensor([[1, 4, 8, 20], [2, 6, 12, 19]], device="cuda"),
    )

    assert belief.shape == (2, 4, 128)
    assert mean.shape == (2, 4, 22)
    assert log_std.shape == (2, 4, 22)
    assert float(log_std.min()) >= -5.0
    assert float(log_std.max()) <= 2.0
    assert sum(parameter.numel() for parameter in model.parameters()) < 3_000_000


def test_gaussian_nll_backpropagates_through_encoder_and_decoder() -> None:
    torch = pytest.importorskip("torch")
    from latency_meta_mdp.gaussian_belief_config import load_gaussian_belief_config
    from latency_meta_mdp.gaussian_belief_model import (
        GaussianBeliefModel,
        diagonal_gaussian_nll,
    )

    model = GaussianBeliefModel(
        load_gaussian_belief_config(
            Path("configs/belief/dinov3_gaussian_belief_v1.yaml")
        )
    ).cuda()
    mean, log_std, _belief = model(
        vision_history=torch.randn(2, 6, 2, 196, 384, device="cuda", dtype=torch.float16),
        proprio_history=torch.randn(2, 6, 16, device="cuda"),
        remaining_actions=torch.randn(2, 25, 7, device="cuda"),
        latency_probabilities=torch.full((2, 20), 0.05, device="cuda"),
        delay_ticks=torch.tensor([[1, 2, 3, 4], [4, 8, 12, 20]], device="cuda"),
    )
    target = torch.randn_like(mean)

    loss = diagonal_gaussian_nll(mean=mean, log_std=log_std, target=target)
    loss.backward()

    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.encoder.parameters())
    assert all(parameter.grad is not None for parameter in model.decoder.parameters())
