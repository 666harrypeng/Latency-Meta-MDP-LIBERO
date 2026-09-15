from __future__ import annotations

import pytest


def test_state_probe_has_expected_shape_and_trains_only_probe_parameters() -> None:
    torch = pytest.importorskip("torch")
    from latency_meta_mdp.legacy.vision_state_probe import TemporalVisionStateProbe

    model = TemporalVisionStateProbe(
        history_sample_count=6,
        patch_projection_dim=64,
        temporal_hidden_dim=128,
    ).cuda()
    vision = torch.randn(2, 6, 2, 196, 384, device="cuda", dtype=torch.float16)
    proprio = torch.randn(2, 6, 16, device="cuda")

    prediction = model(vision, proprio)
    prediction.square().mean().backward()

    assert prediction.shape == (2, 9)
    assert sum(parameter.numel() for parameter in model.parameters()) < 500_000
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert vision.grad is None
    assert proprio.grad is None
