from __future__ import annotations

from pathlib import Path

import pytest


def test_caed902_gaussian_checkpoints_load_strictly_after_package_isolation() -> None:
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file

    from latency_meta_mdp.legacy.belief.gaussian.config import load_gaussian_belief_config
    from latency_meta_mdp.legacy.belief.gaussian.model import GaussianBeliefModel
    from latency_meta_mdp.legacy.gaussian_belief_model import (
        GaussianBeliefModel as LegacyGaussianBeliefModel,
    )

    assert LegacyGaussianBeliefModel is GaussianBeliefModel

    root = Path("outputs/analysis/gaussian_belief/dinov3-first-tranche-caed902/L1")
    if not root.is_dir():
        pytest.skip("checkpoint compatibility requires the local caed902 artifact")
    config = load_gaussian_belief_config(
        Path("configs/legacy/belief/dinov3_gaussian_belief_v1.yaml")
    )
    model = GaussianBeliefModel(config)

    full_result = model.load_state_dict(load_file(root / "model.safetensors"), strict=True)
    encoder_result = model.encoder.load_state_dict(
        load_file(root / "encoder.safetensors"),
        strict=True,
    )

    assert full_result.missing_keys == []
    assert full_result.unexpected_keys == []
    assert encoder_result.missing_keys == []
    assert encoder_result.unexpected_keys == []
    assert sum(parameter.numel() for parameter in model.parameters()) == 902_316
    assert torch.isfinite(next(model.parameters())).all()
