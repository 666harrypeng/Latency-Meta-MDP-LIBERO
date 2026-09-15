from __future__ import annotations

from pathlib import Path


def test_temporal_state_probe_config_is_lightweight_and_k6() -> None:
    from latency_meta_mdp.legacy.vision_probe_config import load_vision_probe_config

    config = load_vision_probe_config(
        Path("configs/legacy/analysis/dinov3_temporal_state_probe_v1.yaml")
    )

    assert config.probe_id == "dinov3_temporal_state_probe_v1"
    assert config.history_sample_count == 6
    assert config.patch_projection_dim == 64
    assert config.temporal_hidden_dim == 128
    assert config.batch_size == 16
    assert config.max_epochs == 100
    assert config.early_stopping_patience == 12
    assert config.random_seed == 20260824
