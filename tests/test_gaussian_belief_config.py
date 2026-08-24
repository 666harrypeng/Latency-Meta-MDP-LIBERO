from __future__ import annotations

from pathlib import Path


def test_gaussian_belief_config_matches_locked_compact_contract() -> None:
    from latency_meta_mdp.gaussian_belief_config import load_gaussian_belief_config

    config = load_gaussian_belief_config(
        Path("configs/belief/dinov3_gaussian_belief_v1.yaml")
    )

    assert config.model_id == "dinov3_gaussian_belief_v1"
    assert config.history_sample_count == 6
    assert config.belief_token_count == 4
    assert config.model_dim == 128
    assert config.patch_pool_query_count == 2
    assert config.sampled_delay_query_count == 4
    assert config.minimum_log_std == -5.0
    assert config.maximum_log_std == 2.0
