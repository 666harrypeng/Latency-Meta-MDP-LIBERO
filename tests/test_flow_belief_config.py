from __future__ import annotations

from pathlib import Path


def test_flow_belief_config_matches_mainline_contract() -> None:
    from latency_meta_mdp.legacy.belief.flow.config import load_flow_belief_config

    config = load_flow_belief_config(Path("configs/legacy/belief/dinov3_flow_belief_v1.yaml"))

    assert config.model_id == "dinov3_flow_belief_v1"
    assert config.history_sample_count == 6
    assert config.belief_token_count == 8
    assert config.model_dim == 192
    assert config.patch_pool_query_count == 4
    assert config.transformer_head_count == 6
    assert config.fusion_layer_count == 3
    assert config.flow_hidden_dim == 512
    assert config.flow_residual_block_count == 6
    assert config.sampled_delay_query_count == 4
    assert config.solver == "heun"
    assert config.solver_step_count == 16
    assert config.evaluation_sample_count == 32
