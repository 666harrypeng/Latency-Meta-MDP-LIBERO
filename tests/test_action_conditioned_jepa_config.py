from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

MODEL_PATH = Path("configs/belief/action_conditioned_jepa/model.yaml")
L1_PATH = Path("configs/belief/action_conditioned_jepa/l1.yaml")
L2_PATH = Path("configs/belief/action_conditioned_jepa/l2.yaml")
L3_PATH = Path("configs/belief/action_conditioned_jepa/l3.yaml")


def test_l3_config_resolves_locked_architecture() -> None:
    from latency_meta_mdp.belief.action_conditioned_jepa.config import (
        load_action_conditioned_jepa_config,
    )

    config = load_action_conditioned_jepa_config(
        model_path=MODEL_PATH,
        level_path=L3_PATH,
    )

    assert config.model_config_id == "action_conditioned_jepa_return_belief"
    assert config.level_config_id == "action_conditioned_jepa_l3"
    assert config.level == 3
    assert config.formal_tick_us == 20_000
    assert config.history_ticks == 6
    assert config.executed_control_ticks == 5
    assert config.maximum_delay_ticks == 20
    assert config.prediction_horizon == 50
    assert config.action_dim == 7
    assert config.predictor_width == 384
    assert config.predictor_depth == 6
    assert config.predictor_heads == 16
    assert config.mlp_ratio == 4.0
    assert config.qkv_bias is True
    assert config.layer_norm_epsilon == 1e-6
    assert config.dropout == 0.0
    assert config.attention_dropout == 0.0
    assert config.drop_path == 0.0
    assert config.adaln_init_scale_factor == 10
    assert config.activation == "gelu"
    assert config.position_encoding == "rope_time_y_x"
    assert config.view_identity_encoding == "additive_learned"
    assert config.camera_order == ("agentview", "wrist")
    assert config.camera_sources == ("agentview", "robot0_eye_in_hand")
    assert config.patch_token_count == 196
    assert config.visual_feature_dim == 384
    assert config.proprio_dim == 16
    assert config.temporal_attention == "block_causal"
    assert config.temporal_attention_window == 6
    assert config.control_conditioning == "per_transition_adaln"
    assert config.control_normalization == "controller_native"
    assert config.visual_normalization == "frozen_dino_coordinate"
    assert config.proprio_input_normalization == "per_level_nominal_train"
    assert config.future_proprio_output == "physical_si"
    assert config.one_step_context == "teacher_forced"
    assert config.rollout_context == "predicted_stop_gradient"
    assert config.rollout_endpoint_sampling == "stratified_uniform_2_20"
    assert config.visual_loss_weight == 1.0
    assert config.proprio_loss_weight == 1.0
    assert config.public_latent_dtype == "float16"
    assert config.internal_compute_dtype == "bfloat16"
    assert config.temporal_contract.contract_id == "h50_e25_d20_k6_v1"
    assert config.action_contract.contract_id == "panda_osc_pose_delta_v1"
    assert config.vision_encoder.encoder_id == "dinov3_vits16_lvd1689m_224_v1"
    assert config.upstream_reference.reference_id == "jepa_wms_adaln_depth6_metaworld"
    assert config.launch_support.minimum_cursor == 5
    assert config.launch_support.maximum_cursor == 25
    assert config.launch_support.remaining_controls_at_latest_cursor == 25


def test_level_configs_change_only_level_identity() -> None:
    from latency_meta_mdp.belief.action_conditioned_jepa.config import (
        load_action_conditioned_jepa_config,
    )

    configs = tuple(
        load_action_conditioned_jepa_config(model_path=MODEL_PATH, level_path=path)
        for path in (L1_PATH, L2_PATH, L3_PATH)
    )

    assert tuple(config.level for config in configs) == (1, 2, 3)
    assert tuple(config.level_config_id for config in configs) == (
        "action_conditioned_jepa_l1",
        "action_conditioned_jepa_l2",
        "action_conditioned_jepa_l3",
    )
    shared = [
        (
            config.history_ticks,
            config.maximum_delay_ticks,
            config.predictor_width,
            config.predictor_depth,
            config.camera_order,
        )
        for config in configs
    ]
    assert shared[0] == shared[1] == shared[2]


def test_model_config_rejects_latency_law_inside_predictor(tmp_path: Path) -> None:
    from latency_meta_mdp.belief.action_conditioned_jepa.config import (
        load_action_conditioned_jepa_config,
    )

    path = tmp_path / "model.yaml"
    path.write_text(
        MODEL_PATH.read_text(encoding="utf-8") + "\nlatency_probabilities: beta_8_65\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fields"):
        load_action_conditioned_jepa_config(model_path=path, level_path=L3_PATH)


def test_level_config_rejects_training_or_model_fields(tmp_path: Path) -> None:
    from latency_meta_mdp.belief.action_conditioned_jepa.config import (
        load_action_conditioned_jepa_config,
    )

    path = tmp_path / "l3.yaml"
    path.write_text(
        "schema_version: 1\nconfig_id: action_conditioned_jepa_l3\nlevel: 3\nbatch_size: 32\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fields"):
        load_action_conditioned_jepa_config(model_path=MODEL_PATH, level_path=path)


def test_launch_support_is_derived_from_h50_e25_d20_k6() -> None:
    from latency_meta_mdp.belief.action_conditioned_jepa.contracts import (
        JepaLaunchSupportContract,
    )

    contract = JepaLaunchSupportContract(
        minimum_cursor=5,
        maximum_cursor=25,
        prediction_horizon=50,
        maximum_delay_ticks=20,
    )

    assert contract.remaining_controls_at_latest_cursor == 25
    assert contract.is_belief_ready(5)
    assert contract.is_belief_ready(25)
    assert not contract.is_belief_ready(4)
    assert not contract.is_belief_ready(26)
    contract.require_supported_launch_cursor(5)
    contract.require_supported_launch_cursor(25)
    with pytest.raises(ValueError, match="availability interval"):
        contract.require_supported_launch_cursor(4)
    with pytest.raises(ValueError, match="buffer coverage"):
        JepaLaunchSupportContract(
            minimum_cursor=5,
            maximum_cursor=31,
            prediction_horizon=50,
            maximum_delay_ticks=20,
        )


def test_resolved_config_cannot_drift_from_public_tensor_contract() -> None:
    from latency_meta_mdp.belief.action_conditioned_jepa.config import (
        load_action_conditioned_jepa_config,
    )

    config = load_action_conditioned_jepa_config(
        model_path=MODEL_PATH,
        level_path=L3_PATH,
    )
    temporal = replace(
        config.temporal_contract,
        contract_id="h50_e25_d19_k6_v1",
        maximum_delay_ticks=19,
    )

    with pytest.raises(ValueError, match="public tensor contract"):
        replace(config, temporal_contract=temporal)
