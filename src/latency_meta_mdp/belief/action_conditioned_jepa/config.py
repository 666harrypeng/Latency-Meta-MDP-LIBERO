"""Exact semantic configuration for the Action-Conditioned JEPA Belief."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from latency_meta_mdp.belief.action_conditioned_jepa.contracts import (
    JepaLaunchSupportContract,
)
from latency_meta_mdp.belief.action_conditioned_jepa.upstream_adapter import (
    UpstreamReference,
    load_upstream_reference,
)
from latency_meta_mdp.control import ActionContract, load_action_contract
from latency_meta_mdp.temporal_contract import TemporalContract, load_temporal_contract
from latency_meta_mdp.vision_encoder import VisionEncoderSpec, load_vision_encoder_spec


@dataclass(frozen=True)
class _ModelSemantics:
    schema_version: int
    config_id: str
    temporal_config: str
    temporal_contract_id: str
    control_config: str
    control_contract_id: str
    vision_config: str
    vision_encoder_id: str
    upstream_reference_config: str
    upstream_reference_id: str
    predictor_width: int
    predictor_depth: int
    predictor_heads: int
    mlp_ratio: float
    qkv_bias: bool
    layer_norm_epsilon: float
    dropout: float
    attention_dropout: float
    drop_path: float
    adaln_init_scale_factor: int
    activation: str
    position_encoding: str
    view_identity_encoding: str
    camera_order: tuple[str, str]
    camera_sources: tuple[str, str]
    proprio_dim: int
    temporal_attention: str
    control_conditioning: str
    control_normalization: str
    visual_normalization: str
    proprio_input_normalization: str
    future_proprio_output: str
    one_step_context: str
    rollout_context: str
    rollout_endpoint_sampling: str
    visual_loss_weight: float
    proprio_loss_weight: float
    public_latent_dtype: str
    internal_compute_dtype: str

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.config_id != "action_conditioned_jepa_return_belief":
            raise ValueError("unsupported Action-Conditioned JEPA model config")
        for name in (
            "temporal_config",
            "control_config",
            "vision_config",
            "upstream_reference_config",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be a non-empty relative path")
        integers = (
            self.predictor_width,
            self.predictor_depth,
            self.predictor_heads,
            self.adaln_init_scale_factor,
            self.proprio_dim,
        )
        if any(type(value) is not int or value <= 0 for value in integers):
            raise ValueError("JEPA model dimensions must be positive integers")
        if self.predictor_width % self.predictor_heads:
            raise ValueError("predictor width must be divisible by the attention-head count")
        if type(self.mlp_ratio) is not float or self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if type(self.qkv_bias) is not bool:
            raise TypeError("qkv_bias must be boolean")
        if type(self.layer_norm_epsilon) is not float or self.layer_norm_epsilon <= 0:
            raise ValueError("layer_norm_epsilon must be positive")
        if any(
            type(value) is not float or not 0.0 <= value < 1.0
            for value in (self.dropout, self.attention_dropout, self.drop_path)
        ):
            raise ValueError("dropout values must be floats in [0,1)")
        if self.camera_order != ("agentview", "wrist") or self.camera_sources != (
            "agentview",
            "robot0_eye_in_hand",
        ):
            raise ValueError("JEPA camera identities or order are invalid")
        expected_strings = {
            "activation": "gelu",
            "position_encoding": "rope_time_y_x",
            "view_identity_encoding": "additive_learned",
            "temporal_attention": "block_causal",
            "control_conditioning": "per_transition_adaln",
            "control_normalization": "controller_native",
            "visual_normalization": "frozen_dino_coordinate",
            "proprio_input_normalization": "per_level_nominal_train",
            "future_proprio_output": "physical_si",
            "one_step_context": "teacher_forced",
            "rollout_context": "predicted_stop_gradient",
            "rollout_endpoint_sampling": "stratified_uniform_2_20",
            "public_latent_dtype": "float16",
            "internal_compute_dtype": "bfloat16",
        }
        if any(getattr(self, name) != value for name, value in expected_strings.items()):
            raise ValueError("Action-Conditioned JEPA model semantics are invalid")
        if (
            type(self.visual_loss_weight) is not float
            or type(self.proprio_loss_weight) is not float
            or self.visual_loss_weight <= 0
            or self.proprio_loss_weight <= 0
        ):
            raise ValueError("modality loss weights must be positive floats")


@dataclass(frozen=True)
class _LevelIdentity:
    schema_version: int
    config_id: str
    level: int

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or type(self.level) is not int
            or self.level not in (1, 2, 3)
            or self.config_id != f"action_conditioned_jepa_l{self.level}"
        ):
            raise ValueError("Action-Conditioned JEPA level identity is invalid")


@dataclass(frozen=True)
class ActionConditionedJepaConfig:
    model_config_id: str
    level_config_id: str
    level: int
    temporal_contract: TemporalContract
    action_contract: ActionContract
    vision_encoder: VisionEncoderSpec
    upstream_reference: UpstreamReference
    predictor_width: int
    predictor_depth: int
    predictor_heads: int
    mlp_ratio: float
    qkv_bias: bool
    layer_norm_epsilon: float
    dropout: float
    attention_dropout: float
    drop_path: float
    adaln_init_scale_factor: int
    activation: str
    position_encoding: str
    view_identity_encoding: str
    camera_order: tuple[str, str]
    camera_sources: tuple[str, str]
    proprio_dim: int
    temporal_attention: str
    control_conditioning: str
    control_normalization: str
    visual_normalization: str
    proprio_input_normalization: str
    future_proprio_output: str
    one_step_context: str
    rollout_context: str
    rollout_endpoint_sampling: str
    visual_loss_weight: float
    proprio_loss_weight: float
    public_latent_dtype: str
    internal_compute_dtype: str
    launch_support: JepaLaunchSupportContract

    def __post_init__(self) -> None:
        if self.temporal_contract.formal_tick_us != self.action_contract.formal_tick_us:
            raise ValueError("temporal and control formal clocks disagree")
        public_dimensions = (
            self.temporal_contract.history_sample_count,
            self.temporal_contract.maximum_delay_ticks,
            self.action_contract.action_dim,
            self.vision_encoder.patch_token_count,
            self.vision_encoder.feature_dim,
            self.proprio_dim,
            len(self.camera_order),
        )
        if public_dimensions != (6, 20, 7, 196, 384, 16, 2):
            raise ValueError("resolved config disagrees with the public tensor contract")
        if self.predictor_width != self.vision_encoder.feature_dim:
            raise ValueError("predictor width must preserve the frozen DINO feature width")
        if self.camera_order != ("agentview", "wrist"):
            raise ValueError("the current JEPA cache requires agentview,wrist order")
        expected_launch = (
            self.temporal_contract.history_sample_count - 1,
            self.temporal_contract.launch_trigger_horizon,
            self.temporal_contract.prediction_horizon,
            self.temporal_contract.maximum_delay_ticks,
        )
        observed_launch = (
            self.launch_support.minimum_cursor,
            self.launch_support.maximum_cursor,
            self.launch_support.prediction_horizon,
            self.launch_support.maximum_delay_ticks,
        )
        if observed_launch != expected_launch:
            raise ValueError("launch support disagrees with the resolved temporal contract")

    @property
    def formal_tick_us(self) -> int:
        return self.temporal_contract.formal_tick_us

    @property
    def history_ticks(self) -> int:
        return self.temporal_contract.history_sample_count

    @property
    def executed_control_ticks(self) -> int:
        return self.history_ticks - 1

    @property
    def maximum_delay_ticks(self) -> int:
        return self.temporal_contract.maximum_delay_ticks

    @property
    def prediction_horizon(self) -> int:
        return self.temporal_contract.prediction_horizon

    @property
    def action_dim(self) -> int:
        return self.action_contract.action_dim

    @property
    def patch_token_count(self) -> int:
        return self.vision_encoder.patch_token_count

    @property
    def visual_feature_dim(self) -> int:
        return self.vision_encoder.feature_dim

    @property
    def temporal_attention_window(self) -> int:
        return self.history_ticks


def _load_exact(path: Path, contract_type: type[Any], *, name: str) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {field.name for field in fields(contract_type)}
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{name} fields are invalid")
    return value


def _resolve(model_path: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute():
        raise ValueError("JEPA dependency paths must be relative to model.yaml")
    return (model_path.parent / path).resolve()


def load_action_conditioned_jepa_config(
    *,
    model_path: Path,
    level_path: Path,
) -> ActionConditionedJepaConfig:
    model_path = Path(model_path).resolve()
    level_path = Path(level_path).resolve()
    raw_model = _load_exact(model_path, _ModelSemantics, name="JEPA model config")
    raw_model["camera_order"] = tuple(raw_model["camera_order"])
    raw_model["camera_sources"] = tuple(raw_model["camera_sources"])
    model = _ModelSemantics(**raw_model)
    level = _LevelIdentity(**_load_exact(level_path, _LevelIdentity, name="JEPA level config"))
    temporal = load_temporal_contract(_resolve(model_path, model.temporal_config))
    action = load_action_contract(_resolve(model_path, model.control_config))
    vision = load_vision_encoder_spec(_resolve(model_path, model.vision_config))
    upstream = load_upstream_reference(_resolve(model_path, model.upstream_reference_config))
    identities = (
        (temporal.contract_id, model.temporal_contract_id),
        (action.contract_id, model.control_contract_id),
        (vision.encoder_id, model.vision_encoder_id),
        (upstream.reference_id, model.upstream_reference_id),
    )
    if any(actual != expected for actual, expected in identities):
        raise ValueError("JEPA dependency semantic identity is inconsistent")
    launch_support = JepaLaunchSupportContract(
        minimum_cursor=temporal.history_sample_count - 1,
        maximum_cursor=temporal.launch_trigger_horizon,
        prediction_horizon=temporal.prediction_horizon,
        maximum_delay_ticks=temporal.maximum_delay_ticks,
    )
    source_only = {
        "schema_version",
        "config_id",
        "temporal_config",
        "temporal_contract_id",
        "control_config",
        "control_contract_id",
        "vision_config",
        "vision_encoder_id",
        "upstream_reference_config",
        "upstream_reference_id",
    }
    semantics = {
        field.name: getattr(model, field.name)
        for field in fields(_ModelSemantics)
        if field.name not in source_only
    }
    return ActionConditionedJepaConfig(
        model_config_id=model.config_id,
        level_config_id=level.config_id,
        level=level.level,
        temporal_contract=temporal,
        action_contract=action,
        vision_encoder=vision,
        upstream_reference=upstream,
        launch_support=launch_support,
        **semantics,
    )
