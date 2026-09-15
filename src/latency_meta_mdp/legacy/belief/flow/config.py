"""Validated configuration for the Flow Matching Belief mainline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class FlowBeliefConfig:
    schema_version: int
    model_id: str
    history_sample_count: int
    belief_token_count: int
    model_dim: int
    patch_pool_query_count: int
    transformer_head_count: int
    fusion_layer_count: int
    dropout: float
    flow_hidden_dim: int
    flow_residual_block_count: int
    flow_time_fourier_frequency_count: int
    delay_fourier_frequency_count: int
    sampled_delay_query_count: int
    validation_flow_draw_count: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_epochs: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    gradient_clip_norm: float
    solver: str
    solver_step_count: int
    evaluation_sample_count: int
    random_seed: int
    evaluation_seed: int

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.model_id != "dinov3_flow_belief_v1":
            raise ValueError("unsupported Flow Belief config")
        for name in (
            "history_sample_count",
            "belief_token_count",
            "model_dim",
            "patch_pool_query_count",
            "transformer_head_count",
            "fusion_layer_count",
            "flow_hidden_dim",
            "flow_residual_block_count",
            "flow_time_fourier_frequency_count",
            "delay_fourier_frequency_count",
            "sampled_delay_query_count",
            "validation_flow_draw_count",
            "batch_size",
            "max_epochs",
            "early_stopping_patience",
            "solver_step_count",
            "evaluation_sample_count",
            "random_seed",
            "evaluation_seed",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.history_sample_count != 6 or self.belief_token_count != 8:
            raise ValueError("Flow Belief mainline requires K6 and eight belief tokens")
        if self.model_dim % self.transformer_head_count:
            raise ValueError("model_dim must be divisible by transformer_head_count")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if not 0.0 < self.learning_rate < 1.0:
            raise ValueError("learning_rate must lie in (0, 1)")
        if not 0.0 <= self.weight_decay < 1.0:
            raise ValueError("weight_decay must lie in [0, 1)")
        if not 0.0 <= self.early_stopping_min_delta < 1.0:
            raise ValueError("early_stopping_min_delta must lie in [0, 1)")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        if self.solver not in {"euler", "heun"}:
            raise ValueError("Flow Belief solver must be euler or heun")


def load_flow_belief_config(path: Path) -> FlowBeliefConfig:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != set(FlowBeliefConfig.__dataclass_fields__):
        raise ValueError("Flow Belief config fields are invalid")
    return FlowBeliefConfig(**value)
