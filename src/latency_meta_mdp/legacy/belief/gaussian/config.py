"""Configuration for the compact Gaussian return-belief baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class GaussianBeliefConfig:
    schema_version: int
    model_id: str
    history_sample_count: int
    belief_token_count: int
    model_dim: int
    patch_pool_query_count: int
    transformer_head_count: int
    fusion_layer_count: int
    dropout: float
    delay_fourier_frequency_count: int
    minimum_log_std: float
    maximum_log_std: float
    sampled_delay_query_count: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_epochs: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    random_seed: int

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.model_id != "dinov3_gaussian_belief_v1":
            raise ValueError("unsupported Gaussian belief config")
        for name in (
            "history_sample_count",
            "belief_token_count",
            "model_dim",
            "patch_pool_query_count",
            "transformer_head_count",
            "fusion_layer_count",
            "delay_fourier_frequency_count",
            "sampled_delay_query_count",
            "batch_size",
            "max_epochs",
            "early_stopping_patience",
            "random_seed",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.history_sample_count != 6 or self.belief_token_count != 4:
            raise ValueError("Gaussian belief baseline requires K6 and four belief tokens")
        if self.model_dim % self.transformer_head_count:
            raise ValueError("model_dim must be divisible by transformer_head_count")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if not self.minimum_log_std < self.maximum_log_std:
            raise ValueError("Gaussian log-standard-deviation bounds are invalid")
        if not 0.0 < self.learning_rate < 1.0:
            raise ValueError("learning_rate must lie in (0, 1)")
        if not 0.0 <= self.weight_decay < 1.0:
            raise ValueError("weight_decay must lie in [0, 1)")
        if not 0.0 <= self.early_stopping_min_delta < 1.0:
            raise ValueError("early_stopping_min_delta must lie in [0, 1)")


def load_gaussian_belief_config(path: Path) -> GaussianBeliefConfig:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != set(GaussianBeliefConfig.__dataclass_fields__):
        raise ValueError("Gaussian belief config fields are invalid")
    return GaussianBeliefConfig(**value)
