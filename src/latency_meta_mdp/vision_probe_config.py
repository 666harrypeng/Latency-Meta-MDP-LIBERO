"""Validated configuration for the frozen-vision temporal state probe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class VisionProbeConfig:
    schema_version: int
    probe_id: str
    history_sample_count: int
    patch_projection_dim: int
    temporal_hidden_dim: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_epochs: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    random_seed: int

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.probe_id != "dinov3_temporal_state_probe_v1":
            raise ValueError("unsupported temporal state probe config")
        for name in (
            "history_sample_count",
            "patch_projection_dim",
            "temporal_hidden_dim",
            "batch_size",
            "max_epochs",
            "early_stopping_patience",
            "random_seed",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.history_sample_count != 6:
            raise ValueError("the current temporal state probe requires K=6")
        if not 0.0 < self.learning_rate < 1.0:
            raise ValueError("learning_rate must lie in (0, 1)")
        if not 0.0 <= self.weight_decay < 1.0:
            raise ValueError("weight_decay must lie in [0, 1)")
        if not 0.0 <= self.early_stopping_min_delta < 1.0:
            raise ValueError("early_stopping_min_delta must lie in [0, 1)")


def load_vision_probe_config(path: Path) -> VisionProbeConfig:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != set(VisionProbeConfig.__dataclass_fields__):
        raise ValueError("temporal state probe config fields are invalid")
    return VisionProbeConfig(**value)
