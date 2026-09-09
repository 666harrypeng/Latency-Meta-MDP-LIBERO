"""Configuration for frozen-Encoder latency-law retention probes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class LawRetentionProbeConfig:
    schema_version: int
    probe_id: str
    training_context_limit: int
    validation_context_limit: int
    encoder_batch_size: int
    readout_batch_size: int
    readout_max_epochs: int
    readout_patience: int
    learning_rate: float
    weight_decay: float
    effective_mean_mae_ms_max: float
    baseline_relative_improvement_min: float
    ordering_accuracy_min: float
    random_seed: int

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.probe_id != "dinov3_flow_belief_law_retention_probe_v1"
        ):
            raise ValueError("unsupported law-retention probe config")
        integer_values = (
            self.training_context_limit,
            self.validation_context_limit,
            self.encoder_batch_size,
            self.readout_batch_size,
            self.readout_max_epochs,
            self.readout_patience,
            self.random_seed,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in integer_values
        ):
            raise ValueError("law-retention integer controls must be positive")
        if self.readout_patience > self.readout_max_epochs:
            raise ValueError("law-retention patience cannot exceed maximum epochs")
        numeric_values = (
            self.learning_rate,
            self.weight_decay,
            self.effective_mean_mae_ms_max,
            self.baseline_relative_improvement_min,
            self.ordering_accuracy_min,
        )
        if any(not np.isfinite(value) for value in numeric_values):
            raise ValueError("law-retention numeric controls must be finite")
        if (
            self.learning_rate <= 0.0
            or self.weight_decay < 0.0
            or self.effective_mean_mae_ms_max <= 0.0
            or not 0.0 < self.baseline_relative_improvement_min < 1.0
            or not 0.0 < self.ordering_accuracy_min <= 1.0
        ):
            raise ValueError("law-retention numeric controls are invalid")


def load_law_retention_probe_config(path: Path) -> LawRetentionProbeConfig:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = set(LawRetentionProbeConfig.__dataclass_fields__)
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("law-retention probe config fields are invalid")
    return LawRetentionProbeConfig(**value)
