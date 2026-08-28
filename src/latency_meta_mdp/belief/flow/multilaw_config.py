"""Configuration for multi-law Flow Belief training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_SUPPORTED_TRAINING_FAMILIES = {
    "dinov3_flow_belief_multilaw_v2": "truncated_beta_family_5_26_400ms_v1",
    "dinov3_flow_belief_multilaw_v3": "truncated_beta_family_8_65_400ms_v1",
}


@dataclass(frozen=True)
class MultiLawFlowTrainingConfig:
    schema_version: int
    training_id: str
    latency_law_family_id: str
    tail_query_uniform_mix: float
    early_stopping_patience: int

    def __post_init__(self) -> None:
        expected_family = _SUPPORTED_TRAINING_FAMILIES.get(self.training_id)
        if (
            self.schema_version != 1
            or expected_family is None
            or self.latency_law_family_id != expected_family
            or not 0.0 < self.tail_query_uniform_mix < 0.5
            or isinstance(self.early_stopping_patience, bool)
            or not isinstance(self.early_stopping_patience, int)
            or self.early_stopping_patience <= 0
        ):
            raise ValueError("multi-law Flow training config is invalid")


def load_multilaw_flow_training_config(path: Path) -> MultiLawFlowTrainingConfig:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = set(MultiLawFlowTrainingConfig.__dataclass_fields__)
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("multi-law Flow training config fields are invalid")
    return MultiLawFlowTrainingConfig(**value)
