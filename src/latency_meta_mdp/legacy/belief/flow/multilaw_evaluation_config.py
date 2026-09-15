"""Configuration for nominal, in-family, and shifted-law Flow evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import yaml


@dataclass(frozen=True)
class ShiftedLatencyLawSpec:
    mean_logit_offset: float
    log_concentration_offset: float
    uniform_floor: float

    def __post_init__(self) -> None:
        if (
            not -0.5 <= self.mean_logit_offset <= 0.5
            or not -0.5 <= self.log_concentration_offset <= 0.5
            or not 0.0 <= self.uniform_floor <= 0.1
        ):
            raise ValueError("shifted latency-law specification is invalid")


@dataclass(frozen=True)
class MultiLawFlowEvaluationConfig:
    schema_version: int
    evaluation_id: str
    shifted_laws: Mapping[str, ShiftedLatencyLawSpec]

    def __post_init__(self) -> None:
        laws = dict(self.shifted_laws)
        if (
            self.schema_version != 1
            or self.evaluation_id != "dinov3_flow_belief_multilaw_evaluation_v1"
            or tuple(laws) != ("fast", "slow", "wide")
            or any(not isinstance(value, ShiftedLatencyLawSpec) for value in laws.values())
        ):
            raise ValueError("multi-law Flow evaluation config is invalid")
        object.__setattr__(self, "shifted_laws", MappingProxyType(laws))


def load_multilaw_flow_evaluation_config(path: Path) -> MultiLawFlowEvaluationConfig:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "evaluation_id",
        "shifted_laws",
    }:
        raise ValueError("multi-law Flow evaluation config fields are invalid")
    raw_laws = value.pop("shifted_laws")
    if not isinstance(raw_laws, dict):
        raise ValueError("multi-law Flow shifted laws must be a mapping")
    laws = {
        name: ShiftedLatencyLawSpec(**row)
        for name, row in raw_laws.items()
        if isinstance(row, dict)
    }
    if len(laws) != len(raw_laws):
        raise ValueError("multi-law Flow shifted-law rows are invalid")
    return MultiLawFlowEvaluationConfig(**value, shifted_laws=laws)
