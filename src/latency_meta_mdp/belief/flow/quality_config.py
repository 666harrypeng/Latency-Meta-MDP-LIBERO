"""Canonical configuration for deterministic Flow Belief quality samples."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

from latency_meta_mdp.vision_probe_data import ProbeSplit

_DISPLAY_DELAYS = (1, 5, 10, 15, 20)
_CASE_QUANTILES = (0.5, 0.9, 0.95)
_SELECTION_ROLES = (
    "typical",
    "p90_hard",
    "p95_hard",
    "pre_handoff",
    "handoff_adjacent",
    "motion_transition",
    "short_delay_hard",
    "long_delay_hard",
)


@dataclass(frozen=True)
class FlowBeliefQualitySampleConfig:
    schema_version: int
    quality_id: str
    evaluation_split: ProbeSplit
    display_delay_ticks: tuple[int, ...]
    sample_count: int
    solver: str
    solver_step_count: int
    case_quantiles: tuple[float, ...]
    summary_parity_atol: float
    summary_parity_rtol: float
    selection_roles: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.quality_id != "flow_belief_quality_samples_v1":
            raise ValueError("unsupported Flow Belief quality-sample config")
        if self.evaluation_split is not ProbeSplit.VALIDATION:
            raise ValueError("quality-sample export requires the validation split")
        if self.display_delay_ticks != _DISPLAY_DELAYS:
            raise ValueError("display delay ticks must be the canonical unique tuple")
        if (
            isinstance(self.sample_count, bool)
            or self.sample_count != 32
            or self.solver != "heun"
            or isinstance(self.solver_step_count, bool)
            or self.solver_step_count != 16
        ):
            raise ValueError("quality-sample Flow protocol is invalid")
        if self.case_quantiles != _CASE_QUANTILES:
            raise ValueError("quality-sample quantiles are invalid")
        if (
            not math.isfinite(self.summary_parity_atol)
            or not math.isfinite(self.summary_parity_rtol)
            or self.summary_parity_atol <= 0.0
            or self.summary_parity_rtol <= 0.0
        ):
            raise ValueError("summary parity tolerances must be finite and positive")
        if self.selection_roles != _SELECTION_ROLES:
            raise ValueError("quality-sample selection roles are invalid")


def load_flow_belief_quality_sample_config(
    path: Path,
) -> FlowBeliefQualitySampleConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Flow Belief quality-sample config must be a mapping")
    expected = set(FlowBeliefQualitySampleConfig.__dataclass_fields__)
    if set(raw) != expected:
        raise ValueError("Flow Belief quality-sample config fields are invalid")
    try:
        evaluation_split = ProbeSplit(raw["evaluation_split"])
        display_delay_ticks = tuple(raw["display_delay_ticks"])
        case_quantiles = tuple(float(value) for value in raw["case_quantiles"])
        selection_roles = tuple(raw["selection_roles"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Flow Belief quality-sample config values are invalid") from exc
    return FlowBeliefQualitySampleConfig(
        **{
            **raw,
            "evaluation_split": evaluation_split,
            "display_delay_ticks": display_delay_ticks,
            "case_quantiles": case_quantiles,
            "selection_roles": selection_roles,
        }
    )
