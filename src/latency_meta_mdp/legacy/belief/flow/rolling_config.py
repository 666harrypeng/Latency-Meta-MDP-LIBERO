"""Strict configuration for seed-locked Flow Belief rolling inspection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class FlowBeliefRollingConfig:
    schema_version: int
    report_id: str
    evaluation_split: str
    source_phases: tuple[str, ...]
    history_sample_count: int
    inspection_stride_ticks: int
    display_delay_ticks: tuple[int, ...]
    sample_count: int
    solver: str
    solver_step_count: int
    include_approach_anchor: bool
    include_last_full_window_anchor: bool
    video_fps: int

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.report_id != "flow_belief_rolling_inspection_v1":
            raise ValueError("unsupported Flow Belief rolling inspection config")
        if self.evaluation_split != "validation":
            raise ValueError("rolling inspection requires the validation split")
        if self.source_phases != ("pregrasp", "approach"):
            raise ValueError("rolling inspection source phases are invalid")
        if self.history_sample_count != 6:
            raise ValueError("rolling inspection history contract is invalid")
        if (
            isinstance(self.inspection_stride_ticks, bool)
            or not isinstance(self.inspection_stride_ticks, int)
            or self.inspection_stride_ticks <= 0
        ):
            raise ValueError("rolling inspection stride must be a positive integer")
        if self.display_delay_ticks != (1, 5, 10, 15, 20):
            raise ValueError("rolling inspection display delays are invalid")
        if self.sample_count != 32:
            raise ValueError("rolling inspection sample count is invalid")
        if self.solver != "heun":
            raise ValueError("rolling inspection solver is invalid")
        if (
            isinstance(self.solver_step_count, bool)
            or not isinstance(self.solver_step_count, int)
            or self.solver_step_count <= 0
        ):
            raise ValueError("rolling inspection solver-step count is invalid")
        if self.include_approach_anchor is not True:
            raise ValueError("rolling inspection requires the approach anchor")
        if self.include_last_full_window_anchor is not True:
            raise ValueError("rolling inspection requires the final full-window anchor")
        if (
            isinstance(self.video_fps, bool)
            or not isinstance(self.video_fps, int)
            or self.video_fps <= 0
        ):
            raise ValueError("rolling inspection video fps must be positive")


def load_flow_belief_rolling_config(path: Path) -> FlowBeliefRollingConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != set(FlowBeliefRollingConfig.__dataclass_fields__):
        raise ValueError("Flow Belief rolling config fields are invalid")
    try:
        return FlowBeliefRollingConfig(
            **{
                **raw,
                "source_phases": tuple(raw["source_phases"]),
                "display_delay_ticks": tuple(raw["display_delay_ticks"]),
            }
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Flow Belief rolling config values are invalid") from exc
