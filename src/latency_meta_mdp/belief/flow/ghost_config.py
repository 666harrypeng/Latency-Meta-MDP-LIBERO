"""Configuration for agent-view Flow Belief ghost rendering."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class FlowBeliefGhostConfig:
    schema_version: int
    renderer_id: str
    camera_name: str
    width: int
    height: int
    display_delay_ticks: tuple[int, ...]
    overlay_alpha: float
    overlap_alpha: float
    overlap_hatch_alpha: float
    overlap_hatch_spacing_px: int
    ground_truth_outline_width_px: int
    prediction_outline_width_px: int
    ground_truth_rgb: tuple[int, int, int]
    prediction_rgb: tuple[int, int, int]
    overlap_rgb: tuple[int, int, int]
    medoid_metric: str
    ground_truth_rgb_mae_max: float
    ball_centroid_error_px_max: float
    robot_mask_iou_min: float

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.renderer_id != "flow_belief_agentview_ghost_v1":
            raise ValueError("unsupported Flow Belief ghost config")
        if self.camera_name != "agentview":
            raise ValueError("Flow Belief ghost rendering requires agentview")
        if (self.width, self.height) != (256, 256):
            raise ValueError("Flow Belief ghost rendering requires 256 x 256 images")
        if self.display_delay_ticks != (1, 5, 10, 15, 20):
            raise ValueError("Flow Belief ghost display delays are invalid")
        if not math.isfinite(self.overlay_alpha) or not 0.0 < self.overlay_alpha <= 1.0:
            raise ValueError("Flow Belief ghost overlay alpha must lie in (0, 1]")
        if not math.isfinite(self.overlap_alpha) or not 0.0 < self.overlap_alpha < 0.5:
            raise ValueError("Flow Belief ghost overlap alpha must lie in (0, 0.5)")
        if not math.isfinite(self.overlap_hatch_alpha) or not 0.0 < self.overlap_hatch_alpha < 1.0:
            raise ValueError("Flow Belief ghost hatch alpha must lie in (0, 1)")
        if (
            isinstance(self.overlap_hatch_spacing_px, bool)
            or not isinstance(self.overlap_hatch_spacing_px, int)
            or self.overlap_hatch_spacing_px < 4
        ):
            raise ValueError("Flow Belief ghost hatch spacing must be at least four pixels")
        outline_widths = (
            self.ground_truth_outline_width_px,
            self.prediction_outline_width_px,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in outline_widths
        ):
            raise ValueError("Flow Belief ghost outline widths must be positive integers")
        if self.ground_truth_outline_width_px <= self.prediction_outline_width_px:
            raise ValueError("ground-truth outline must be wider than prediction outline")
        for name in ("ground_truth_rgb", "prediction_rgb", "overlap_rgb"):
            color = getattr(self, name)
            if len(color) != 3 or any(
                isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255
                for value in color
            ):
                raise ValueError(f"Flow Belief ghost {name} is invalid")
        if self.medoid_metric != "normalized_l2":
            raise ValueError("Flow Belief ghost medoid metric is invalid")
        thresholds = (
            self.ground_truth_rgb_mae_max,
            self.ball_centroid_error_px_max,
            self.robot_mask_iou_min,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in thresholds):
            raise ValueError("Flow Belief ghost parity thresholds must be positive")
        if self.robot_mask_iou_min > 1.0:
            raise ValueError("Flow Belief ghost robot mask IoU cannot exceed one")


def load_flow_belief_ghost_config(path: Path) -> FlowBeliefGhostConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != set(FlowBeliefGhostConfig.__dataclass_fields__):
        raise ValueError("Flow Belief ghost config fields are invalid")
    try:
        return FlowBeliefGhostConfig(
            **{
                **raw,
                "display_delay_ticks": tuple(raw["display_delay_ticks"]),
                "ground_truth_rgb": tuple(raw["ground_truth_rgb"]),
                "prediction_rgb": tuple(raw["prediction_rgb"]),
                "overlap_rgb": tuple(raw["overlap_rgb"]),
            }
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Flow Belief ghost config values are invalid") from exc
