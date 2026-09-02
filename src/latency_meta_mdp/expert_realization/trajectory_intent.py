"""Typed smooth-approach intent and canonical moving-ball grasp funnel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from latency_meta_mdp.expert_realization.contracts import (
    StrategyFamily,
    StrategyParameters,
    TaskInstanceId,
)
from latency_meta_mdp.expert_realization.shared_prefix import (
    SharedPrefixAnchor,
    _compare_shared_prefix_anchors,
)
from latency_meta_mdp.expert_realization.task_instance import (
    MaterializedTaskInstance,
    _readonly_exact,
    _thaw_json,
)
from latency_meta_mdp.motion import MotionSample, load_motion_config, motion_profile_from_mapping
from latency_meta_mdp.task import load_task_spec

_FORMAL_TICK_US = 20_000
_OBJECT_MOTION_ANCHOR_TICK = 150


def _profile(task_instance: MaterializedTaskInstance):
    config = load_motion_config(
        task_instance.project_root
        / f"configs/motion/dynamic_grasp_lift_l{task_instance.task_instance_id.level}.yaml"
    )
    return motion_profile_from_mapping(
        _thaw_json(task_instance.motion_profile_mapping),
        config=config,
    )


def _sample_profile_at_tick(
    task_instance: MaterializedTaskInstance, formal_tick: int
) -> MotionSample:
    if not isinstance(task_instance, MaterializedTaskInstance):
        raise TypeError("task_instance must be a MaterializedTaskInstance")
    if type(formal_tick) is not int or formal_tick < 0:
        raise ValueError("formal_tick must be a non-negative integer")
    task_instance.validate_publication_consistency()
    return _profile(task_instance).sample(formal_tick * _FORMAL_TICK_US)


def _chord_xy(task_instance: MaterializedTaskInstance) -> np.ndarray:
    mapping = _thaw_json(task_instance.motion_profile_mapping)
    profile_type = mapping["type"]
    if profile_type == "constant_velocity":
        start, end = mapping["start_xy"], mapping["end_xy"]
    elif profile_type == "cubic_polynomial":
        start, end = mapping["waypoint_positions_xy"][0], mapping["waypoint_positions_xy"][-1]
    elif profile_type == "piecewise_polynomial":
        start, end = mapping["segments"][0]["start_xy"], mapping["segments"][-1]["end_xy"]
    else:
        raise ValueError("unsupported moving-ball profile type")
    return np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)


def _unit_perpendicular_xy(
    task_instance: MaterializedTaskInstance, velocity_world: np.ndarray
) -> np.ndarray:
    direction = np.asarray(velocity_world[:2], dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-12:
        direction = _chord_xy(task_instance)
        norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-12:
        raise ValueError("object motion has no defined XY direction")
    tangent = direction / norm
    perpendicular = np.array([-tangent[1], tangent[0]], dtype=np.float64)
    perpendicular.setflags(write=False)
    return perpendicular


def _array(value: Any, *, shape: tuple[int | None, ...], name: str) -> np.ndarray:
    result = _readonly_exact(value, dtype=np.dtype(np.float64), shape=shape, name=name)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _proper_rotation(value: Any, *, name: str) -> np.ndarray:
    result = _array(value, shape=(3, 3), name=name)
    if not np.allclose(result.T @ result, np.eye(3), atol=1.0e-9, rtol=0.0) or not np.isclose(
        np.linalg.det(result), 1.0, atol=1.0e-9, rtol=0.0
    ):
        raise ValueError(f"{name} must be a proper rotation")
    return result


@dataclass(frozen=True)
class ApproachCurveIntent:
    """One complete approach curve; guide regions are soft constraints, not segments."""

    start_position_world: np.ndarray
    soft_guide_regions_world: np.ndarray
    soft_guide_radius_m: np.ndarray
    funnel_entry_target_tick: int
    funnel_entry_deadline_tick: int
    funnel_entry_position_world: np.ndarray
    funnel_entry_tangent_world: np.ndarray
    fixed_orientation_world: np.ndarray
    time_scaling_profile: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "start_position_world",
            _array(self.start_position_world, shape=(3,), name="start_position_world"),
        )
        guides = _array(
            self.soft_guide_regions_world,
            shape=(None, 3),
            name="soft_guide_regions_world",
        )
        if len(guides) > 2:
            raise ValueError("at most two soft approach guides are allowed")
        object.__setattr__(self, "soft_guide_regions_world", guides)
        radii = _array(
            self.soft_guide_radius_m,
            shape=(len(guides),),
            name="soft_guide_radius_m",
        )
        if np.any(radii <= 0.0):
            raise ValueError("soft guide radii must be positive")
        object.__setattr__(self, "soft_guide_radius_m", radii)
        if (
            type(self.funnel_entry_target_tick) is not int
            or type(self.funnel_entry_deadline_tick) is not int
            or not 5 < self.funnel_entry_target_tick <= self.funnel_entry_deadline_tick
        ):
            raise ValueError("funnel-entry timing is invalid")
        object.__setattr__(
            self,
            "funnel_entry_position_world",
            _array(
                self.funnel_entry_position_world,
                shape=(3,),
                name="funnel_entry_position_world",
            ),
        )
        tangent = _array(
            self.funnel_entry_tangent_world,
            shape=(3,),
            name="funnel_entry_tangent_world",
        )
        if not np.isclose(np.linalg.norm(tangent), 1.0, atol=1.0e-12, rtol=0.0):
            raise ValueError("funnel-entry tangent must be a unit vector")
        if abs(float(tangent[2])) > 1.0e-12:
            raise ValueError("funnel-entry tangent must be horizontal before centered descent")
        object.__setattr__(self, "funnel_entry_tangent_world", tangent)
        object.__setattr__(
            self,
            "fixed_orientation_world",
            _proper_rotation(self.fixed_orientation_world, name="fixed_orientation_world"),
        )
        if self.time_scaling_profile not in {
            "minimum_jerk_direct",
            "minimum_jerk_high_arc",
            "minimum_jerk_lateral_arc",
            "minimum_jerk_delayed",
        }:
            raise ValueError("unknown approach time-scaling profile")


@dataclass(frozen=True)
class CanonicalGraspFunnel:
    """Task-specific deterministic funnel shared by all approach families."""

    ball_relative_entry_offset_world: np.ndarray
    centered_pad_axis_world: np.ndarray
    fixed_orientation_world: np.ndarray
    eef_capture_offset_world: np.ndarray
    close_earliest_tick: int
    close_target_tick: int
    close_deadline_tick: int
    handoff_earliest_tick: int
    handoff_deadline_tick: int
    close_dwell_ticks: int
    bilateral_contact_acquisition_ticks: int
    centering_tolerance_m: float
    distance_tolerance_m: float
    relative_speed_tolerance_mps: float
    lift_relative_displacement_world: np.ndarray
    requires_physical_handoff: bool
    symmetric_close_command: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ball_relative_entry_offset_world",
            _array(
                self.ball_relative_entry_offset_world,
                shape=(3,),
                name="ball_relative_entry_offset_world",
            ),
        )
        pad_axis = _array(
            self.centered_pad_axis_world,
            shape=(3,),
            name="centered_pad_axis_world",
        )
        if not np.isclose(np.linalg.norm(pad_axis), 1.0, atol=1.0e-12, rtol=0.0):
            raise ValueError("centered pad axis must be a unit vector")
        object.__setattr__(self, "centered_pad_axis_world", pad_axis)
        object.__setattr__(
            self,
            "fixed_orientation_world",
            _proper_rotation(self.fixed_orientation_world, name="fixed_orientation_world"),
        )
        object.__setattr__(
            self,
            "eef_capture_offset_world",
            _array(
                self.eef_capture_offset_world,
                shape=(3,),
                name="eef_capture_offset_world",
            ),
        )
        ticks = (
            self.close_earliest_tick,
            self.close_target_tick,
            self.close_deadline_tick,
            self.handoff_earliest_tick,
            self.handoff_deadline_tick,
        )
        if any(type(value) is not int for value in ticks) or not (
            0
            <= ticks[0]
            <= ticks[1]
            <= ticks[2]
            <= ticks[3]
            < ticks[4]
            < _OBJECT_MOTION_ANCHOR_TICK
        ):
            raise ValueError("close and handoff windows are invalid")
        for name in ("close_dwell_ticks", "bilateral_contact_acquisition_ticks"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "centering_tolerance_m",
            "distance_tolerance_m",
            "relative_speed_tolerance_mps",
        ):
            value = getattr(self, name)
            if type(value) is not float or not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite float")
        object.__setattr__(
            self,
            "lift_relative_displacement_world",
            _array(
                self.lift_relative_displacement_world,
                shape=(3,),
                name="lift_relative_displacement_world",
            ),
        )
        if self.requires_physical_handoff is not True or self.symmetric_close_command is not True:
            raise ValueError("canonical funnel requires physical handoff and symmetric close")


@dataclass(frozen=True)
class PlannedMotionIntent:
    task_instance_id: TaskInstanceId
    strategy: StrategyParameters
    family: StrategyFamily
    capture_position_world: np.ndarray
    capture_velocity_world: np.ndarray
    lateral_direction_xy: np.ndarray
    approach: ApproachCurveIntent
    grasp_funnel: CanonicalGraspFunnel
    tracking_error_clip_m: float

    def __post_init__(self) -> None:
        if not isinstance(self.task_instance_id, TaskInstanceId):
            raise TypeError("task_instance_id must be a TaskInstanceId")
        if (
            not isinstance(self.strategy, StrategyParameters)
            or self.family is not self.strategy.family
        ):
            raise ValueError("intent family must match its strategy")
        for name in ("capture_position_world", "capture_velocity_world"):
            object.__setattr__(self, name, _array(getattr(self, name), shape=(3,), name=name))
        lateral = _array(self.lateral_direction_xy, shape=(2,), name="lateral_direction_xy")
        if not np.isclose(np.linalg.norm(lateral), 1.0, atol=1.0e-12, rtol=0.0):
            raise ValueError("lateral direction must be a unit vector")
        object.__setattr__(self, "lateral_direction_xy", lateral)
        if not isinstance(self.approach, ApproachCurveIntent):
            raise TypeError("approach must be an ApproachCurveIntent")
        if not isinstance(self.grasp_funnel, CanonicalGraspFunnel):
            raise TypeError("grasp_funnel must be a CanonicalGraspFunnel")
        if (
            self.approach.funnel_entry_deadline_tick
            + self.strategy.funnel_descent_ticks
            + self.grasp_funnel.close_dwell_ticks
            >= self.grasp_funnel.handoff_deadline_tick
        ):
            raise ValueError("latest funnel entry leaves insufficient grasp/handoff time")
        if type(self.tracking_error_clip_m) is not float or self.tracking_error_clip_m <= 0.0:
            raise ValueError("tracking_error_clip_m must be positive")


def _validate_workspace(
    task_instance: MaterializedTaskInstance, points: tuple[np.ndarray, ...]
) -> None:
    task = load_task_spec(task_instance.project_root / "configs/task/dynamic_grasp_lift_l0.yaml")
    x_limit = task.table_full_size[0] / 2.0 - task.ball_radius_m
    y_limit = task.table_full_size[1] / 2.0 - task.ball_radius_m
    z_min = task.table_offset[2]
    z_max = task.table_offset[2] + 0.60
    for index, point in enumerate(points):
        if (
            abs(float(point[0])) > x_limit
            or abs(float(point[1])) > y_limit
            or not z_min <= float(point[2]) <= z_max
        ):
            raise ValueError(f"trajectory intent point[{index}] lies outside the task workspace")


def build_trajectory_intent(
    task_instance: MaterializedTaskInstance,
    anchor: SharedPrefixAnchor,
    strategy: StrategyParameters,
) -> PlannedMotionIntent:
    """Construct one smooth-approach intent without creating executable actions."""
    if not isinstance(task_instance, MaterializedTaskInstance):
        raise TypeError("task_instance must be a MaterializedTaskInstance")
    if not isinstance(anchor, SharedPrefixAnchor):
        raise TypeError("anchor must be a SharedPrefixAnchor")
    if not isinstance(strategy, StrategyParameters):
        raise TypeError("strategy must be StrategyParameters")
    task_instance.validate_publication_consistency()
    if anchor.decision_source_tick != task_instance.decision_source_tick:
        raise ValueError("anchor does not match the task-instance decision tick")
    _compare_shared_prefix_anchors(task_instance.expected_anchor, anchor)

    capture = _sample_profile_at_tick(task_instance, strategy.close_target_tick)
    if capture.terminal:
        raise ValueError("capture target lies at or beyond terminal object motion")
    capture_position = np.asarray(capture.position, dtype=np.float64)
    capture_velocity = np.asarray(capture.velocity, dtype=np.float64)
    lateral_direction = _unit_perpendicular_xy(task_instance, capture_velocity)
    entry_offset = np.array([0.0, 0.0, strategy.funnel_entry_height_m], dtype=np.float64)
    entry_target_tick = strategy.close_target_tick - strategy.funnel_descent_ticks
    handoff_deadline_tick = min(
        strategy.close_target_tick + strategy.handoff_window_ticks,
        _OBJECT_MOTION_ANCHOR_TICK - 1,
    )
    latest_safe_entry_tick = (
        handoff_deadline_tick
        - strategy.funnel_descent_ticks
        - strategy.close_dwell_ticks
        - 1
    )
    entry_deadline_tick = min(
        entry_target_tick + strategy.funnel_entry_deadline_slack_ticks,
        latest_safe_entry_tick,
    )
    entry_sample = _sample_profile_at_tick(task_instance, entry_target_tick)
    if entry_sample.terminal:
        raise ValueError("funnel-entry target lies at or beyond terminal object motion")
    entry_object_position = np.asarray(entry_sample.position, dtype=np.float64)
    entry_object_velocity = np.asarray(entry_sample.velocity, dtype=np.float64)
    entry_position = entry_object_position + entry_offset
    midpoint = 0.5 * (np.asarray(anchor.anchor_eef_position_world) + entry_position)

    if strategy.family is StrategyFamily.CANONICAL_DIRECT:
        guides = np.empty((0, 3), dtype=np.float64)
        time_scaling = "minimum_jerk_direct"
    elif strategy.family is StrategyFamily.EARLY_HIGH_ARC:
        guide = np.array(midpoint, copy=True)
        assert strategy.high_arc_extra_height_m is not None
        guide[2] = (
            max(anchor.anchor_eef_position_world[2], entry_position[2])
            + strategy.high_arc_extra_height_m
        )
        guides = guide[None, :]
        time_scaling = "minimum_jerk_high_arc"
    elif strategy.family is StrategyFamily.LATERAL_ARC:
        guide = np.array(midpoint, copy=True)
        assert strategy.lateral_offset_m is not None
        assert strategy.lateral_direction_sign is not None
        guide[:2] += strategy.lateral_direction_sign * strategy.lateral_offset_m * lateral_direction
        guide[2] = max(anchor.anchor_eef_position_world[2], entry_position[2])
        guides = guide[None, :]
        time_scaling = "minimum_jerk_lateral_arc"
    else:
        guides = np.empty((0, 3), dtype=np.float64)
        time_scaling = "minimum_jerk_delayed"

    tangent = np.array(entry_object_velocity, copy=True)
    tangent[2] = 0.0
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1.0e-12:
        chord = _chord_xy(task_instance)
        tangent = np.array([chord[0], chord[1], 0.0], dtype=np.float64)
        tangent_norm = float(np.linalg.norm(tangent))
    tangent /= tangent_norm
    orientation = np.asarray(anchor.anchor_eef_orientation_matrix_world, dtype=np.float64)
    close_target = strategy.close_target_tick
    half_width = strategy.close_window_half_width_ticks
    lift_displacement = np.array(
        [0.0, 0.0, strategy.lift_vertical_displacement_m],
        dtype=np.float64,
    )
    funnel = CanonicalGraspFunnel(
        ball_relative_entry_offset_world=entry_offset,
        centered_pad_axis_world=orientation[:, 0],
        fixed_orientation_world=orientation,
        eef_capture_offset_world=np.array(
            [0.0, 0.0, strategy.grasp_eef_height_offset_m],
            dtype=np.float64,
        ),
        close_earliest_tick=close_target - half_width,
        close_target_tick=close_target,
        close_deadline_tick=close_target + half_width,
        handoff_earliest_tick=close_target + half_width,
        handoff_deadline_tick=handoff_deadline_tick,
        close_dwell_ticks=strategy.close_dwell_ticks,
        bilateral_contact_acquisition_ticks=strategy.bilateral_contact_acquisition_ticks,
        centering_tolerance_m=strategy.close_centering_tolerance_m,
        distance_tolerance_m=strategy.close_distance_tolerance_m,
        relative_speed_tolerance_mps=strategy.close_relative_speed_tolerance_mps,
        lift_relative_displacement_world=lift_displacement,
        requires_physical_handoff=True,
        symmetric_close_command=True,
    )
    approach = ApproachCurveIntent(
        start_position_world=np.asarray(anchor.anchor_eef_position_world, dtype=np.float64),
        soft_guide_regions_world=guides,
        soft_guide_radius_m=np.full(
            len(guides),
            strategy.soft_guide_radius_m,
            dtype=np.float64,
        ),
        funnel_entry_target_tick=entry_target_tick,
        funnel_entry_deadline_tick=entry_deadline_tick,
        funnel_entry_position_world=entry_position,
        funnel_entry_tangent_world=tangent,
        fixed_orientation_world=orientation,
        time_scaling_profile=time_scaling,
    )
    _validate_workspace(
        task_instance,
        (
            approach.start_position_world,
            *tuple(approach.soft_guide_regions_world),
            approach.funnel_entry_position_world,
            capture_position,
            capture_position + lift_displacement,
        ),
    )
    return PlannedMotionIntent(
        task_instance_id=task_instance.task_instance_id,
        strategy=strategy,
        family=strategy.family,
        capture_position_world=capture_position,
        capture_velocity_world=capture_velocity,
        lateral_direction_xy=lateral_direction,
        approach=approach,
        grasp_funnel=funnel,
        tracking_error_clip_m=strategy.tracking_error_clip_m,
    )
