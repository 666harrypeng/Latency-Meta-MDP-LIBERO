"""Simulator-free moving-target keypose construction for structured expert planning."""

from __future__ import annotations

import math
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
_MIN_POST_CLOSE_HANDOFF_SLACK_TICKS = 20


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


@dataclass(frozen=True)
class InterceptionPlan:
    task_instance_id: TaskInstanceId
    strategy: StrategyParameters
    family: StrategyFamily
    interception_tick: int
    interception_time_us: int
    pregrasp_arrival_tick: int
    object_interception_position_world: np.ndarray
    object_interception_velocity_world: np.ndarray
    initial_eef_position_world: np.ndarray
    fixed_orientation_world: np.ndarray
    guide_positions_world: np.ndarray
    pregrasp_position_world: np.ndarray
    grasp_affordance_position_world: np.ndarray
    lift_target_world: np.ndarray
    lateral_direction_xy: np.ndarray
    close_dwell_ticks: int
    tracking_error_clip_m: float
    time_scaling_profile: str
    requires_physical_handoff: bool
    rotation_action_variation: bool
    iid_per_tick_action_noise: bool

    def __post_init__(self) -> None:
        if not isinstance(self.task_instance_id, TaskInstanceId):
            raise TypeError("task_instance_id must be a TaskInstanceId")
        if (
            not isinstance(self.strategy, StrategyParameters)
            or self.family is not self.strategy.family
        ):
            raise ValueError("plan family must match its validated strategy")
        if (
            type(self.interception_tick) is not int
            or self.interception_time_us != self.interception_tick * _FORMAL_TICK_US
            or type(self.pregrasp_arrival_tick) is not int
            or not 5 < self.pregrasp_arrival_tick < self.interception_tick
        ):
            raise ValueError("interception and pregrasp timing are invalid")
        array_contracts: dict[str, tuple[np.dtype[Any], tuple[int | None, ...]]] = {
            "object_interception_position_world": (np.dtype(np.float64), (3,)),
            "object_interception_velocity_world": (np.dtype(np.float64), (3,)),
            "initial_eef_position_world": (np.dtype(np.float64), (3,)),
            "fixed_orientation_world": (np.dtype(np.float64), (3, 3)),
            "guide_positions_world": (np.dtype(np.float64), (None, 3)),
            "pregrasp_position_world": (np.dtype(np.float64), (3,)),
            "grasp_affordance_position_world": (np.dtype(np.float64), (3,)),
            "lift_target_world": (np.dtype(np.float64), (3,)),
            "lateral_direction_xy": (np.dtype(np.float64), (2,)),
        }
        for name, (dtype, shape) in array_contracts.items():
            value = _readonly_exact(getattr(self, name), dtype=dtype, shape=shape, name=name)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        rotation = self.fixed_orientation_world
        rotation_is_orthogonal = np.allclose(
            rotation.T @ rotation, np.eye(3), atol=1.0e-9, rtol=0.0
        )
        determinant_is_one = np.isclose(
            np.linalg.det(rotation), 1.0, atol=1.0e-9, rtol=0.0
        )
        if not rotation_is_orthogonal or not determinant_is_one:
            raise ValueError("fixed_orientation_world must be a proper rotation")
        if not np.isclose(np.linalg.norm(self.lateral_direction_xy), 1.0, atol=1.0e-12):
            raise ValueError("lateral_direction_xy must be a unit vector")
        if self.close_dwell_ticks != self.strategy.close_dwell_ticks:
            raise ValueError("close dwell must match the validated strategy")
        if type(self.tracking_error_clip_m) is not float or (
            self.tracking_error_clip_m != self.strategy.tracking_error_clip_m
        ):
            raise ValueError("tracking clip must match the validated strategy")
        if self.requires_physical_handoff is not True:
            raise ValueError("lift must require physical handoff")
        if (
            self.rotation_action_variation is not False
            or self.iid_per_tick_action_noise is not False
        ):
            raise ValueError("pilot keyposes cannot introduce rotation or per-tick noise")


def _workspace_points(plan: InterceptionPlan) -> tuple[np.ndarray, ...]:
    return (
        plan.initial_eef_position_world,
        *tuple(plan.guide_positions_world),
        plan.pregrasp_position_world,
        plan.grasp_affordance_position_world,
        plan.lift_target_world,
    )


def _validate_workspace(task_instance: MaterializedTaskInstance, plan: InterceptionPlan) -> None:
    task = load_task_spec(task_instance.project_root / "configs/task/dynamic_grasp_lift_l0.yaml")
    x_limit = task.table_full_size[0] / 2.0 - task.ball_radius_m
    y_limit = task.table_full_size[1] / 2.0 - task.ball_radius_m
    z_min = task.table_offset[2]
    z_max = task.table_offset[2] + 0.60
    for index, point in enumerate(_workspace_points(plan)):
        if (
            abs(float(point[0])) > x_limit
            or abs(float(point[1])) > y_limit
            or not z_min <= float(point[2]) <= z_max
        ):
            raise ValueError(f"keypose[{index}] lies outside the task workspace")


def build_interception_keyposes(
    task_instance: MaterializedTaskInstance,
    anchor: SharedPrefixAnchor,
    strategy: StrategyParameters,
) -> InterceptionPlan:
    """Construct privileged expert keyposes without creating actions or planner outputs."""
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
    if (
        strategy.interception_tick
        + strategy.close_dwell_ticks
        + _MIN_POST_CLOSE_HANDOFF_SLACK_TICKS
        > _OBJECT_MOTION_ANCHOR_TICK
    ):
        raise ValueError("strategy leaves insufficient post-close handoff slack")

    sample = _sample_profile_at_tick(task_instance, strategy.interception_tick)
    if sample.terminal:
        raise ValueError("interception tick lies at or beyond terminal object motion")
    object_position = np.asarray(sample.position, dtype=np.float64)
    object_velocity = np.asarray(sample.velocity, dtype=np.float64)
    perpendicular = _unit_perpendicular_xy(task_instance, object_velocity)
    pregrasp = object_position + np.array([0.0, 0.0, strategy.pregrasp_height_m])
    approach_ticks = max(1, int(math.floor(strategy.interception_lead_seconds / 0.02 + 0.5)))
    arrival_tick = strategy.interception_tick - approach_ticks
    midpoint = 0.5 * (anchor.anchor_eef_position_world + pregrasp)

    if strategy.family is StrategyFamily.CANONICAL_DIRECT:
        guides = np.empty((0, 3), dtype=np.float64)
        time_scaling = "nominal"
    elif strategy.family is StrategyFamily.EARLY_HIGH_ARC:
        guide = np.array(midpoint, copy=True)
        guide[2] = max(anchor.anchor_eef_position_world[2], pregrasp[2]) + (
            0.5 * strategy.pregrasp_height_m
        )
        guides = guide[None, :]
        time_scaling = "early_smooth"
    elif strategy.family is StrategyFamily.LATERAL_ARC:
        guide = np.array(midpoint, copy=True)
        guide[:2] += (
            strategy.lateral_direction_sign * strategy.lateral_offset_m * perpendicular
        )
        guide[2] = max(anchor.anchor_eef_position_world[2], pregrasp[2])
        guides = guide[None, :]
        time_scaling = "lateral_smooth"
    else:
        guides = np.empty((0, 3), dtype=np.float64)
        time_scaling = "minimum_jerk_slow"

    lift = np.array(object_position, copy=True)
    lift[:2] += (
        strategy.lift_lateral_direction_sign
        * strategy.lift_lateral_offset_m
        * perpendicular
    )
    lift[2] += strategy.lift_vertical_offset_m
    plan = InterceptionPlan(
        task_instance_id=task_instance.task_instance_id,
        strategy=strategy,
        family=strategy.family,
        interception_tick=strategy.interception_tick,
        interception_time_us=strategy.interception_tick * _FORMAL_TICK_US,
        pregrasp_arrival_tick=arrival_tick,
        object_interception_position_world=object_position,
        object_interception_velocity_world=object_velocity,
        initial_eef_position_world=anchor.anchor_eef_position_world,
        fixed_orientation_world=anchor.anchor_eef_orientation_matrix_world,
        guide_positions_world=guides,
        pregrasp_position_world=pregrasp,
        grasp_affordance_position_world=object_position,
        lift_target_world=lift,
        lateral_direction_xy=perpendicular,
        close_dwell_ticks=strategy.close_dwell_ticks,
        tracking_error_clip_m=strategy.tracking_error_clip_m,
        time_scaling_profile=time_scaling,
        requires_physical_handoff=True,
        rotation_action_variation=strategy.rotation_action_variation,
        iid_per_tick_action_noise=strategy.iid_per_tick_action_noise,
    )
    _validate_workspace(task_instance, plan)
    return plan
