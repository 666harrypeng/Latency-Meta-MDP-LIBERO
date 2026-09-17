"""Small surface-conveyor configuration with inherited Panda and camera resources."""

from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np
import yaml

from latency_meta_mdp.envs.task import TaskSpec, load_task_spec
from latency_meta_mdp.io.paths import repository_root


@dataclass(frozen=True)
class ConveyorSpec:
    scene: TaskSpec
    action_contract_path: str
    instruction: str
    belt_center_xy: tuple
    belt_size_xy: tuple
    upstream_extension_m: float
    agentview_retreat_m: float
    agentview_upstream_shift_m: float
    belt_top_z: float
    belt_speed_mps: float
    drive_time_constant_s: float
    max_drive_accel_mps2: float
    spacing_speed_mps: float
    spawn_x: tuple
    spawn_y: tuple
    clearance_m: float
    pair_max_separation_m: float
    first_spawn_tick: int
    supply_ticks: int
    spawn_count: int | None
    drain_ticks: int
    intervals: dict
    interval_probabilities: tuple
    colors: tuple
    goal_center: tuple
    goal_radius_m: float
    goal_rgba: tuple
    minimum_lift_m: float
    grasp_ticks: int
    release_width_m: float
    release_opening_delta_m: float
    miss_height_z: float

    @property
    def table_full_size(self):
        x, y, z = self.scene.table_full_size
        return x, y + self.upstream_extension_m, z

    @property
    def table_offset(self):
        x, y, z = self.scene.table_offset
        return x, y - self.upstream_extension_m / 2, z

    @property
    def transport_center_xy(self):
        x, y = self.belt_center_xy
        return x, y - self.upstream_extension_m / 2

    @property
    def transport_size_xy(self):
        x, y = self.belt_size_xy
        return x, y + self.upstream_extension_m

    @property
    def agentview_position(self):
        # Translate along camera-local +Z and upstream; keep orientation and FOV.
        w, x, y, z = self.scene.agentview_quaternion_wxyz
        back = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
        pos = np.asarray(self.scene.agentview_position) + self.agentview_retreat_m * back
        pos[1] -= self.agentview_upstream_shift_m
        return tuple(float(v) for v in pos)

    def validate(self):
        scalars = (
            self.belt_speed_mps,
            self.drive_time_constant_s,
            self.max_drive_accel_mps2,
            self.spacing_speed_mps,
            self.clearance_m,
            self.pair_max_separation_m,
            self.goal_radius_m,
            self.minimum_lift_m,
            self.release_width_m,
            self.release_opening_delta_m,
        )
        if not np.isfinite(scalars).all() or min(scalars) <= 0:
            raise ValueError("conveyor speeds, forces and geometric tolerances must be positive")
        if self.spacing_speed_mps > self.belt_speed_mps:
            raise ValueError("spacing screen must use a conservative transport speed")
        shifts = (
            self.upstream_extension_m,
            self.agentview_retreat_m,
            self.agentview_upstream_shift_m,
        )
        if not np.isfinite(shifts).all() or min(shifts) < 0:
            raise ValueError("upstream extension and camera translations must be nonnegative")
        for value in (self.first_spawn_tick, self.supply_ticks, self.drain_ticks, self.grasp_ticks):
            if type(value) is not int or value < 1:
                raise ValueError("conveyor timing uses positive integer formal ticks")
        if self.first_spawn_tick >= self.supply_ticks:
            raise ValueError("supply window must contain arrivals")
        if self.spawn_count is not None and (
            type(self.spawn_count) is not int or self.spawn_count < 1
        ):
            raise ValueError("spawn_count must be a positive integer or null")
        if set(self.intervals) != {"moderate", "sparse", "burst"} or any(
            len(v) != 2 or any(type(t) is not int or t < 1 for t in v) or v[0] > v[1]
            for v in self.intervals.values()
        ):
            raise ValueError("invalid arrival intervals")
        probabilities = np.asarray(self.interval_probabilities)
        if (
            probabilities.shape != (3,)
            or not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or not np.isclose(probabilities.sum(), 1)
        ):
            raise ValueError("arrival probabilities must sum to one")
        radius = self.scene.ball_radius_m
        if self.pair_max_separation_m <= 2 * radius + self.clearance_m:
            raise ValueError("pair spacing must permit non-overlapping parcels")
        if (
            len(self.belt_center_xy) != 2
            or len(self.belt_size_xy) != 2
            or min(self.belt_size_xy) <= 2 * radius
        ):
            raise ValueError("invalid belt geometry")
        for dim, bounds in enumerate((self.spawn_x, self.spawn_y)):
            c, size = self.transport_center_xy[dim], self.transport_size_xy[dim]
            if (
                len(bounds) != 2
                or bounds[0] >= bounds[1]
                or not c - size / 2 + radius <= bounds[0] < bounds[1] <= c + size / 2 - radius
            ):
                raise ValueError("spawn region must fit on the belt")
        if (
            len(self.goal_center) != 3
            or self.goal_center[2] - self.goal_radius_m <= self.belt_top_z
            or self.miss_height_z >= self.scene.table_offset[2]
        ):
            raise ValueError("goal must clear the surface; miss plane must be below the table")
        if not self.colors or any(
            len(c) != 4 or not all(0 <= v <= 1 for v in c) or c[-1] != 1 for c in self.colors
        ):
            raise ValueError("parcel colors must be opaque RGBA")
        if len(self.goal_rgba) != 4 or not all(0 <= v <= 1 for v in self.goal_rgba):
            raise ValueError("goal color must be RGBA")


def load_conveyor_spec(path: Path) -> ConveyorSpec:
    raw = yaml.safe_load(Path(path).read_text())
    if (
        raw.pop("schema_version") != 1
        or raw.pop("task_id") != "conveyor_sort"
        or raw.pop("variant") != "surface"
    ):
        raise ValueError("only conveyor_sort surface v1 is implemented")
    scene = load_task_spec(repository_root() / raw.pop("reference_scene"))
    raw.setdefault("spawn_count", None)
    tuple_fields = {
        "belt_center_xy",
        "belt_size_xy",
        "spawn_x",
        "spawn_y",
        "goal_center",
        "goal_rgba",
        "interval_probabilities",
    }
    if set(raw) != {f.name for f in fields(ConveyorSpec)} - {"scene"}:
        raise ValueError("conveyor config fields differ from schema")
    raw = {k: tuple(v) if k in tuple_fields else v for k, v in raw.items()}
    raw["colors"] = tuple(tuple(c) for c in raw["colors"])
    spec = ConveyorSpec(scene=scene, **raw)
    spec.validate()
    return spec


def apply_view_profile(spec: ConveyorSpec, path: Path) -> ConveyorSpec:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or set(raw) != {
        "agentview_retreat_m",
        "agentview_upstream_shift_m",
    }:
        raise ValueError("view profiles may only change main-camera translation")
    result = replace(spec, **raw)
    result.validate()
    return result
