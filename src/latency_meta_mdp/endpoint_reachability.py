"""Stationary-ball grasp-and-lift endpoint reachability calibration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import yaml

from latency_meta_mdp.artifacts import collect_implementation_provenance, sha256_file


def _ordered_values(raw: Any, *, name: str) -> tuple[float, ...]:
    if not isinstance(raw, list) or len(raw) < 3:
        raise ValueError(f"{name} must contain at least three values")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw):
        raise TypeError(f"{name} must contain only numbers")
    values = tuple(float(value) for value in raw)
    if not all(np.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain only finite values")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError(f"{name} must be strictly increasing")
    return values


@dataclass(frozen=True)
class EndpointGridPoint:
    point_id: str
    x_index: int
    y_index: int
    x_m: float
    y_m: float


@dataclass(frozen=True)
class EndpointReachabilitySpec:
    schema_version: int
    calibration_id: str
    scene_seed: int
    x_values_m: tuple[float, ...]
    y_values_m: tuple[float, ...]
    interior_margin_cells: int
    camera_size_px: int
    handoff_deadline_us: int

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> EndpointReachabilitySpec:
        expected = {
            "schema_version",
            "calibration_id",
            "scene_seed",
            "x_values_m",
            "y_values_m",
            "interior_margin_cells",
            "camera_size_px",
            "handoff_deadline_us",
        }
        if set(raw) != expected:
            raise ValueError("endpoint reachability config fields are invalid")
        if raw["schema_version"] != 1:
            raise ValueError("endpoint reachability schema_version must be 1")
        if raw["calibration_id"] != "panda_endpoint_reachability_v1":
            raise ValueError("unsupported endpoint reachability calibration_id")
        for name in (
            "scene_seed",
            "interior_margin_cells",
            "camera_size_px",
            "handoff_deadline_us",
        ):
            value = raw[name]
            minimum = 0 if name in {"scene_seed", "interior_margin_cells"} else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                qualifier = "non-negative" if minimum == 0 else "positive"
                raise ValueError(f"{name} must be a {qualifier} integer")
        x_values = _ordered_values(raw["x_values_m"], name="x_values_m")
        y_values = _ordered_values(raw["y_values_m"], name="y_values_m")
        margin = raw["interior_margin_cells"]
        if 2 * margin >= min(len(x_values), len(y_values)):
            raise ValueError("interior_margin_cells leaves no candidate interior")
        if raw["handoff_deadline_us"] % 20_000:
            raise ValueError("handoff_deadline_us must lie on the formal grid")
        return cls(
            schema_version=1,
            calibration_id=raw["calibration_id"],
            scene_seed=raw["scene_seed"],
            x_values_m=x_values,
            y_values_m=y_values,
            interior_margin_cells=margin,
            camera_size_px=raw["camera_size_px"],
            handoff_deadline_us=raw["handoff_deadline_us"],
        )

    def grid_points(self) -> tuple[EndpointGridPoint, ...]:
        return tuple(
            EndpointGridPoint(
                point_id=f"x{x_index:02d}-y{y_index:02d}",
                x_index=x_index,
                y_index=y_index,
                x_m=x_m,
                y_m=y_m,
            )
            for y_index, y_m in enumerate(self.y_values_m)
            for x_index, x_m in enumerate(self.x_values_m)
        )


def load_endpoint_reachability_spec(path: Path) -> EndpointReachabilitySpec:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("endpoint reachability config must be a YAML mapping")
    return EndpointReachabilitySpec.from_mapping(raw)


@dataclass(frozen=True)
class EndpointReachabilityResult:
    point: EndpointGridPoint
    succeeded: bool
    terminal_status: str
    terminal_reason: str
    terminal_time_us: int
    first_contact_us: int | None
    stable_grasp_us: int | None
    handoff_us: int | None
    final_lift_height_m: float
    event_times_us: Mapping[str, int]

    def __post_init__(self) -> None:
        if not isinstance(self.point, EndpointGridPoint):
            raise TypeError("point must be an EndpointGridPoint")
        if not isinstance(self.succeeded, bool):
            raise TypeError("succeeded must be boolean")
        if self.terminal_status not in {"success", "failure", "timeout"}:
            raise ValueError("terminal_status is invalid")
        if not self.terminal_reason:
            raise ValueError("terminal_reason must be non-empty")
        if (
            isinstance(self.terminal_time_us, bool)
            or not isinstance(self.terminal_time_us, int)
            or self.terminal_time_us < 0
            or self.terminal_time_us % 20_000
        ):
            raise ValueError("terminal_time_us must lie on the formal grid")
        for name in ("first_contact_us", "stable_grasp_us", "handoff_us"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value % 2_000
            ):
                raise ValueError(f"{name} must lie on the physics grid")
        if not np.isfinite(self.final_lift_height_m):
            raise ValueError("final_lift_height_m must be finite")
        events = dict(self.event_times_us)
        if any(
            not isinstance(name, str)
            or not name
            or isinstance(time_us, bool)
            or not isinstance(time_us, int)
            or time_us < 0
            or time_us % 2_000
            for name, time_us in events.items()
        ):
            raise ValueError("event_times_us must map names to physics-grid times")
        object.__setattr__(self, "event_times_us", MappingProxyType(events))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "event_times_us": dict(self.event_times_us),
            "final_lift_height_m": self.final_lift_height_m,
            "first_contact_us": self.first_contact_us,
            "handoff_us": self.handoff_us,
            "point_id": self.point.point_id,
            "stable_grasp_us": self.stable_grasp_us,
            "succeeded": self.succeeded,
            "terminal_reason": self.terminal_reason,
            "terminal_status": self.terminal_status,
            "terminal_time_us": self.terminal_time_us,
            "x_index": self.point.x_index,
            "x_m": self.point.x_m,
            "y_index": self.point.y_index,
            "y_m": self.point.y_m,
        }


@dataclass(frozen=True)
class SafeRectangle:
    x_index_min: int
    x_index_max: int
    y_index_min: int
    y_index_max: int
    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float

    @property
    def cell_count(self) -> int:
        return (self.x_index_max - self.x_index_min + 1) * (
            self.y_index_max - self.y_index_min + 1
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "cell_count": self.cell_count,
            "x_index_max": self.x_index_max,
            "x_index_min": self.x_index_min,
            "x_max_m": self.x_max_m,
            "x_min_m": self.x_min_m,
            "y_index_max": self.y_index_max,
            "y_index_min": self.y_index_min,
            "y_max_m": self.y_max_m,
            "y_min_m": self.y_min_m,
        }


def _validated_result_grid(
    *,
    spec: EndpointReachabilitySpec,
    results: tuple[EndpointReachabilityResult, ...],
) -> np.ndarray:
    expected = {point.point_id: point for point in spec.grid_points()}
    if len(results) != len(expected) or len({result.point.point_id for result in results}) != len(
        expected
    ):
        raise ValueError("results must contain exactly one result for every grid point")
    actual = {result.point.point_id: result for result in results}
    if set(actual) != set(expected) or any(
        actual[point_id].point != point for point_id, point in expected.items()
    ):
        raise ValueError("results must contain exactly one result for every grid point")
    grid = np.zeros((len(spec.y_values_m), len(spec.x_values_m)), dtype=bool)
    for result in results:
        grid[result.point.y_index, result.point.x_index] = result.succeeded
    return grid


def derive_safe_rectangle(
    *,
    spec: EndpointReachabilitySpec,
    results: tuple[EndpointReachabilityResult, ...],
) -> SafeRectangle | None:
    """Find the largest all-success grid rectangle after an inward cell margin."""

    success = _validated_result_grid(spec=spec, results=results)
    for result in results:
        success[result.point.y_index, result.point.x_index] = bool(
            result.succeeded
            and result.handoff_us is not None
            and result.handoff_us <= spec.handoff_deadline_us
        )
    margin = spec.interior_margin_cells
    best: tuple[int, int, int, int] | None = None
    best_area = 0
    height, width = success.shape
    for y_min in range(height):
        for y_max in range(y_min, height):
            for x_min in range(width):
                for x_max in range(x_min, width):
                    if not np.all(success[y_min : y_max + 1, x_min : x_max + 1]):
                        continue
                    safe_x_min = x_min + margin
                    safe_x_max = x_max - margin
                    safe_y_min = y_min + margin
                    safe_y_max = y_max - margin
                    if safe_x_min > safe_x_max or safe_y_min > safe_y_max:
                        continue
                    area = (safe_x_max - safe_x_min + 1) * (safe_y_max - safe_y_min + 1)
                    if area > best_area:
                        best_area = area
                        best = (safe_x_min, safe_x_max, safe_y_min, safe_y_max)
    if best is None:
        return None
    x_min, x_max, y_min, y_max = best
    return SafeRectangle(
        x_index_min=x_min,
        x_index_max=x_max,
        y_index_min=y_min,
        y_index_max=y_max,
        x_min_m=spec.x_values_m[x_min],
        x_max_m=spec.x_values_m[x_max],
        y_min_m=spec.y_values_m[y_min],
        y_max_m=spec.y_values_m[y_max],
    )


def run_endpoint_attempt(
    *,
    project_root: Path,
    spec: EndpointReachabilitySpec,
    point: EndpointGridPoint,
) -> EndpointReachabilityResult:
    """Run one fresh stationary endpoint through the production task stack."""

    from latency_meta_mdp.backend import FormalStepExecutor, RoboSuitePlant
    from latency_meta_mdp.config import load_runtime_config
    from latency_meta_mdp.control import load_action_contract
    from latency_meta_mdp.expert import ExpertObservation, ScriptedBallExpert, load_expert_config
    from latency_meta_mdp.handoff import (
        HandoffAwareBallWorld,
        OneWayHandoff,
        PandaBallContactDetector,
    )
    from latency_meta_mdp.motion import DrivenBallWorld, StationaryProfile
    from latency_meta_mdp.outcomes import EpisodeOutcomeTracker, OutcomeCriteria, OutcomeStatus
    from latency_meta_mdp.snapshots import BoundarySnapshotter
    from latency_meta_mdp.task import load_task_spec, make_dynamic_grasp_lift_environment
    from latency_meta_mdp.timing import ClockLedger

    root = project_root.resolve()
    runtime = load_runtime_config(root / "configs/runtime/robosuite_v1.yaml")
    contract = load_action_contract(root / "configs/control/panda_osc_pose_delta_v1.yaml")
    expert_config = load_expert_config(root / "configs/expert/panda_ball_feedback_v1.yaml")
    task_spec = load_task_spec(root / "configs/task/dynamic_grasp_lift_l0.yaml")
    if (
        runtime.physics_dt_us != contract.physics_dt_us
        or runtime.formal_tick_us != contract.formal_tick_us
    ):
        raise ValueError("endpoint calibration configs do not share one formal clock")
    env = make_dynamic_grasp_lift_environment(
        spec=task_spec,
        seed=spec.scene_seed,
        offscreen=False,
        controller_config=contract.to_robosuite_config(),
    )
    try:
        contract.verify_runtime(env)
        tracker = EpisodeOutcomeTracker(
            OutcomeCriteria(
                physics_dt_us=runtime.physics_dt_us,
                formal_tick_us=runtime.formal_tick_us,
                stable_grasp_dwell_us=40_000,
                lift_height_m=task_spec.lift_success_height_m,
                lift_dwell_us=100_000,
                grasp_deadline_us=None,
                lift_timeout_us=10_000_000,
            )
        )
        handoff = OneWayHandoff(
            env=env,
            action_contract=contract,
            contact_detector=PandaBallContactDetector(env),
            outcome_tracker=tracker,
        )
        profile = StationaryProfile(
            position_xy=np.array([point.x_m, point.y_m], dtype=float),
            workspace_z=task_spec.ball_initial_position[2],
        )
        executor = FormalStepExecutor(
            plant=RoboSuitePlant(
                env=env,
                snapshotter=BoundarySnapshotter(
                    camera_names=(),
                    width=spec.camera_size_px,
                    height=spec.camera_size_px,
                ),
                world_writer=HandoffAwareBallWorld(
                    driver=DrivenBallWorld(profile=profile, motion_level=0),
                    handoff=handoff,
                ),
                physics_point_observer=handoff.on_physics_point,
                control_observer=handoff.on_control_applied,
            ),
            ledger=ClockLedger(
                physics_dt_us=runtime.physics_dt_us,
                formal_tick_us=runtime.formal_tick_us,
            ),
        )
        expert = ScriptedBallExpert(action_contract=contract, config=expert_config)
        snapshot = executor.initialize()
        max_steps = expert_config.collection_max_duration_us // runtime.formal_tick_us
        for _ in range(max_steps):
            arm = env.robots[0].part_controllers["right"]
            decision = expert.next_action(
                observation=ExpertObservation.from_snapshot(
                    snapshot,
                    world_to_base_rotation=arm.origin_ori.T,
                ),
                handoff_state=handoff.state,
            )
            snapshot = executor.step_formal(decision.action)
            if tracker.status is not OutcomeStatus.RUNNING:
                break

        qualified = bool(
            tracker.status is OutcomeStatus.SUCCESS
            and tracker.terminal_time_us is not None
            and tracker.terminal_time_us < expert_config.collection_max_duration_us
        )
        if tracker.status is OutcomeStatus.RUNNING:
            terminal_status = "timeout"
            terminal_reason = "calibration_timeout"
            terminal_time_us = expert_config.collection_max_duration_us
        elif qualified:
            terminal_status = "success"
            terminal_reason = tracker.terminal_reason.value
            terminal_time_us = tracker.terminal_time_us
        elif tracker.status is OutcomeStatus.SUCCESS:
            terminal_status = "failure"
            terminal_reason = "collection_deadline_exceeded"
            terminal_time_us = tracker.terminal_time_us
        else:
            terminal_status = "failure"
            if tracker.terminal_reason is None or tracker.terminal_time_us is None:
                raise RuntimeError("terminal calibration attempt is missing outcome metadata")
            terminal_reason = tracker.terminal_reason.value
            terminal_time_us = tracker.terminal_time_us
        if terminal_time_us is None:
            raise RuntimeError("calibration attempt is missing terminal time")
        return EndpointReachabilityResult(
            point=point,
            succeeded=qualified,
            terminal_status=terminal_status,
            terminal_reason=terminal_reason,
            terminal_time_us=terminal_time_us,
            first_contact_us=tracker.first_contact_us,
            stable_grasp_us=tracker.stable_grasp_us,
            handoff_us=tracker.handoff_us,
            final_lift_height_m=env.goal_measurements().lift_height_m,
            event_times_us={event.kind: event.time_us for event in tracker.events},
        )
    finally:
        env.close()


def build_endpoint_reachability_report(
    *,
    spec: EndpointReachabilitySpec,
    results: tuple[EndpointReachabilityResult, ...],
    implementation_revision: str,
    implementation_source_sha256: str,
    implementation_dirty: bool,
) -> dict[str, Any]:
    _validated_result_grid(spec=spec, results=results)
    if len(implementation_revision) != 40 or any(
        character not in "0123456789abcdef" for character in implementation_revision
    ):
        raise ValueError("implementation_revision must be a lowercase Git SHA-1")
    if len(implementation_source_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in implementation_source_sha256
    ):
        raise ValueError("implementation_source_sha256 must be a lowercase SHA-256")
    if not isinstance(implementation_dirty, bool):
        raise TypeError("implementation_dirty must be boolean")
    safe_rectangle = derive_safe_rectangle(spec=spec, results=results)
    by_id = {result.point.point_id: result for result in results}
    ordered_results = tuple(by_id[point.point_id] for point in spec.grid_points())
    successful_count = sum(result.succeeded for result in ordered_results)
    within_handoff_deadline_count = sum(
        result.succeeded
        and result.handoff_us is not None
        and result.handoff_us <= spec.handoff_deadline_us
        for result in ordered_results
    )
    blockers = [] if safe_rectangle is not None else ["no_all_success_interior_rectangle"]
    if implementation_dirty:
        blockers.append("implementation_dirty")
    return {
        "blockers": blockers,
        "calibration_id": spec.calibration_id,
        "camera_size_px": spec.camera_size_px,
        "eligible": not blockers,
        "failed_point_count": len(ordered_results) - successful_count,
        "handoff_deadline_us": spec.handoff_deadline_us,
        "implementation_revision": implementation_revision,
        "implementation_source_sha256": implementation_source_sha256,
        "implementation_dirty": implementation_dirty,
        "interior_margin_cells": spec.interior_margin_cells,
        "physics_dt_us": 2_000,
        "formal_tick_us": 20_000,
        "point_count": len(ordered_results),
        "results": [
            {
                **result.to_mapping(),
                "within_handoff_deadline": bool(
                    result.succeeded
                    and result.handoff_us is not None
                    and result.handoff_us <= spec.handoff_deadline_us
                ),
            }
            for result in ordered_results
        ],
        "safe_rectangle": None if safe_rectangle is None else safe_rectangle.to_mapping(),
        "scene_seed": spec.scene_seed,
        "schema_version": 1,
        "successful_point_count": successful_count,
        "within_handoff_deadline_point_count": within_handoff_deadline_count,
        "x_values_m": list(spec.x_values_m),
        "y_values_m": list(spec.y_values_m),
    }


def render_endpoint_reachability_svg(report: Mapping[str, Any]) -> str:
    """Render a dependency-free heatmap whose cells retain report point IDs."""

    x_values = tuple(float(value) for value in report["x_values_m"])
    y_values = tuple(float(value) for value in report["y_values_m"])
    results = report["results"]
    if len(results) != len(x_values) * len(y_values):
        raise ValueError("heatmap report does not contain a complete grid")
    cell = 42
    left = 92
    top = 58
    bottom = 88
    right = 36
    width = left + cell * len(x_values) + right
    height = top + cell * len(y_values) + bottom
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:DejaVu Sans,sans-serif;fill:#182230}",
        ".title{font-size:18px;font-weight:700}.axis{font-size:11px}",
        ".grid-cell{stroke:#ffffff;stroke-width:2}",
        ".success{fill:#3aa76d}.late{fill:#f59e0b}.failure{fill:#d94b4b}",
        ".safe-outline{fill:none;stroke:#172554;stroke-width:4;stroke-dasharray:7 4}",
        "</style>",
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<text class="title" x="20" y="28">Panda stationary endpoint reachability</text>',
    ]
    for result in results:
        x_index = int(result["x_index"])
        y_index = int(result["y_index"])
        x = left + x_index * cell
        y = top + (len(y_values) - 1 - y_index) * cell
        if result["within_handoff_deadline"]:
            status = "success"
        elif result["succeeded"]:
            status = "late"
        else:
            status = "failure"
        point_id = str(result["point_id"])
        lines.append(
            f'<rect class="grid-cell {status}" data-point-id="{point_id}" '
            f'x="{x}" y="{y}" width="{cell}" height="{cell}"/>'
        )
    safe = report["safe_rectangle"]
    if safe is not None:
        safe_x = left + int(safe["x_index_min"]) * cell
        safe_y = top + (len(y_values) - 1 - int(safe["y_index_max"])) * cell
        safe_width = (int(safe["x_index_max"]) - int(safe["x_index_min"]) + 1) * cell
        safe_height = (int(safe["y_index_max"]) - int(safe["y_index_min"]) + 1) * cell
        lines.append(
            f'<rect class="safe-outline" x="{safe_x}" y="{safe_y}" '
            f'width="{safe_width}" height="{safe_height}"/>'
        )
    for x_index, x_value in enumerate(x_values):
        x = left + x_index * cell + cell / 2
        lines.append(
            f'<text class="axis" x="{x:.1f}" y="{top + len(y_values) * cell + 20}" '
            f'text-anchor="middle">{x_value:.2f}</text>'
        )
    for y_index, y_value in enumerate(y_values):
        y = top + (len(y_values) - 1 - y_index) * cell + cell / 2 + 4
        lines.append(
            f'<text class="axis" x="{left - 10}" y="{y:.1f}" '
            f'text-anchor="end">{y_value:.2f}</text>'
        )
    lines.extend(
        [
            f'<text class="axis" x="{left + len(x_values) * cell / 2:.1f}" '
            f'y="{height - 28}" text-anchor="middle">world x (m)</text>',
            f'<text class="axis" transform="translate(22 {top + len(y_values) * cell / 2:.1f}) '
            'rotate(-90)" text-anchor="middle">world y (m)</text>',
            f'<rect x="20" y="{height - 20}" width="14" height="14" fill="#3aa76d"/>',
            f'<text class="axis" x="40" y="{height - 8}">success</text>',
            f'<rect x="105" y="{height - 20}" width="14" height="14" fill="#f59e0b"/>',
            f'<text class="axis" x="125" y="{height - 8}">late handoff</text>',
            f'<rect x="230" y="{height - 20}" width="14" height="14" fill="#d94b4b"/>',
            f'<text class="axis" x="250" y="{height - 8}">failed / timeout</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def run_endpoint_reachability_calibration(
    *,
    project_root: Path,
    spec: EndpointReachabilitySpec,
    on_result: Callable[[int, int, EndpointReachabilityResult], None] | None = None,
) -> dict[str, Any]:
    points = spec.grid_points()
    results: list[EndpointReachabilityResult] = []
    for index, point in enumerate(points, start=1):
        result = run_endpoint_attempt(project_root=project_root, spec=spec, point=point)
        results.append(result)
        if on_result is not None:
            on_result(index, len(points), result)
    root = project_root.resolve()
    provenance = collect_implementation_provenance(root)
    report = build_endpoint_reachability_report(
        spec=spec,
        results=tuple(results),
        implementation_revision=provenance.revision,
        implementation_source_sha256=provenance.source_sha256,
        implementation_dirty=provenance.dirty,
    )
    report["config_sha256"] = {
        "calibration": sha256_file(
            root / "configs/control/panda_endpoint_reachability_v1.yaml"
        ),
        "control": sha256_file(root / "configs/control/panda_osc_pose_delta_v1.yaml"),
        "expert": sha256_file(root / "configs/expert/panda_ball_feedback_v1.yaml"),
        "runtime": sha256_file(root / "configs/runtime/robosuite_v1.yaml"),
        "task": sha256_file(root / "configs/task/dynamic_grasp_lift_l0.yaml"),
    }
    return report
