"""Executable G1 certification for the 2 ms / 20 ms RoboSuite clock."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from latency_meta_mdp.backend import (
    FormalStepExecutor,
    RoboSuitePlant,
    instrument_stock_step,
    make_g1_environment,
)
from latency_meta_mdp.snapshots import BoundarySnapshot, BoundarySnapshotter
from latency_meta_mdp.timing import ClockLedger

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SHA256_LENGTH = 64


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _g1_source_sha256() -> str:
    paths = [
        _PROJECT_ROOT / "pyproject.toml",
        _PROJECT_ROOT / "uv.lock",
        _PROJECT_ROOT / "configs/runtime/robosuite_v1.yaml",
        _PROJECT_ROOT / "src/latency_meta_mdp/timing.py",
        _PROJECT_ROOT / "src/latency_meta_mdp/backend.py",
        _PROJECT_ROOT / "src/latency_meta_mdp/snapshots.py",
        _PROJECT_ROOT / "src/latency_meta_mdp/artifacts.py",
        _PROJECT_ROOT / "src/latency_meta_mdp/calibration.py",
        _PROJECT_ROOT / "src/latency_meta_mdp/cli/calibrate_timing.py",
    ]
    digest = hashlib.sha256()
    for path in paths:
        if not path.exists():
            continue
        relative = path.relative_to(_PROJECT_ROOT).as_posix().encode()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _hash_arrays(arrays: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


class BoundaryProbeCubeDriver:
    """Move the stock Lift cube analytically to expose boundary timing in images."""

    def __init__(
        self,
        env: Any,
        *,
        velocity_y_mps: float = 0.04,
        jump_time_us: int = 500_000,
        jump_y_m: float = 0.03,
    ) -> None:
        self.base_qpos = np.array(env.sim.data.get_joint_qpos(env.cube.joints[0]), copy=True)
        self.velocity_y_mps = velocity_y_mps
        self.jump_time_us = jump_time_us
        self.jump_y_m = jump_y_m

    def __call__(self, env: Any, current_time_us: int) -> dict[str, np.ndarray]:
        target = self.base_qpos.copy()
        target[1] += self.velocity_y_mps * current_time_us / 1_000_000
        if current_time_us >= self.jump_time_us:
            target[1] += self.jump_y_m
        env.sim.data.set_joint_qpos(env.cube.joints[0], target)
        env.sim.data.set_joint_qvel(env.cube.joints[0], np.zeros(6, dtype=float))
        return {"cube_target_qpos": target}


@dataclass(frozen=True)
class _LaneResult:
    summary: dict[str, Any]
    snapshots: tuple[BoundarySnapshot, ...]
    centroids: tuple[np.ndarray, ...]


def _marker_centroid(env: Any, snapshot: BoundarySnapshot) -> tuple[np.ndarray, int]:
    import mujoco

    segmentation = snapshot.cameras["agentview"].segmentation
    if segmentation is None:
        raise RuntimeError("agentview segmentation is required for G1 calibration")
    geom_ids = {
        env.sim.model.geom_name2id(name)
        for name in (*env.cube.visual_geoms, *env.cube.contact_geoms)
    }
    mask = (segmentation[:, :, 0] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(
        segmentation[:, :, 1], list(geom_ids)
    )
    rows, columns = np.nonzero(mask)
    if not len(columns):
        return np.array([math.nan, math.nan]), 0
    return np.array([float(columns.mean()), float(rows.mean())]), int(len(columns))


def _run_lane(*, seed: int, camera_width: int, camera_height: int) -> _LaneResult:
    env = make_g1_environment(seed=seed, offscreen=True)
    try:
        driver = BoundaryProbeCubeDriver(env)
        snapshotter = BoundarySnapshotter(
            camera_names=("agentview", "robot0_eye_in_hand"),
            segmentation_camera_names=("agentview",),
            width=camera_width,
            height=camera_height,
        )
        plant = RoboSuitePlant(env=env, snapshotter=snapshotter, world_writer=driver)
        ledger = ClockLedger(physics_dt_us=2_000, formal_tick_us=20_000)
        executor = FormalStepExecutor(plant=plant, ledger=ledger)
        snapshots = [executor.initialize()]
        action = np.zeros(env.action_dim, dtype=float)
        for _ in range(50):
            snapshots.append(executor.step_formal(action))

        centroid_and_count = [_marker_centroid(env, snapshot) for snapshot in snapshots]
        centroids = tuple(value[0] for value in centroid_and_count)
        pixel_counts = [value[1] for value in centroid_and_count]
        centroid_array = np.stack(centroids)
        marker_span = float(np.linalg.norm(centroid_array[-1] - centroid_array[0]))
        centroid_deltas = np.linalg.norm(np.diff(centroid_array, axis=0), axis=1)
        largest_jump_tick = int(np.argmax(centroid_deltas)) + 1
        expected_jump_tick = driver.jump_time_us // ledger.formal_tick_us
        expected_jump_centroid = float(centroid_deltas[expected_jump_tick - 1])
        robot_drift = max(
            float(np.max(np.abs(snapshot.robot_qpos - snapshots[0].robot_qpos)))
            for snapshot in snapshots
        )
        command_error = max(
            float(
                np.max(
                    np.abs(
                        snapshot.object_qpos - snapshot.commanded_world["cube_target_qpos"]
                    )
                )
            )
            for snapshot in snapshots
        )
        body_command_error = max(
            float(
                np.max(
                    np.abs(
                        snapshot.object_body_pos
                        - snapshot.commanded_world["cube_target_qpos"][:3]
                    )
                )
            )
            for snapshot in snapshots
        )
        source_mismatches = sum(
            sample.source_physics_step != snapshot.physics_step_index
            or sample.source_formal_tick != snapshot.formal_tick_index
            or sample.source_time_us != snapshot.time_us
            for snapshot in snapshots
            for sample in snapshot.cameras.values()
        )
        sim_time_error = max(
            abs(snapshot.sim_time_seconds - snapshot.time_us / 1_000_000)
            for snapshot in snapshots
        )
        state_arrays = [
            array
            for snapshot in snapshots
            for array in (
                snapshot.qpos,
                snapshot.qvel,
                snapshot.act,
                snapshot.actuator_ctrl,
                snapshot.robot_qpos,
                snapshot.robot_qvel,
                snapshot.eef_pos,
                snapshot.eef_xmat,
                snapshot.object_qpos,
                snapshot.object_body_pos,
                snapshot.object_body_quat_wxyz,
            )
        ]
        image_arrays = [
            snapshot.cameras[name].rgb
            for snapshot in snapshots
            for name in ("agentview", "robot0_eye_in_hand")
        ]
        summary = {
            "boundary_count": plant.boundary_count,
            "camera_names": ["agentview", "robot0_eye_in_hand"],
            "camera_source_mismatch_count": source_mismatches,
            "compatibility_tick_count": ledger.compatibility_tick_index,
            "control_refresh_count": plant.control_refresh_count,
            "final_sim_time_seconds": float(env.sim.data.time),
            "formal_tick_count": ledger.formal_tick_index,
            "goal_refresh_count": plant.goal_refresh_count,
            "marker_body_command_max_abs_error": body_command_error,
            "marker_centroid_span_pixels": marker_span,
            "marker_command_max_abs_error": command_error,
            "marker_expected_jump_centroid_pixels": expected_jump_centroid,
            "marker_largest_jump_tick": largest_jump_tick,
            "marker_min_visible_pixels": min(pixel_counts),
            "marker_missing_frame_count": sum(count == 0 for count in pixel_counts),
            "physics_step_count": ledger.physics_step_index,
            "rgb_trace_sha256": _hash_arrays(image_arrays),
            "robot_hold_max_abs_drift_rad": robot_drift,
            "sim_time_representation_max_abs_error_seconds": sim_time_error,
            "snapshot_count": len(snapshots),
            "state_trace_sha256": _hash_arrays(state_arrays),
            "step1_count": plant.step1_count,
            "step2_count": plant.step2_count,
            "world_write_count": plant.world_write_count,
        }
        return _LaneResult(summary=summary, snapshots=tuple(snapshots), centroids=centroids)
    finally:
        env.close()


def _snapshot_state_max_abs(left: BoundarySnapshot, right: BoundarySnapshot) -> float:
    values = []
    for left_value, right_value in (
        (left.qpos, right.qpos),
        (left.qvel, right.qvel),
        (left.act, right.act),
        (left.actuator_ctrl, right.actuator_ctrl),
        (left.robot_qpos, right.robot_qpos),
        (left.robot_qvel, right.robot_qvel),
        (left.eef_pos, right.eef_pos),
        (left.eef_xmat, right.eef_xmat),
        (left.object_qpos, right.object_qpos),
        (left.object_body_pos, right.object_body_pos),
        (left.object_body_quat_wxyz, right.object_body_quat_wxyz),
    ):
        values.append(float(np.max(np.abs(left_value - right_value), initial=0.0)))
    return max(values, default=0.0)


def _compare_lanes(left: _LaneResult, right: _LaneResult) -> dict[str, Any]:
    state_errors = [
        _snapshot_state_max_abs(a, b) for a, b in zip(left.snapshots, right.snapshots, strict=True)
    ]
    compatibility_indexes = range(0, len(left.snapshots), 5)
    shared_errors = [state_errors[index] for index in compatibility_indexes]
    centroid_errors = [
        float(np.linalg.norm(a - b)) for a, b in zip(left.centroids, right.centroids, strict=True)
    ]
    rgb_error = 0
    for left_snapshot, right_snapshot in zip(left.snapshots, right.snapshots, strict=True):
        for camera_name in ("agentview", "robot0_eye_in_hand"):
            left_rgb = left_snapshot.cameras[camera_name].rgb.astype(np.int16)
            right_rgb = right_snapshot.cameras[camera_name].rgb.astype(np.int16)
            rgb_error = max(rgb_error, int(np.max(np.abs(left_rgb - right_rgb), initial=0)))
    return {
        "rgb_max_abs_error": rgb_error,
        "segmentation_centroid_max_error_pixels": max(centroid_errors, default=0.0),
        "shared_100ms_state_max_abs_error": max(shared_errors, default=0.0),
        "state_max_abs_error": max(state_errors, default=0.0),
    }


def run_g1_calibration(
    *,
    seed: int,
    camera_width: int,
    camera_height: int,
    prerequisites: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Run stock instrumentation plus two independent one-second formal lanes."""
    import mujoco
    import robosuite

    stock_env = make_g1_environment(seed=seed, offscreen=False)
    try:
        integrator_value = int(stock_env.sim.model.opt.integrator)
        callback_present = mujoco.get_mjcb_control() is not None
        stock = instrument_stock_step(
            stock_env,
            np.zeros(stock_env.action_dim, dtype=float),
            observable_name="robot0_joint_pos",
        )
    finally:
        stock_env.close()

    lanes = [
        _run_lane(seed=seed, camera_width=camera_width, camera_height=camera_height)
        for _ in range(2)
    ]
    config = {
        "camera_height": camera_height,
        "camera_names": ["agentview", "robot0_eye_in_hand"],
        "camera_width": camera_width,
        "compatibility_stride_ticks": 5,
        "duration_us": 1_000_000,
        "formal_tick_us": 20_000,
        "marker_jump_time_us": 500_000,
        "marker_jump_y_m": 0.03,
        "marker_velocity_y_mps": 0.04,
        "physics_dt_us": 2_000,
        "seed": seed,
    }
    raw_report = {
        "blockers": [],
        "calibration_version": "g1_timing_v1",
        "config": config,
        "config_sha256": _json_sha256(config),
        "eligible": False,
        "external_control_callback_present": callback_present,
        "implementation_revision": _git_revision(),
        "implementation_source_sha256": _g1_source_sha256(),
        "integrator": (
            "Euler"
            if integrator_value == int(mujoco.mjtIntegrator.mjINT_EULER)
            else f"MuJoCoIntegrator({integrator_value})"
        ),
        "lanes": [lane.summary for lane in lanes],
        "mujoco_version": mujoco.__version__,
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "prerequisites": json.loads(json.dumps(prerequisites, sort_keys=True)),
        "repeat_comparison": _compare_lanes(lanes[0], lanes[1]),
        "robosuite_version": robosuite.__version__,
        "schema_version": 1,
        "stock_step": {
            "control_refresh_count": stock.control_refresh_count,
            "end_time_us": round(stock.end_time_seconds * 1_000_000),
            "goal_refresh_count": stock.goal_refresh_count,
            "latest_sample_time_us": round(stock.latest_sample_time_seconds * 1_000_000),
            "observable_update_count": stock.observable_update_count,
            "policy_step_flags": list(stock.policy_step_flags),
            "returned_observation_age_us": round(
                stock.returned_observation_age_seconds * 1_000_000
            ),
            "start_time_us": round(stock.start_time_seconds * 1_000_000),
            "step1_count": stock.step1_count,
            "step2_count": stock.step2_count,
        },
        "thresholds": {
            "marker_command_max_abs_error": 1e-12,
            "marker_expected_jump_centroid_min_pixels": 3.0,
            "marker_min_centroid_span_pixels": 1.0,
            "robot_hold_max_abs_drift_rad": 1e-3,
            "segmentation_centroid_max_error_pixels": 1.0,
            "state_max_abs_error": 1e-9,
        },
    }
    return validate_g1_report(raw_report)


def validate_g1_report(report: dict[str, Any]) -> dict[str, Any]:
    """Derive G1 eligibility from measured values rather than caller claims."""
    required = {
        "calibration_version",
        "config",
        "config_sha256",
        "external_control_callback_present",
        "implementation_revision",
        "implementation_source_sha256",
        "integrator",
        "lanes",
        "mujoco_version",
        "python_version",
        "prerequisites",
        "repeat_comparison",
        "robosuite_version",
        "schema_version",
        "stock_step",
        "thresholds",
    }
    missing = sorted(required - set(report))
    if missing:
        raise ValueError(f"missing G1 report fields: {missing}")
    blockers: list[str] = []
    config = report["config"]
    if report["schema_version"] != 1:
        blockers.append("wrong_schema_version")
    if report["calibration_version"] != "g1_timing_v1":
        blockers.append("wrong_calibration_version")
    if report["config_sha256"] != _json_sha256(config):
        blockers.append("config_hash_mismatch")
    expected_config = {
        "compatibility_stride_ticks": 5,
        "duration_us": 1_000_000,
        "formal_tick_us": 20_000,
        "physics_dt_us": 2_000,
    }
    for name, expected in expected_config.items():
        if config.get(name) != expected:
            blockers.append(f"config_{name}_mismatch")
    if config.get("camera_names") != ["agentview", "robot0_eye_in_hand"]:
        blockers.append("camera_names_mismatch")
    if config.get("marker_jump_time_us") != 500_000 or config.get("marker_jump_y_m") != 0.03:
        blockers.append("marker_jump_config_mismatch")
    if report["python_version"].split(".")[:2] != ["3", "10"]:
        blockers.append("python_version_mismatch")
    if report["robosuite_version"] != "1.5.2":
        blockers.append("robosuite_version_mismatch")
    if report["mujoco_version"] != "3.3.3":
        blockers.append("mujoco_version_mismatch")
    if report["integrator"] != "Euler":
        blockers.append("integrator_mismatch")
    if report["external_control_callback_present"] is not False:
        blockers.append("external_control_callback_present")
    if report["implementation_revision"] != _git_revision():
        blockers.append("implementation_revision_mismatch")
    if report["implementation_source_sha256"] != _g1_source_sha256():
        blockers.append("implementation_source_hash_mismatch")
    for name in ("config_sha256", "implementation_source_sha256"):
        value = report[name]
        if not isinstance(value, str) or len(value) != _SHA256_LENGTH:
            blockers.append(f"{name}_malformed")
    prerequisites = report["prerequisites"]
    if not isinstance(prerequisites, dict) or set(prerequisites) != {"g0"}:
        blockers.append("g0_prerequisite_missing")
    else:
        reference = prerequisites["g0"]
        if not isinstance(reference, dict) or set(reference) != {"manifest", "sha256"}:
            blockers.append("g0_prerequisite_malformed")
        elif (
            not isinstance(reference["manifest"], str)
            or not isinstance(reference["sha256"], str)
            or len(reference["sha256"]) != _SHA256_LENGTH
        ):
            blockers.append("g0_prerequisite_malformed")

    stock = report["stock_step"]
    for name, expected in {
        "control_refresh_count": 10,
        "end_time_us": 20_000,
        "goal_refresh_count": 1,
        "observable_update_count": 10,
        "start_time_us": 0,
        "step1_count": 10,
        "step2_count": 10,
    }.items():
        if stock.get(name) != expected:
            blockers.append(f"stock_{name}_mismatch")
    if stock.get("policy_step_flags") != [True] + [False] * 9:
        blockers.append("stock_policy_step_flags_mismatch")
    age_us = stock.get("returned_observation_age_us")
    if not isinstance(age_us, int) or not 0 <= age_us <= 20_000:
        blockers.append("stock_observation_age_invalid")

    lanes = report["lanes"]
    if not isinstance(lanes, list) or len(lanes) != 2:
        blockers.append("lane_count_mismatch")
        lanes = []
    thresholds = report["thresholds"]
    for index, lane in enumerate(lanes):
        exact = {
            "boundary_count": 51,
            "camera_source_mismatch_count": 0,
            "compatibility_tick_count": 10,
            "control_refresh_count": 500,
            "formal_tick_count": 50,
            "goal_refresh_count": 50,
            "marker_missing_frame_count": 0,
            "physics_step_count": 500,
            "snapshot_count": 51,
            "step1_count": 501,
            "step2_count": 500,
            "world_write_count": 501,
        }
        for name, expected in exact.items():
            if lane.get(name) != expected:
                blockers.append(f"lane_{index}_{name}_mismatch")
        if lane.get("camera_names") != ["agentview", "robot0_eye_in_hand"]:
            blockers.append(f"lane_{index}_camera_names_mismatch")
        numeric_limits = {
            "marker_body_command_max_abs_error": thresholds["marker_command_max_abs_error"],
            "marker_command_max_abs_error": thresholds["marker_command_max_abs_error"],
            "robot_hold_max_abs_drift_rad": thresholds["robot_hold_max_abs_drift_rad"],
            "sim_time_representation_max_abs_error_seconds": 1e-12,
        }
        for name, limit in numeric_limits.items():
            value = lane.get(name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value > limit:
                blockers.append(f"lane_{index}_{name}_exceeded")
        span = lane.get("marker_centroid_span_pixels")
        if (
            not isinstance(span, (int, float))
            or not math.isfinite(span)
            or span < thresholds["marker_min_centroid_span_pixels"]
        ):
            blockers.append(f"lane_{index}_marker_centroid_span_too_small")
        if lane.get("marker_largest_jump_tick") != 25:
            blockers.append(f"lane_{index}_marker_jump_tick_mismatch")
        jump_pixels = lane.get("marker_expected_jump_centroid_pixels")
        if (
            not isinstance(jump_pixels, (int, float))
            or not math.isfinite(jump_pixels)
            or jump_pixels < thresholds["marker_expected_jump_centroid_min_pixels"]
        ):
            blockers.append(f"lane_{index}_marker_jump_not_visible_at_expected_boundary")
        if lane.get("marker_min_visible_pixels", 0) <= 0:
            blockers.append(f"lane_{index}_marker_not_visible")

    comparison = report["repeat_comparison"]
    for name, limit in {
        "segmentation_centroid_max_error_pixels": thresholds[
            "segmentation_centroid_max_error_pixels"
        ],
        "shared_100ms_state_max_abs_error": thresholds["state_max_abs_error"],
        "state_max_abs_error": thresholds["state_max_abs_error"],
    }.items():
        value = comparison.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value > limit:
            blockers.append(f"repeat_{name}_exceeded")
    if comparison.get("rgb_max_abs_error") != 0:
        blockers.append("repeat_rgb_mismatch")

    validated = {key: value for key, value in report.items() if key not in {"eligible", "blockers"}}
    validated["blockers"] = sorted(set(blockers))
    validated["eligible"] = not validated["blockers"]
    return validated
