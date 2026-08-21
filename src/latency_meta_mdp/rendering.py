"""Deterministic task-camera rendering and evidence publication."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image

from latency_meta_mdp.snapshots import BoundarySnapshotter
from latency_meta_mdp.task import load_task_spec, make_dynamic_grasp_lift_environment
from latency_meta_mdp.timing import ClockLedger


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _ball_visible_pixels(env: Any, segmentation: np.ndarray) -> int:
    geom_ids = {
        env.sim.model.geom_name2id(name)
        for name in (*env.ball.visual_geoms, *env.ball.contact_geoms)
    }
    mask = (segmentation[:, :, 0] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(
        segmentation[:, :, 1], list(geom_ids)
    )
    return int(mask.sum())


def render_task_setup(
    *,
    task_config_path: Path,
    output_dir: Path,
    seed: int,
    width: int,
    height: int,
) -> Path:
    """Render both policy cameras at one coherent initial boundary."""
    if output_dir.exists():
        raise FileExistsError(f"render output already exists: {output_dir}")
    if width <= 0 or height <= 0:
        raise ValueError("render dimensions must be positive")
    spec = load_task_spec(task_config_path)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.building-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"render staging output already exists: {staging}")

    env = make_dynamic_grasp_lift_environment(spec=spec, seed=seed, offscreen=True)
    try:
        snapshotter = BoundarySnapshotter(
            camera_names=spec.policy_camera_names,
            segmentation_camera_names=spec.policy_camera_names,
            width=width,
            height=height,
        )
        snapshot = snapshotter.capture(
            env=env,
            ledger=ClockLedger(physics_dt_us=2_000, formal_tick_us=20_000),
            commanded_world={},
        )
        staging.mkdir()
        cameras: dict[str, Any] = {}
        for camera_name in spec.policy_camera_names:
            sample = snapshot.cameras[camera_name]
            if sample.segmentation is None:
                raise RuntimeError(f"missing segmentation for camera: {camera_name}")
            image_name = f"{camera_name}.png"
            image_path = staging / image_name
            Image.fromarray(sample.rgb).save(image_path)
            camera_id = env.sim.model.camera_name2id(camera_name)
            parent_body_id = int(env.sim.model.cam_bodyid[camera_id])
            cameras[camera_name] = {
                "ball_visible_pixels": _ball_visible_pixels(env, sample.segmentation),
                "fovy_degrees": float(env.sim.model.cam_fovy[camera_id]),
                "image": image_name,
                "model_parent_body": env.sim.model.body_id2name(parent_body_id),
                "model_pose_frame": "world" if parent_body_id == 0 else "parent_body",
                "model_position": np.asarray(env.sim.model.cam_pos[camera_id]).tolist(),
                "model_quaternion_wxyz": np.asarray(env.sim.model.cam_quat[camera_id]).tolist(),
                "sha256": _sha256(image_path),
                "world_position_at_capture": np.asarray(
                    env.sim.data.cam_xpos[camera_id]
                ).tolist(),
                "world_rotation_matrix_at_capture": np.asarray(
                    env.sim.data.cam_xmat[camera_id]
                )
                .reshape(3, 3)
                .tolist(),
            }
        manifest = {
            "agentview_resource_id": spec.agentview_resource_id,
            "ball_diameter_m": spec.ball_diameter_m,
            "cameras": cameras,
            "height": height,
            "schema_version": 2,
            "seed": seed,
            "source_formal_tick": snapshot.formal_tick_index,
            "source_physics_step": snapshot.physics_step_index,
            "source_time_us": snapshot.time_us,
            "task_config_sha256": _sha256(task_config_path),
            "task_id": spec.task_id,
            "width": width,
        }
        _write_json(staging / "manifest.json", manifest)
        staging.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        env.close()
    return output_dir / "manifest.json"
