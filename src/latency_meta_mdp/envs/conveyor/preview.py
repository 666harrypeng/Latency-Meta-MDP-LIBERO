"""Record a passive surface-conveyor scene with the native two-camera layout."""

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from PIL import Image

from latency_meta_mdp.envs.conveyor.arrivals import ARRIVAL_PROTOCOL_ID
from latency_meta_mdp.envs.conveyor.config import apply_view_profile, load_conveyor_spec
from latency_meta_mdp.envs.conveyor.task import make_conveyor_runtime
from latency_meta_mdp.io.policy_video import DualCameraVideoWriter


def record_preview(config_path, output_dir, *, seed=27, belt_speed=None, view_path=None):
    spec = load_conveyor_spec(config_path)
    if view_path is not None:
        spec = apply_view_profile(spec, view_path)
    if belt_speed is not None:
        spec = replace(
            spec,
            belt_speed_mps=belt_speed,
            spacing_speed_mps=spec.spacing_speed_mps * belt_speed / spec.belt_speed_mps,
        )
    spec.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    runtime = make_conveyor_runtime(spec, seed=seed, offscreen=True)
    env, world = runtime.env, runtime.world
    started = time.monotonic()
    status = {
        "task_id": "conveyor_sort",
        "variant": "surface",
        "delivery_protocol": world.goal.protocol_id,
        "view_profile": str(view_path) if view_path else "default",
        "seed": seed,
        "scope": "passive_scene_preview_no_policy_or_grasping",
        "status": "running",
        "config": asdict(spec),
        "physics_dt_us": 2000,
        "formal_tick_us": 20000,
        "arrival_protocol": ARRIVAL_PROTOCOL_ID,
    }
    (output / "arrivals.json").write_text(json.dumps([asdict(e) for e in world.arrivals], indent=2))
    (output / "status.json").write_text(json.dumps(status, indent=2))
    hold = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    initial_qpos = env.sim.data.qpos[env.robots[0]._ref_joint_pos_indexes].copy()
    maximum_qpos_drift = 0.0
    camera_ids = [env.sim.model.camera_name2id(name) for name in spec.scene.policy_camera_names]
    status["cameras"] = {
        name: {
            "parent_body": int(env.sim.model.cam_bodyid[i]),
            "position": env.sim.model.cam_pos[i].tolist(),
            "quaternion": env.sim.model.cam_quat[i].tolist(),
            "fovy": float(env.sim.model.cam_fovy[i]),
        }
        for name, i in zip(spec.scene.policy_camera_names, camera_ids)
    }
    try:
        observation = runtime.executor.initialize()
        with (
            DualCameraVideoWriter(output / "preview.mp4") as video,
            (output / "trajectory.jsonl").open("x") as trace,
        ):
            while True:
                video.write(observation)
                trace.write(json.dumps(world.trace(observation.formal_tick)) + "\n")
                drift = np.max(np.abs(observation.state[:7] - initial_qpos))
                maximum_qpos_drift = max(maximum_qpos_drift, float(drift))
                if observation.formal_tick in (0, 150, 975, 1100, 1550):
                    Image.fromarray(
                        np.concatenate([observation.image, observation.wrist_image], axis=1)
                    ).save(output / f"frame-{observation.formal_tick:05d}.png")
                if observation.formal_tick % 100 == 0 or env.done:
                    progress = {"tick": observation.formal_tick, **world.ledger.summary()}
                    print(json.dumps(progress), flush=True)
                    (output / "progress.json").write_text(json.dumps(progress, indent=2))
                if env.done:
                    break
                observation = runtime.executor.step_formal(hold)
        status.update(
            status="completed",
            **world.ledger.summary(),
            frames=video.frame_count,
            video="preview.mp4",
            wall_seconds=time.monotonic() - started,
            robot_max_qpos_drift_rad=maximum_qpos_drift,
            drive_invocations=world.drive_invocations,
            collisions=world.collision_events,
            events=world.ledger.events,
            delivery_events=world.goal.events,
            all_scheduled_spawned=world.next_arrival == len(world.arrivals),
        )
        if not status["all_scheduled_spawned"]:
            raise RuntimeError("preview did not honor its full arrival schedule")
    except BaseException as error:
        status.update(status="failed", error=str(error))
        raise
    finally:
        (output / "status.json").write_text(json.dumps(status, indent=2))
        env.close()
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/tasks/conveyor_sort/surface.yaml")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--belt-speed", type=float)
    parser.add_argument("--view", type=Path, help="Optional parallel main-camera view profile")
    args = parser.parse_args()
    record_preview(
        args.config,
        args.output_dir,
        seed=args.seed,
        belt_speed=args.belt_speed,
        view_path=args.view,
    )


if __name__ == "__main__":
    main()
