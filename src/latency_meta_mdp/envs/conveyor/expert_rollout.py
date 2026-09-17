"""Validate the physical conveyor teacher and optionally record both policy cameras."""

import argparse
import json
import time
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from PIL import Image

from latency_meta_mdp.envs.conveyor.arrivals import ARRIVAL_PROTOCOL_ID
from latency_meta_mdp.envs.conveyor.config import load_conveyor_spec
from latency_meta_mdp.envs.conveyor.expert import (
    EXPERT_PROTOCOL_ID,
    ConveyorExpert,
    load_expert_spec,
)
from latency_meta_mdp.envs.conveyor.task import make_conveyor_runtime
from latency_meta_mdp.io.policy_video import DualCameraVideoWriter


def run_expert(
    config_path,
    expert_path,
    output_dir,
    *,
    seed=27,
    video=True,
    belt_speed=None,
    record_source=False,
    source_purpose="development_smoke",
):
    spec, expert_spec = load_conveyor_spec(config_path), load_expert_spec(expert_path)
    if belt_speed is not None:
        spec = replace(
            spec,
            belt_speed_mps=belt_speed,
            spacing_speed_mps=spec.spacing_speed_mps * belt_speed / spec.belt_speed_mps,
        )
    spec.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = dict(
        status="running",
        scope="physical_expert_validation_not_learned_policy_eval",
        task_id="conveyor_sort",
        seed=seed,
        config=asdict(spec),
        expert_config=asdict(expert_spec),
        expert_protocol=EXPERT_PROTOCOL_ID,
        physics_dt_us=2000,
        formal_tick_us=20000,
        arrival_protocol=ARRIVAL_PROTOCOL_ID,
    )
    (output / "status.json").write_text(json.dumps(status, indent=2))
    runtime = None
    try:
        runtime = make_conveyor_runtime(spec, seed=seed, offscreen=video or record_source)
        env, world = runtime.env, runtime.world
        status["action_contract"] = runtime.action_contract.contract_id
        status["controller_config"] = runtime.action_contract.to_robosuite_config()
        expert = ConveyorExpert(runtime, expert_spec)
        status["delivery_protocol"] = world.goal.protocol_id
        (output / "arrivals.json").write_text(
            json.dumps([asdict(e) for e in world.arrivals], indent=2)
        )
        observation = runtime.executor.initialize()
        with ExitStack() as stack:
            recorder = None
            if record_source:
                from latency_meta_mdp.data.conveyor.source import ConveyorRecorder

                recorder = stack.enter_context(
                    ConveyorRecorder(
                        output / "source",
                        seed=seed,
                        action_contract_id=runtime.action_contract.contract_id,
                        purpose=source_purpose,
                        context={
                            "scene": asdict(spec),
                            "expert": asdict(expert_spec),
                            "expert_protocol": EXPERT_PROTOCOL_ID,
                            "controller_config": status["controller_config"],
                            "delivery_protocol": world.goal.protocol_id,
                            "arrival_protocol": ARRIVAL_PROTOCOL_ID,
                        },
                    )
                )
                recorder.start(observation)
            writer = (
                stack.enter_context(DualCameraVideoWriter(output / "expert.mp4")) if video else None
            )
            trace = stack.enter_context((output / "trajectory.jsonl").open("x"))
            last_phase, last_events = None, 0
            while True:
                tick = runtime.executor.ledger.formal_tick_index
                if writer:
                    writer.write(observation)
                row = world.trace(tick)
                row["eef"] = env.sim.data.site_xpos[expert.site].tolist()
                fingers = env.sim.data.qpos[
                    env.robots[0]._ref_gripper_joint_pos_indexes[expert.arm]
                ]
                row["gripper_width"] = float(fingers[0] - fingers[1])
                row["statuses"] = dict(world.ledger.statuses)
                if env.done:
                    trace.write(json.dumps(row) + "\n")
                    break
                decision = expert.next_action()
                row.update(
                    phase=decision.phase,
                    target_parcel=decision.parcel_id,
                    target_eef=decision.target_eef.tolist(),
                    action=decision.action.tolist(),
                    grasped=[
                        i
                        for i, state in world.ledger.statuses.items()
                        if state == "active"
                        and env._check_grasp(env.robots[0].gripper, env.parcels[i])
                    ],
                )
                trace.write(json.dumps(row) + "\n")
                if (
                    tick % 100 == 0
                    or len(world.ledger.events) != last_events
                    or decision.phase != last_phase
                ):
                    progress = dict(
                        tick=tick,
                        phase=decision.phase,
                        target=decision.parcel_id,
                        **world.ledger.summary(),
                    )
                    print(json.dumps(progress), flush=True)
                    (output / "progress.json").write_text(json.dumps(progress, indent=2))
                if video and decision.phase != last_phase:
                    Image.fromarray(
                        np.concatenate([observation.image, observation.wrist_image], axis=1)
                    ).save(output / f"frame-{tick:05d}-{decision.phase}.png")
                last_phase, last_events = decision.phase, len(world.ledger.events)
                observation = runtime.executor.step_formal(decision.action)
                if recorder:
                    recorder.append(
                        decision.action,
                        observation,
                        success_delta=int(env.boundary_reward),
                        done=bool(env.done),
                    )
            if world.next_arrival != len(world.arrivals):
                raise RuntimeError("expert rollout missed scheduled arrivals")
            outcome = world.ledger.summary()
            status["expert_admitted"] = (
                outcome["spawned"] > 0 and outcome["successes"] == outcome["spawned"]
            )
            if recorder and status["expert_admitted"]:
                recorder.finish(world.ledger.summary())
                status["source_manifest"] = "source/manifest.json"
            elif recorder:
                status["source_rejected"] = "episode_not_fully_successful"
            if writer:
                status.update(video="expert.mp4", frames=writer.frame_count)
        status.update(
            status="completed",
            **world.ledger.summary(),
            all_scheduled_spawned=world.next_arrival == len(world.arrivals),
            events=world.ledger.events,
            delivery_events=world.goal.events,
            expert_events=expert.events,
            collisions=world.collision_events,
        )
        if not status["all_scheduled_spawned"]:
            raise RuntimeError("expert rollout missed scheduled arrivals")
    except BaseException as error:
        status.update(status="failed", error=str(error))
        raise
    finally:
        status["wall_seconds"] = time.monotonic() - started
        (output / "status.json").write_text(json.dumps(status, indent=2))
        if runtime is not None:
            runtime.env.close()
    print(
        json.dumps(
            {
                key: status[key]
                for key in ("status", "spawned", "successes", "misses", "timeouts", "end_tick")
            }
        ),
        flush=True,
    )
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/tasks/conveyor_sort/surface.yaml")
    )
    parser.add_argument(
        "--expert-config", type=Path, default=Path("configs/data/expert/conveyor_surface.yaml")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--belt-speed", type=float)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--record-source", action="store_true", help="Record a lossless development source episode"
    )
    args = parser.parse_args()
    run_expert(
        args.config,
        args.expert_config,
        args.output_dir,
        seed=args.seed,
        belt_speed=args.belt_speed,
        video=not args.no_video,
        record_source=args.record_source,
    )


if __name__ == "__main__":
    main()
