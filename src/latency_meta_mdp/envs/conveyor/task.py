"""Panda conveyor scene with task-local layout and the existing split-step executor."""

from dataclasses import dataclass
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np

from latency_meta_mdp.envs.backend import FormalStepExecutor, RoboSuitePlant
from latency_meta_mdp.envs.control import ActionContract, load_action_contract
from latency_meta_mdp.envs.conveyor.arrivals import sample_arrivals
from latency_meta_mdp.envs.conveyor.config import ConveyorSpec
from latency_meta_mdp.envs.conveyor.world import ConveyorWorld
from latency_meta_mdp.envs.snapshots import make_offscreen_context_current
from latency_meta_mdp.io.paths import repository_root
from latency_meta_mdp.runtime.policy_execution import PolicyObservation
from latency_meta_mdp.runtime.timing import ClockLedger


def _vector(values):
    return " ".join(str(float(v)) for v in values)


def _environment_class():
    from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
    from robosuite.models.arenas import TableArena
    from robosuite.models.objects import BallObject
    from robosuite.models.tasks import ManipulationTask

    class ConveyorEnv(ManipulationEnv):
        def __init__(self, spec, arrivals, controller, seed, offscreen):
            self.conveyor_spec, self.arrivals = spec, arrivals
            self.boundary_reward = 0.0
            super().__init__(
                robots="Panda",
                controller_configs=controller,
                initialization_noise=None,
                use_camera_obs=False,
                has_renderer=False,
                has_offscreen_renderer=offscreen,
                render_camera="agentview",
                control_freq=50,
                lite_physics=True,
                horizon=10000,
                ignore_done=True,
                hard_reset=False,
                camera_names=list(spec.scene.policy_camera_names),
                camera_heights=256,
                camera_widths=256,
                camera_depths=False,
                camera_segmentations=None,
                seed=seed,
            )

        def _load_model(self):
            super()._load_model()
            spec, scene = self.conveyor_spec, self.conveyor_spec.scene
            robot = self.robots[0].robot_model
            robot.set_base_xpos(robot.base_xpos_offset["table"](scene.table_full_size[0]))
            arena = TableArena(
                table_full_size=spec.table_full_size,
                table_friction=(1.0, 0.005, 0.0001),
                table_offset=spec.table_offset,
            )
            arena.set_origin([0, 0, 0])
            arena.set_camera(
                camera_name="agentview",
                pos=spec.agentview_position,
                quat=scene.agentview_quaternion_wxyz,
                camera_attribs={"fovy": str(scene.agentview_fovy_degrees)},
            )
            arena.worldbody.append(
                ET.Element(
                    "geom",
                    name="conveyor_surface",
                    type="box",
                    pos=_vector([*spec.transport_center_xy, spec.belt_top_z - 0.005]),
                    size=_vector(
                        [spec.transport_size_xy[0] / 2, spec.transport_size_xy[1] / 2, 0.005]
                    ),
                    rgba="0.16 0.19 0.23 1",
                    friction="1 0.005 0.0001",
                    group="1",
                )
            )
            arena.worldbody.append(
                ET.Element(
                    "site",
                    name="conveyor_goal",
                    type="sphere",
                    pos=_vector(spec.goal_center),
                    size=_vector([spec.goal_radius_m]),
                    rgba=_vector(spec.goal_rgba),
                )
            )
            self.parcels = [
                BallObject(
                    name=f"parcel_{e.parcel_id}",
                    size=[scene.ball_radius_m],
                    density=scene.ball_density_kg_m3,
                    friction=list(scene.ball_friction),
                    rgba=list(e.color),
                    joints="default",
                )
                for e in self.arrivals
            ]
            # Compile gravity compensation support for inactive object slots.
            # Runtime activation turns it off for each freely moving parcel.
            for parcel in self.parcels:
                parcel.get_obj().set("gravcomp", "1")
            self.model = ManipulationTask(
                mujoco_arena=arena, mujoco_robots=[robot], mujoco_objects=self.parcels
            )

        def _setup_references(self):
            super()._setup_references()
            model = self.sim.model
            self.belt_geom_id = model.geom_name2id("conveyor_surface")
            self.parcel_body_ids = np.array([model.body_name2id(p.root_body) for p in self.parcels])
            self.parcel_geom_ids = [
                np.array([model.geom_name2id(g) for g in (*p.contact_geoms, *p.visual_geoms)])
                for p in self.parcels
            ]
            self.contact_to_parcel = {
                model.geom_name2id(g): i
                for i, p in enumerate(self.parcels)
                for g in p.contact_geoms
            }
            # RoboSuite refreshes references on soft reset, after the slots have
            # already been parked. Preserve the original compiled contact masks.
            if not hasattr(self, "original_masks"):
                self.original_masks = [
                    (model.geom_contype[ids].copy(), model.geom_conaffinity[ids].copy())
                    for ids in self.parcel_geom_ids
                ]

        def park(self, parcel_id):
            ids, body = self.parcel_geom_ids[parcel_id], self.parcel_body_ids[parcel_id]
            self.sim.model.geom_contype[ids] = 0
            self.sim.model.geom_conaffinity[ids] = 0
            self.sim.model.geom_rgba[ids, 3] = 0
            self.sim.model.body_gravcomp[body] = 1
            joint = self.parcels[parcel_id].joints[0]
            self.sim.data.set_joint_qpos(joint, [0, 0, -2 - parcel_id, 1, 0, 0, 0])
            self.sim.data.set_joint_qvel(joint, np.zeros(6))
            self.sim.data.xfrc_applied[body] = 0

        def activate(self, event, position):
            i = event.parcel_id
            ids, body = self.parcel_geom_ids[i], self.parcel_body_ids[i]
            self.sim.model.geom_contype[ids], self.sim.model.geom_conaffinity[ids] = (
                self.original_masks[i]
            )
            self.sim.model.geom_rgba[ids] = event.color
            self.sim.model.body_gravcomp[body] = 0
            self.sim.data.set_joint_qpos(self.parcels[i].joints[0], [*position, 1, 0, 0, 0])
            speed = self.conveyor_spec.belt_speed_mps
            self.sim.data.set_joint_qvel(
                self.parcels[i].joints[0],
                [0, speed, 0, -speed / self.conveyor_spec.scene.ball_radius_m, 0, 0],
            )

        def _reset_internal(self):
            super()._reset_internal()
            self.boundary_reward = 0.0
            for i in range(len(self.parcels)):
                self.park(i)
            self.sim.forward()

        def reward(self, action=None):
            return self.boundary_reward

        def _check_success(self):
            return False  # Task reporting uses the per-parcel ledger, not Lift's boolean goal.

        def _destroy_sim(self):
            make_offscreen_context_current(self)
            super()._destroy_sim()

    return ConveyorEnv


class ConveyorSnapshotter:
    def __init__(self, size=256):
        self.size = size

    def capture(self, *, env, ledger, commanded_world):
        ledger.validate_sim_time(float(env.sim.data.time))
        if not env.has_offscreen_renderer:
            return None  # Headless physics checks must not masquerade as camera observations.
        make_offscreen_context_current(env)
        images = [
            np.flipud(env.sim.render(width=self.size, height=self.size, camera_name=name))
            for name in env.conveyor_spec.scene.policy_camera_names
        ]
        robot, data = env.robots[0], env.sim.data
        arm = robot.arms[0]
        gripper = data.qpos[robot._ref_gripper_joint_pos_indexes[arm]]
        gripvel = data.qvel[robot._ref_gripper_joint_vel_indexes[arm]]
        state = np.concatenate(
            [
                data.qpos[robot._ref_joint_pos_indexes],
                data.qvel[robot._ref_joint_vel_indexes],
                [gripper[0] - gripper[1], gripvel[0] - gripvel[1]],
            ]
        )
        return PolicyObservation(
            ledger.formal_tick_index, *images, state, prompt=env.conveyor_spec.instruction
        )


@dataclass
class ConveyorRuntime:
    spec: ConveyorSpec
    env: Any
    world: ConveyorWorld
    executor: FormalStepExecutor
    action_contract: ActionContract


def make_conveyor_runtime(
    spec: ConveyorSpec, *, seed: int, offscreen: bool = True, camera_size: int = 256
) -> ConveyorRuntime:
    import mujoco

    arrivals = sample_arrivals(spec, seed=seed)
    contract = load_action_contract(repository_root() / spec.action_contract_path)
    env = _environment_class()(spec, arrivals, contract.to_robosuite_config(), seed, offscreen)
    try:
        env.sim.model.opt.integrator = int(mujoco.mjtIntegrator.mjINT_EULER)
        env.reset()
        contract.verify_runtime(env)
        world = ConveyorWorld(env, spec, arrivals)
        plant = RoboSuitePlant(
            env=env,
            snapshotter=ConveyorSnapshotter(camera_size),
            world_writer=world.before_step,
            physics_point_observer=world.after_step1,
            control_observer=world.on_control,
        )
        executor = FormalStepExecutor(
            plant=plant, ledger=ClockLedger(physics_dt_us=2000, formal_tick_us=20000)
        )
        return ConveyorRuntime(spec, env, world, executor, contract)
    except BaseException:
        env.close()
        raise
