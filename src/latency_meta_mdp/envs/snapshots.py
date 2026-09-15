"""Direct boundary snapshots that bypass RoboSuite's observable cache."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from latency_meta_mdp.runtime.timing import ClockLedger


def _readonly_copy(value: Any) -> np.ndarray:
    copied = np.array(value, copy=True)
    copied.setflags(write=False)
    return copied


def make_offscreen_context_current(env: Any) -> None:
    """Select the owning GL context before rendering or freeing its MuJoCo objects."""
    context = getattr(getattr(env, "sim", None), "_render_context_offscreen", None)
    if context is not None:
        context.gl_ctx.make_current()


@dataclass(frozen=True)
class CameraSample:
    name: str
    rgb: np.ndarray
    segmentation: np.ndarray | None
    source_physics_step: int
    source_formal_tick: int
    source_time_us: int


@dataclass(frozen=True)
class BoundarySnapshot:
    physics_step_index: int
    formal_tick_index: int
    time_us: int
    sim_time_seconds: float
    qpos: np.ndarray
    qvel: np.ndarray
    act: np.ndarray
    actuator_ctrl: np.ndarray
    robot_qpos: np.ndarray
    robot_qvel: np.ndarray
    robot_gripper_qpos: np.ndarray
    robot_gripper_qvel: np.ndarray
    eef_pos: np.ndarray
    eef_xmat: np.ndarray
    object_qpos: np.ndarray
    object_qvel: np.ndarray
    object_body_pos: np.ndarray
    object_body_quat_wxyz: np.ndarray
    commanded_world: Mapping[str, np.ndarray]
    cameras: Mapping[str, CameraSample]


class BoundarySnapshotter:
    """Read direct MuJoCo state and synchronously render declared cameras."""

    def __init__(
        self,
        *,
        camera_names: tuple[str, ...],
        segmentation_camera_names: tuple[str, ...] = (),
        width: int,
        height: int,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("camera dimensions must be positive")
        unknown = set(segmentation_camera_names) - set(camera_names)
        if unknown:
            raise ValueError(f"segmentation cameras are not RGB cameras: {sorted(unknown)}")
        self.camera_names = camera_names
        self.segmentation_camera_names = frozenset(segmentation_camera_names)
        self.width = width
        self.height = height

    def capture(
        self,
        *,
        env: Any,
        ledger: ClockLedger,
        commanded_world: Mapping[str, Any],
    ) -> BoundarySnapshot:
        if not ledger.at_formal_boundary:
            raise ValueError("boundary snapshots require a formal boundary")
        ledger.validate_sim_time(float(env.sim.data.time))

        robot = env.robots[0]
        task_object = env.task_object
        task_object_body_id = env.task_object_body_id
        arm = robot.arms[0]
        eef_site_id = robot.eef_site_id[arm]
        robot_qpos_indexes = np.asarray(robot._ref_joint_pos_indexes, dtype=int)
        robot_qvel_indexes = np.asarray(robot._ref_joint_vel_indexes, dtype=int)
        gripper_qpos_indexes = np.asarray(robot._ref_gripper_joint_pos_indexes[arm], dtype=int)
        gripper_qvel_indexes = np.asarray(robot._ref_gripper_joint_vel_indexes[arm], dtype=int)

        cameras: dict[str, CameraSample] = {}
        if self.camera_names:
            # RoboSuite 1.5.2 binds only at construction; GT replay uses a second context.
            make_offscreen_context_current(env)
        for camera_name in self.camera_names:
            rgb = _readonly_copy(
                np.flipud(
                    env.sim.render(
                        width=self.width,
                        height=self.height,
                        camera_name=camera_name,
                    )
                )
            )
            segmentation = None
            if camera_name in self.segmentation_camera_names:
                segmentation = _readonly_copy(
                    np.flipud(
                        env.sim.render(
                            width=self.width,
                            height=self.height,
                            camera_name=camera_name,
                            segmentation=True,
                        )
                    )
                )
            cameras[camera_name] = CameraSample(
                name=camera_name,
                rgb=rgb,
                segmentation=segmentation,
                source_physics_step=ledger.physics_step_index,
                source_formal_tick=ledger.formal_tick_index,
                source_time_us=ledger.time_us,
            )

        frozen_world = {
            name: _readonly_copy(value) for name, value in sorted(commanded_world.items())
        }
        return BoundarySnapshot(
            physics_step_index=ledger.physics_step_index,
            formal_tick_index=ledger.formal_tick_index,
            time_us=ledger.time_us,
            sim_time_seconds=float(env.sim.data.time),
            qpos=_readonly_copy(env.sim.data.qpos),
            qvel=_readonly_copy(env.sim.data.qvel),
            act=_readonly_copy(env.sim.data.act),
            actuator_ctrl=_readonly_copy(env.sim.data.ctrl),
            robot_qpos=_readonly_copy(env.sim.data.qpos[robot_qpos_indexes]),
            robot_qvel=_readonly_copy(env.sim.data.qvel[robot_qvel_indexes]),
            robot_gripper_qpos=_readonly_copy(env.sim.data.qpos[gripper_qpos_indexes]),
            robot_gripper_qvel=_readonly_copy(env.sim.data.qvel[gripper_qvel_indexes]),
            eef_pos=_readonly_copy(env.sim.data.site_xpos[eef_site_id]),
            eef_xmat=_readonly_copy(env.sim.data.site_xmat[eef_site_id].reshape(3, 3)),
            object_qpos=_readonly_copy(env.sim.data.get_joint_qpos(task_object.joints[0])),
            object_qvel=_readonly_copy(env.sim.data.get_joint_qvel(task_object.joints[0])),
            object_body_pos=_readonly_copy(env.sim.data.body_xpos[task_object_body_id]),
            object_body_quat_wxyz=_readonly_copy(env.sim.data.body_xquat[task_object_body_id]),
            commanded_world=MappingProxyType(frozen_world),
            cameras=MappingProxyType(cameras),
        )
