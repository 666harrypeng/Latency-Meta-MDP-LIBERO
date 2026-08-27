"""Deterministic MuJoCo state injection and agent-view segmentation renders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from latency_meta_mdp.belief.flow.ghost_state import ReconstructedReturnState


def _readonly(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    copied = np.array(value, dtype=dtype, copy=True)
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True)
class AgentViewRender:
    rgb: np.ndarray
    segmentation: np.ndarray
    robot_mask: np.ndarray
    ball_mask: np.ndarray
    eef_position: np.ndarray
    robot_qpos: np.ndarray
    object_position: np.ndarray

    def __post_init__(self) -> None:
        height, width = np.asarray(self.rgb).shape[:2]
        shapes = {
            "rgb": (height, width, 3),
            "segmentation": (height, width, 2),
            "robot_mask": (height, width),
            "ball_mask": (height, width),
            "eef_position": (3,),
            "robot_qpos": (7,),
            "object_position": (3,),
        }
        if height <= 0 or width <= 0:
            raise ValueError("agent-view render dimensions are invalid")
        for name, shape in shapes.items():
            if np.asarray(getattr(self, name)).shape != shape:
                raise ValueError(f"agent-view render {name} has invalid shape")
        if np.asarray(self.rgb).dtype != np.uint8:
            raise ValueError("agent-view RGB must be uint8")
        for name, dtype in (
            ("rgb", np.uint8),
            ("segmentation", np.int32),
            ("robot_mask", np.bool_),
            ("ball_mask", np.bool_),
            ("eef_position", np.float64),
            ("robot_qpos", np.float64),
            ("object_position", np.float64),
        ):
            value = np.asarray(getattr(self, name))
            if np.issubdtype(value.dtype, np.floating) and not np.all(np.isfinite(value)):
                raise ValueError(f"agent-view render {name} must be finite")
            object.__setattr__(self, name, _readonly(value, dtype=dtype))


class GhostEnvironmentAdapter:
    def __init__(
        self,
        *,
        env: Any,
        camera_name: str,
        width: int,
        height: int,
    ) -> None:
        if camera_name != "agentview" or width <= 0 or height <= 0:
            raise ValueError("ghost environment requires a valid agentview render")
        self.env = env
        self.camera_name = camera_name
        self.width = width
        self.height = height
        robot = env.robots[0]
        arm = robot.arms[0]
        self._robot_qpos_indexes = np.asarray(robot._ref_joint_pos_indexes, dtype=int)
        self._robot_qvel_indexes = np.asarray(robot._ref_joint_vel_indexes, dtype=int)
        self._gripper_qpos_indexes = np.asarray(
            robot._ref_gripper_joint_pos_indexes[arm], dtype=int
        )
        self._gripper_qvel_indexes = np.asarray(
            robot._ref_gripper_joint_vel_indexes[arm], dtype=int
        )
        self._eef_site_id = int(robot.eef_site_id[arm])
        joint_indexes = np.asarray(robot._ref_joint_indexes, dtype=int)
        self.joint_ranges = _readonly(env.sim.model.jnt_range[joint_indexes], dtype=np.float64)
        gripper = robot.gripper[arm]
        gripper_joint_ids = np.asarray(
            [env.sim.model.joint_name2id(name) for name in gripper.joints],
            dtype=int,
        )
        gripper_ranges = np.asarray(env.sim.model.jnt_range[gripper_joint_ids])
        self.gripper_width_range = (
            float(gripper_ranges[0, 0] - gripper_ranges[1, 1]),
            float(gripper_ranges[0, 1] - gripper_ranges[1, 0]),
        )
        table_half = np.asarray(env.table_full_size[:2], dtype=np.float64) / 2.0
        table_z = float(env.table_offset[2])
        self.object_position_bounds = _readonly(
            np.asarray(
                [
                    [-table_half[0], table_half[0]],
                    [-table_half[1], table_half[1]],
                    [table_z, table_z + 0.6],
                ]
            ),
            dtype=np.float64,
        )
        self._robot_geom_ids = self._descendant_geom_ids(robot.robot_model.root_body)
        self._ball_geom_ids = frozenset(
            env.sim.model.geom_name2id(name)
            for name in (*env.ball.visual_geoms, *env.ball.contact_geoms)
        )

    def _descendant_geom_ids(self, root_body_name: str) -> frozenset[int]:
        model = self.env.sim.model
        root_id = int(model.body_name2id(root_body_name))
        descendants = {root_id}
        changed = True
        while changed:
            changed = False
            for body_id in range(model.nbody):
                if body_id not in descendants and int(model.body_parentid[body_id]) in descendants:
                    descendants.add(body_id)
                    changed = True
        return frozenset(
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) in descendants
        )

    def _mask(self, segmentation: np.ndarray, geom_ids: frozenset[int]) -> np.ndarray:
        return (segmentation[:, :, 0] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(
            segmentation[:, :, 1], tuple(sorted(geom_ids))
        )

    def _apply_state(
        self,
        *,
        state: ReconstructedReturnState,
        sim_time_seconds: float,
    ) -> None:
        if not isinstance(state, ReconstructedReturnState):
            raise TypeError("ghost rendering requires a reconstructed return state")
        if not np.isfinite(sim_time_seconds) or sim_time_seconds < 0.0:
            raise ValueError("ghost render simulation time is invalid")
        data = self.env.sim.data
        data.qpos[self._robot_qpos_indexes] = state.robot_qpos
        data.qvel[self._robot_qvel_indexes] = state.robot_qvel
        data.qpos[self._gripper_qpos_indexes] = state.gripper_qpos
        data.qvel[self._gripper_qvel_indexes] = state.gripper_qvel
        data.set_joint_qpos(self.env.ball.joints[0], state.object_qpos)
        data.set_joint_qvel(self.env.ball.joints[0], state.object_qvel)
        data.time = sim_time_seconds
        self.env.sim.forward()

    def forward_state(
        self,
        *,
        state: ReconstructedReturnState,
        sim_time_seconds: float,
    ) -> np.ndarray:
        self._apply_state(state=state, sim_time_seconds=sim_time_seconds)
        return _readonly(self.env.sim.data.site_xpos[self._eef_site_id], dtype=np.float64)

    def render_state(
        self,
        *,
        state: ReconstructedReturnState,
        sim_time_seconds: float,
    ) -> AgentViewRender:
        self._apply_state(state=state, sim_time_seconds=sim_time_seconds)
        data = self.env.sim.data
        rgb = np.flipud(
            self.env.sim.render(
                width=self.width,
                height=self.height,
                camera_name=self.camera_name,
            )
        ).copy()
        segmentation = np.flipud(
            self.env.sim.render(
                width=self.width,
                height=self.height,
                camera_name=self.camera_name,
                segmentation=True,
            )
        ).copy()
        return AgentViewRender(
            rgb=rgb,
            segmentation=segmentation,
            robot_mask=self._mask(segmentation, self._robot_geom_ids),
            ball_mask=self._mask(segmentation, self._ball_geom_ids),
            eef_position=data.site_xpos[self._eef_site_id],
            robot_qpos=data.qpos[self._robot_qpos_indexes],
            object_position=data.body_xpos[self.env.ball_body_id],
        )
