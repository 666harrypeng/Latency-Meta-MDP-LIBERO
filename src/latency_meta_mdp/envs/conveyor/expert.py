"""Causal geometry teacher using the same 50 Hz OSC actions as a learned policy.

Object/contact state is privileged teacher input. The teacher never reads the
arrival schedule, resets the arm, attaches a parcel, or writes object motion.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

EXPERT_PROTOCOL_ID = "conveyor_feedback_v2"


@dataclass(frozen=True)
class ExpertSpec:
    workspace_y: tuple
    pregrasp_height_m: float
    carry_height_m: float
    tracking_lead_s: float
    position_tolerance_m: float
    translation_clip_m: float
    close_timeout_ticks: int
    release_timeout_ticks: int
    lost_grasp_ticks: int


def load_expert_spec(path: Path) -> ExpertSpec:
    raw = yaml.safe_load(Path(path).read_text())
    if raw.pop("schema_version") != 1 or raw.pop("expert_id") != EXPERT_PROTOCOL_ID:
        raise ValueError("unsupported conveyor expert config")
    raw["workspace_y"] = tuple(raw["workspace_y"])
    spec = ExpertSpec(**raw)
    if len(spec.workspace_y) != 2 or not spec.workspace_y[0] < spec.workspace_y[1]:
        raise ValueError("expert workspace must have ordered y bounds")
    scalars = (
        spec.pregrasp_height_m,
        spec.carry_height_m,
        spec.tracking_lead_s,
        spec.position_tolerance_m,
        spec.translation_clip_m,
    )
    if not np.isfinite(scalars).all() or min(scalars) <= 0:
        raise ValueError("expert motion parameters must be positive")
    if spec.carry_height_m <= spec.pregrasp_height_m:
        raise ValueError("carry height must clear the pregrasp height")
    if any(
        type(t) is not int or t < 1
        for t in (
            spec.close_timeout_ticks,
            spec.release_timeout_ticks,
            spec.lost_grasp_ticks,
        )
    ):
        raise ValueError("expert dwell limits must be positive integer ticks")
    return spec


@dataclass(frozen=True)
class ExpertDecision:
    tick: int
    phase: str
    parcel_id: int | None
    action: np.ndarray
    target_eef: np.ndarray


class ConveyorExpert:
    def __init__(self, runtime, config: ExpertSpec):
        self.runtime, self.config = runtime, config
        env, robot = runtime.env, runtime.env.robots[0]
        self.arm = robot.arms[0]
        self.site = robot.eef_site_id[self.arm]
        self.rotation = env.sim.data.site_xmat[self.site].reshape(3, 3).copy()
        self.waypoint = env.sim.data.site_xpos[self.site].copy()
        if config.translation_clip_m > min(runtime.action_contract.arm_output_high[:3]):
            raise ValueError("expert translation must fit the existing action contract")
        self.phase, self.parcel_id = "wait", None
        self.phase_tick, self.last_tick = 0, -1
        self.grasp_count, self.lost_count = 0, 0
        self.offset = np.zeros(3)
        self.events = []

    def _transition(self, phase, tick, reason=None):
        self.phase, self.phase_tick = phase, tick
        self.events.append(dict(tick=tick, phase=phase, parcel_id=self.parcel_id, reason=reason))

    def _retreat(self, eef, tick, reason):
        self.waypoint = eef.copy()
        self.waypoint[2] = self.runtime.spec.belt_top_z + self.config.carry_height_m
        self._transition("retreat", tick, reason)
        self.parcel_id = None
        self.grasp_count = self.lost_count = 0

    def next_action(self):
        runtime, cfg = self.runtime, self.config
        env, world, spec = runtime.env, runtime.world, runtime.spec
        tick = runtime.executor.ledger.formal_tick_index
        if tick != self.last_tick + 1:
            raise ValueError("conveyor expert requires every 20 ms boundary")
        self.last_tick = tick
        data, robot = env.sim.data, env.robots[0]
        eef = data.site_xpos[self.site].copy()
        tolerance = cfg.position_tolerance_m
        delivered = False
        if self.parcel_id is not None and world.ledger.statuses[self.parcel_id] != "active":
            delivered = world.ledger.statuses[self.parcel_id] == "success"
            self._retreat(eef, tick, world.ledger.statuses[self.parcel_id])
        # A confirmed goal release already leaves the open fingers above the
        # belt. Start the next high approach immediately instead of climbing at
        # the goal first. Failed low grasps still require vertical clearance.
        if self.phase == "retreat" and (
            (
                delivered
                and eef[2]
                >= spec.belt_top_z + spec.scene.ball_radius_m + cfg.pregrasp_height_m - tolerance
            )
            or np.linalg.norm(eef - self.waypoint) < tolerance
        ):
            self._transition("wait", tick)
        if self.phase == "wait":
            candidates = [
                i
                for i, status in world.ledger.statuses.items()
                if status == "active"
                and cfg.workspace_y[0] <= world.position(i)[1] <= cfg.workspace_y[1]
                and world.position(i)[2] >= spec.belt_top_z
            ]
            if candidates:
                self.parcel_id = max(candidates, key=lambda i: world.position(i)[1])
                self._transition("pregrasp", tick)

        target, grip = self.waypoint.copy(), -1.0
        if self.parcel_id is not None:
            i = self.parcel_id
            pos = world.position(i)
            velocity = np.asarray(data.get_joint_qvel(env.parcels[i].joints[0])[:3])
            predicted = pos + np.clip(velocity, -0.3, 0.3) * cfg.tracking_lead_s
            predicted[2] = pos[2]
            grasped = bool(env._check_grasp(robot.gripper, env.parcels[i]))
            self.grasp_count = self.grasp_count + 1 if grasped else 0
            self.lost_count = 0 if grasped else self.lost_count + 1

            if self.phase == "pregrasp":
                target = predicted + [0, 0, cfg.pregrasp_height_m]
                if np.linalg.norm(eef - (pos + [0, 0, cfg.pregrasp_height_m])) < tolerance:
                    self._transition("descend", tick)
            if self.phase == "descend":
                target = predicted
                if np.linalg.norm(eef - pos) < tolerance:
                    self._transition("close", tick)
            if self.phase == "close":
                target, grip = predicted, 1.0
                if self.grasp_count >= spec.grasp_ticks:
                    self.offset = eef - pos
                    self.waypoint = eef.copy()
                    self.waypoint[2] = spec.belt_top_z + cfg.carry_height_m
                    self._transition("lift", tick)
                elif tick - self.phase_tick >= cfg.close_timeout_ticks:
                    self._retreat(eef, tick, "grasp_timeout")
            if self.phase in {"lift", "transfer", "lower"}:
                target, grip = self.waypoint.copy(), 1.0
                if self.lost_count >= cfg.lost_grasp_ticks:
                    self._retreat(eef, tick, "lost_grasp")
                elif self.phase == "lift" and abs(eef[2] - target[2]) < tolerance:
                    self.waypoint[:2] = np.asarray(spec.goal_center)[:2] + self.offset[:2]
                    self._transition("transfer", tick)
                elif self.phase == "transfer" and np.linalg.norm(eef - target) < tolerance:
                    self.waypoint = np.asarray(spec.goal_center) + self.offset
                    self._transition("lower", tick)
                elif self.phase == "lower" and np.linalg.norm(pos - spec.goal_center) < tolerance:
                    progress = world.goal.history.get(i)
                    if progress is not None and progress.ready_width is not None:
                        self._transition("release", tick)
            if self.phase == "release":
                target, grip = self.waypoint.copy(), -1.0
                if tick - self.phase_tick >= cfg.release_timeout_ticks:
                    self._retreat(eef, tick, "release_timeout")
            if self.phase == "retreat":
                target, grip = self.waypoint.copy(), -1.0

        # Delta pose is expressed in the controller's base frame, with its
        # existing scaling and actuator limits; no direct joint/pose writes.
        world_to_base = robot.part_controllers[self.arm].origin_ori.T
        delta = world_to_base @ (target - eef)
        delta *= min(1.0, cfg.translation_clip_m / max(float(np.linalg.norm(delta)), 1e-12))
        orientation = data.site_xmat[self.site].reshape(3, 3)
        rotvec = Rotation.from_matrix(
            world_to_base @ self.rotation @ orientation.T @ world_to_base.T
        ).as_rotvec()
        contract = runtime.action_contract
        arm_action = np.clip(
            np.concatenate([delta, rotvec]) / contract.arm_output_high,
            contract.arm_input_low,
            contract.arm_input_high,
        )
        action = contract.compose_action(arm_reference=arm_action, gripper_command=grip)
        return ExpertDecision(tick, self.phase, self.parcel_id, action, target)
