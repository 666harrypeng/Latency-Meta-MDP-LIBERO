"""Physical Panda grasp contact and one-way release of a driven ball."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from latency_meta_mdp.backend import AppliedControlSample, PreparedPhysicsPoint
from latency_meta_mdp.control import ActionContract
from latency_meta_mdp.motion import DrivenBallWorld, motion_sample_to_mapping
from latency_meta_mdp.outcomes import EpisodeOutcomeTracker, OutcomeStatus


class HandoffState(str, Enum):
    DRIVEN = "driven"
    PHYSICAL = "physical"
    FAILURE = "failure"


@dataclass(frozen=True)
class PadContactSample:
    physics_step_index: int
    formal_tick_index: int
    time_us: int
    left_pad_contact: bool
    right_pad_contact: bool
    left_contact_count: int
    right_contact_count: int

    @property
    def bilateral_contact(self) -> bool:
        return self.left_pad_contact and self.right_pad_contact

    @property
    def any_pad_contact(self) -> bool:
        return self.left_pad_contact or self.right_pad_contact


class PandaBallContactDetector:
    """Read physical contacts between the Panda finger pads and task ball."""

    def __init__(self, env: Any) -> None:
        robot = env.robots[0]
        gripper = robot.gripper[robot.arms[0]]
        self._env = env
        self._left_geoms = frozenset(gripper.important_geoms["left_fingerpad"])
        self._right_geoms = frozenset(gripper.important_geoms["right_fingerpad"])
        self._ball_geoms = frozenset(env.ball.contact_geoms)
        if not self._left_geoms or not self._right_geoms or not self._ball_geoms:
            raise ValueError("Panda pad and ball collision geoms must be non-empty")

    def sample(
        self,
        *,
        time_us: int,
        physics_step_index: int,
        formal_tick_index: int,
    ) -> PadContactSample:
        left_count = 0
        right_count = 0
        for contact in self._env.sim.data.contact[: self._env.sim.data.ncon]:
            geom1 = self._env.sim.model.geom_id2name(contact.geom1)
            geom2 = self._env.sim.model.geom_id2name(contact.geom2)
            pair = {geom1, geom2}
            if pair & self._ball_geoms:
                left_count += int(bool(pair & self._left_geoms))
                right_count += int(bool(pair & self._right_geoms))
        return PadContactSample(
            physics_step_index=physics_step_index,
            formal_tick_index=formal_tick_index,
            time_us=time_us,
            left_pad_contact=left_count > 0,
            right_pad_contact=right_count > 0,
            left_contact_count=left_count,
            right_contact_count=right_count,
        )


class OneWayHandoff:
    """Coordinate causal contact tracking and one irreversible physical release."""

    def __init__(
        self,
        *,
        env: Any,
        action_contract: ActionContract,
        contact_detector: PandaBallContactDetector,
        outcome_tracker: EpisodeOutcomeTracker,
    ) -> None:
        self._env = env
        self._action_contract = action_contract
        self._contact_detector = contact_detector
        self.outcome_tracker = outcome_tracker
        self.state = HandoffState.DRIVEN
        self.active_gripper_command = action_contract.gripper_open_command
        self.release_time_us: int | None = None
        self.release_qpos: np.ndarray | None = None
        self.release_qvel: np.ndarray | None = None
        self.last_contact: PadContactSample | None = None
        self.reset_episode(outcome_tracker)

    def reset_episode(self, outcome_tracker: EpisodeOutcomeTracker) -> None:
        if outcome_tracker.status is not OutcomeStatus.RUNNING or outcome_tracker.events:
            raise ValueError("episode reset requires a fresh outcome tracker")
        self.outcome_tracker = outcome_tracker
        self.state = HandoffState.DRIVEN
        self.active_gripper_command = self._action_contract.gripper_open_command
        self.release_time_us = None
        self.release_qpos = None
        self.release_qvel = None
        self.last_contact = None
        self._env.external_outcome_authority = self

    def fingerprint_payload(self) -> dict[str, Any]:
        """Return a detached serialization of causal handoff state."""

        contact = self.last_contact
        return {
            "state": self.state.value,
            "active_gripper_command": float(self.active_gripper_command),
            "release_time_us": self.release_time_us,
            "release_qpos": (
                None if self.release_qpos is None else np.asarray(self.release_qpos).tolist()
            ),
            "release_qvel": (
                None if self.release_qvel is None else np.asarray(self.release_qvel).tolist()
            ),
            "last_contact": (
                None
                if contact is None
                else {
                    "physics_step_index": contact.physics_step_index,
                    "formal_tick_index": contact.formal_tick_index,
                    "time_us": contact.time_us,
                    "left_pad_contact": contact.left_pad_contact,
                    "right_pad_contact": contact.right_pad_contact,
                    "left_contact_count": contact.left_contact_count,
                    "right_contact_count": contact.right_contact_count,
                }
            ),
            "outcome": self.outcome_tracker.fingerprint_payload(),
        }

    def _require_authority(self) -> None:
        if self._env.external_outcome_authority is not self:
            raise RuntimeError(
                "handoff coordinator was invalidated by environment reset or rebound"
            )

    def on_control_applied(self, sample: AppliedControlSample) -> None:
        self._require_authority()
        _arm_action, gripper_command = self._action_contract.split_action(sample.action)
        self.active_gripper_command = gripper_command

    def on_physics_point(self, point: PreparedPhysicsPoint) -> dict[str, np.ndarray]:
        self._require_authority()
        contact = self._contact_detector.sample(
            time_us=point.time_us,
            physics_step_index=point.physics_step_index,
            formal_tick_index=point.formal_tick_index,
        )
        self.last_contact = contact
        self.outcome_tracker.observe_contact(
            time_us=point.time_us,
            any_pad_contact=contact.any_pad_contact,
            bilateral_contact=contact.bilateral_contact,
            closing_command=self.active_gripper_command > 0,
        )
        committed = False
        if point.at_formal_boundary:
            committed = self.outcome_tracker.evaluate_boundary(
                time_us=point.time_us,
                lift_height_m=self._env.goal_measurements().lift_height_m,
                request_handoff=(
                    self.state is HandoffState.DRIVEN
                    and self.outcome_tracker.handoff_eligible
                ),
            )
            if committed:
                self._commit_release(point.time_us)
            self._apply_terminal_state()
        return {
            "contact_any": np.array(contact.any_pad_contact, dtype=np.bool_),
            "contact_bilateral": np.array(contact.bilateral_contact, dtype=np.bool_),
            "contact_left": np.array(contact.left_pad_contact, dtype=np.bool_),
            "contact_right": np.array(contact.right_pad_contact, dtype=np.bool_),
            "handoff_committed": np.array(committed, dtype=np.bool_),
            "handoff_state": np.array(self.state.value),
            "world_control_active": np.array(
                self.state is HandoffState.DRIVEN,
                dtype=np.bool_,
            ),
        }

    def _commit_release(self, time_us: int) -> None:
        if self.state is not HandoffState.DRIVEN or self.release_time_us is not None:
            raise RuntimeError("driven ball handoff may only commit once")
        qpos = np.array(
            self._env.sim.data.get_joint_qpos(self._env.ball.joints[0]),
            copy=True,
        )
        qvel = np.array(
            self._env.sim.data.get_joint_qvel(self._env.ball.joints[0]),
            copy=True,
        )
        qpos.setflags(write=False)
        qvel.setflags(write=False)
        self.release_time_us = time_us
        self.release_qpos = qpos
        self.release_qvel = qvel
        self.state = HandoffState.PHYSICAL

    def _apply_terminal_state(self) -> None:
        if self.outcome_tracker.status is OutcomeStatus.RUNNING:
            return
        if self.outcome_tracker.status is OutcomeStatus.FAILURE:
            self.state = HandoffState.FAILURE
            self._env.task_failure = True
        self._env.terminal_reason = self.outcome_tracker.terminal_reason.value
        self._env.done = True


@dataclass
class HandoffAwareBallWorld:
    driver: DrivenBallWorld
    handoff: OneWayHandoff
    apply_count: int = 0

    def __call__(self, env: Any, current_time_us: int) -> dict[str, np.ndarray]:
        if self.handoff.state is HandoffState.DRIVEN:
            self.apply_count += 1
            return self.driver(env, current_time_us)
        sample = self.driver.profile.sample(current_time_us)
        return motion_sample_to_mapping(sample, motion_level=self.driver.motion_level)
