"""One-step Panda client adapter for logical-latency infrastructure tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

import numpy as np

from latency_meta_mdp.control import ActionContract
from latency_meta_mdp.latency_harness import LaunchContext, LogicalLatencyHarness

TObservation = TypeVar("TObservation")


class OneStepLatencyClient(Generic[TObservation]):
    """Activate at most one arrived action or execute an explicit hold."""

    def __init__(
        self,
        *,
        action_contract: ActionContract,
        harness: LogicalLatencyHarness[TObservation, np.ndarray],
    ) -> None:
        if getattr(harness, "formal_tick_us", None) != action_contract.formal_tick_us:
            raise ValueError("client and harness must share the action-contract formal clock")
        self.action_contract = action_contract
        self.harness = harness
        self.last_gripper_command = action_contract.gripper_open_command

    def _validated_action(self, value: np.ndarray) -> np.ndarray:
        arm, gripper = self.action_contract.split_action(value)
        return self.action_contract.compose_action(
            arm_reference=arm,
            gripper_command=gripper,
        )

    def _hold_action(self) -> np.ndarray:
        return self.action_contract.compose_action(
            arm_reference=np.zeros(self.action_contract.arm_dim),
            gripper_command=self.last_gripper_command,
        )

    def run_boundary(
        self,
        *,
        formal_tick: int,
        observation: TObservation,
        launch: bool,
        infer: Callable[[LaunchContext[TObservation]], np.ndarray],
        execute: Callable[[np.ndarray], None],
    ) -> np.ndarray:
        if type(launch) is not bool:
            raise TypeError("launch must be boolean")
        if not callable(execute):
            raise TypeError("execute must be callable")
        try:
            arrival = self.harness.open_boundary(formal_tick)
            if arrival is None and launch and not self.harness.pending:
                arrival = self.harness.launch(observation=observation, infer=infer)
            request_id: int | None = None
            if arrival is None:
                action = self._hold_action()
                self.harness.mark_starvation(action=action)
            else:
                self.harness.mark_eligible(arrival)
                action = self._validated_action(np.asarray(arrival.payload, dtype=float))
                self.harness.mark_activated(arrival)
                request_id = arrival.request_id
            try:
                execute(action)
            except BaseException:
                self.harness.mark_faulted()
                raise
            self.harness.mark_execute(request_id=request_id, action=action)
            _arm, gripper = self.action_contract.split_action(action)
            self.last_gripper_command = gripper
            self.harness.close_boundary()
            return action
        except BaseException:
            if not self.harness.faulted:
                self.harness.mark_faulted()
            raise
