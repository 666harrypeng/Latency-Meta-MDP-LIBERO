"""Materialize the controls that can execute before an in-scope return."""

from __future__ import annotations

import numpy as np

from latency_meta_mdp.belief.conditional_return_flow.branch_contracts import (
    ExecutablePrefix,
)
from latency_meta_mdp.control import ActionContract
from latency_meta_mdp.temporal_contract import TemporalContract


def _validated_actions(value: np.ndarray, *, action_contract: ActionContract) -> np.ndarray:
    actions = np.asarray(value)
    if actions.ndim != 2 or actions.shape[1:] != (action_contract.action_dim,):
        raise ValueError("active actions must have shape [N, action_dim]")
    if not np.all(np.isfinite(actions)):
        raise ValueError("active actions must be finite")
    for row in actions:
        action_contract.split_action(row)
    return np.asarray(actions, dtype=np.float32)


def materialize_executable_prefix(
    *,
    active_actions: np.ndarray,
    active_cursor: int,
    last_executed_gripper_command: float,
    maximum_delay_ticks: int,
    action_contract: ActionContract,
) -> ExecutablePrefix:
    actions = _validated_actions(active_actions, action_contract=action_contract)
    if (
        isinstance(active_cursor, bool)
        or not isinstance(active_cursor, int)
        or not 0 <= active_cursor <= len(actions)
    ):
        raise ValueError("active cursor is outside the installed action chunk")
    if maximum_delay_ticks != 20:
        raise ValueError("Conditional Return Flow executable prefixes require D20")
    if not np.isfinite(last_executed_gripper_command):
        raise ValueError("last executed gripper command must be finite")
    if active_cursor:
        _, observed_gripper = action_contract.split_action(actions[active_cursor - 1])
        if abs(observed_gripper - last_executed_gripper_command) > 1e-7:
            raise ValueError("last executed gripper command disagrees with active cursor")
    else:
        action_contract.compose_action(
            arm_reference=np.zeros(action_contract.arm_dim),
            gripper_command=last_executed_gripper_command,
        )

    remaining = actions[active_cursor:]
    real_count = min(len(remaining), maximum_delay_ticks)
    controls = np.zeros((maximum_delay_ticks, action_contract.action_dim), dtype=np.float32)
    controls[:real_count] = remaining[:real_count]
    hold_gripper = (
        float(last_executed_gripper_command)
        if real_count == 0
        else float(controls[real_count - 1, -1])
    )
    controls[real_count:, -1] = hold_gripper
    for row in controls:
        action_contract.split_action(row)
    return ExecutablePrefix(
        controls=controls,
        from_active_buffer_mask=np.arange(maximum_delay_ticks) < real_count,
        last_executed_gripper_command=float(last_executed_gripper_command),
    )


def materialize_teacher_executable_prefix(
    *,
    expert_actions: np.ndarray,
    source_tick: int,
    temporal: TemporalContract,
    action_contract: ActionContract,
) -> ExecutablePrefix:
    actions = _validated_actions(expert_actions, action_contract=action_contract)
    if temporal.maximum_delay_ticks != 20:
        raise ValueError("teacher prefix requires the D20 temporal contract")
    chunk_start, chunk_stop = temporal.teacher_chunk_bounds(
        source_tick=source_tick,
        episode_action_count=len(actions),
    )
    chunk = actions[chunk_start:chunk_stop]
    last_gripper = float(actions[source_tick - 1, -1])
    return materialize_executable_prefix(
        active_actions=chunk,
        active_cursor=temporal.launch_trigger_horizon,
        last_executed_gripper_command=last_gripper,
        maximum_delay_ticks=temporal.maximum_delay_ticks,
        action_contract=action_contract,
    )
