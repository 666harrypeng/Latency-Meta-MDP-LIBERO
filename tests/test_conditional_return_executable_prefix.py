from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from latency_meta_mdp.belief.conditional_return_flow.executable_prefix import (
    materialize_executable_prefix,
    materialize_teacher_executable_prefix,
)
from latency_meta_mdp.control import load_action_contract
from latency_meta_mdp.temporal_contract import load_temporal_contract


def _contract():
    return load_action_contract(Path("configs/control/panda_osc_pose_delta_v1.yaml"))


def _temporal():
    return load_temporal_contract(Path("configs/temporal/h50_e25_d20_k6_v1.yaml"))


def test_hold_uses_last_executed_gripper_not_future_expert() -> None:
    active = np.zeros((3, 7), dtype=np.float32)
    active[:, 6] = np.array([-1.0, -0.5, -0.5])

    prefix = materialize_executable_prefix(
        active_actions=active,
        active_cursor=3,
        last_executed_gripper_command=-0.5,
        maximum_delay_ticks=20,
        action_contract=_contract(),
    )

    assert np.all(prefix.controls[:, :6] == 0)
    assert np.all(prefix.controls[:, 6] == -0.5)
    assert not np.any(prefix.from_active_buffer_mask)


def test_partial_buffer_holds_last_remaining_gripper_command() -> None:
    active = np.zeros((4, 7), dtype=np.float32)
    active[:, 0] = np.array([0.1, 0.2, 0.3, 0.4])
    active[:, 6] = np.array([-1.0, -1.0, 1.0, 1.0])

    prefix = materialize_executable_prefix(
        active_actions=active,
        active_cursor=2,
        last_executed_gripper_command=-1.0,
        maximum_delay_ticks=20,
        action_contract=_contract(),
    )

    np.testing.assert_array_equal(prefix.controls[:2], active[2:])
    assert prefix.from_active_buffer_mask.tolist() == [True, True] + [False] * 18
    assert np.all(prefix.controls[2:, :6] == 0)
    assert np.all(prefix.controls[2:, 6] == 1.0)


def test_prefix_rejects_cursor_and_last_gripper_disagreement() -> None:
    active = np.zeros((3, 7), dtype=np.float32)
    active[:, 6] = np.array([-1.0, -1.0, 1.0])

    with pytest.raises(ValueError, match="last executed gripper"):
        materialize_executable_prefix(
            active_actions=active,
            active_cursor=3,
            last_executed_gripper_command=-1.0,
            maximum_delay_ticks=20,
            action_contract=_contract(),
        )


def test_teacher_prefix_reconstructs_h50_e25() -> None:
    expert_actions = np.zeros((100, 7), dtype=np.float32)
    expert_actions[:, 0] = np.linspace(-0.8, 0.8, 100, dtype=np.float32)
    expert_actions[:35, 6] = -1.0
    expert_actions[35:, 6] = 1.0

    prefix = materialize_teacher_executable_prefix(
        expert_actions=expert_actions,
        source_tick=40,
        temporal=_temporal(),
        action_contract=_contract(),
    )

    np.testing.assert_array_equal(prefix.controls, expert_actions[40:60])
    assert np.all(prefix.from_active_buffer_mask)
    assert prefix.last_executed_gripper_command == expert_actions[39, 6]


def test_prefix_rejects_out_of_contract_control_without_clipping() -> None:
    active = np.zeros((1, 7), dtype=np.float32)
    active[0, 0] = 1e9

    with pytest.raises(ValueError, match="outside the action contract"):
        materialize_executable_prefix(
            active_actions=active,
            active_cursor=0,
            last_executed_gripper_command=-1.0,
            maximum_delay_ticks=20,
            action_contract=_contract(),
        )
