from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def _bridge():
    from latency_meta_mdp.expert_realization.robot_bridge import PandaPlanningBridge

    f64 = np.float64
    return PandaPlanningBridge(
        schema_version=1,
        bridge_id="robosuite_panda_to_curobo_franka_v1",
        joint_names=tuple(f"panda_joint{i}" for i in range(1, 8)),
        joint_lower=np.array([-2.8, -1.7, -2.8, -3.0, -2.8, 0.0, -2.8], dtype=f64),
        joint_upper=np.array([2.8, 1.7, 2.8, -0.1, 2.8, 3.7, 2.8], dtype=f64),
        joint_velocity=np.array([2.175] * 4 + [2.61] * 3, dtype=f64),
        joint_acceleration=np.full(7, 15.0, dtype=f64),
        home_joint_positions=np.array([0.0, 0.2, 0.0, -2.6, 0.0, 2.9, 0.8], dtype=f64),
        base_position_world=np.array([-0.56, 0.0, 0.912], dtype=f64),
        base_rotation_world=np.eye(3, dtype=f64),
        hand_to_tcp_position=np.array([0.0, 0.0, 0.097], dtype=f64),
        hand_to_tcp_rotation=np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        open_gripper_positions=np.array([0.04, 0.04], dtype=f64),
        robosuite_version="1.5.2",
        curobo_version="0.8.0",
        robosuite_urdf_sha256="a" * 64,
        curobo_urdf_sha256="b" * 64,
        curobo_config_sha256="c" * 64,
        robosuite_asset_tree_sha256="d" * 64,
        curobo_asset_tree_sha256="e" * 64,
        curobo_tool_frame="panda_hand",
        robosuite_eef_site="gripper0_right_grip_site",
    )


def test_bridge_contract_is_exact_read_only_and_round_trips() -> None:
    """Break caught: joint/frame/assets drift without changing the bridge identity."""
    from latency_meta_mdp.expert_realization.robot_bridge import PandaPlanningBridge

    bridge = _bridge()
    loaded = PandaPlanningBridge.from_mapping(bridge.to_mapping())

    assert loaded == bridge
    assert bridge.joint_names == tuple(f"panda_joint{i}" for i in range(1, 8))
    assert np.array_equal(bridge.hand_to_tcp_position, [0.0, 0.0, 0.097])
    assert np.array_equal(
        bridge.hand_to_tcp_rotation,
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    )
    for value in (
        bridge.joint_lower,
        bridge.joint_upper,
        bridge.joint_velocity,
        bridge.joint_acceleration,
        bridge.home_joint_positions,
        bridge.base_position_world,
        bridge.base_rotation_world,
        bridge.hand_to_tcp_position,
        bridge.hand_to_tcp_rotation,
    ):
        assert not value.flags.writeable
    with pytest.raises(ValueError):
        PandaPlanningBridge.from_mapping({**bridge.to_mapping(), "unknown": 1})


def test_parity_set_contains_home_interior_and_each_safe_limit_side() -> None:
    """Break caught: FK qualification quietly degenerates to one home configuration."""
    from latency_meta_mdp.expert_realization.robot_bridge import sample_fk_parity_set

    bridge = _bridge()
    samples = sample_fk_parity_set(bridge, interior_count=16)

    assert samples.shape == (31, 7)
    assert samples.dtype == np.float64
    assert np.array_equal(samples[0], bridge.home_joint_positions)
    assert np.all(samples > bridge.joint_lower)
    assert np.all(samples < bridge.joint_upper)
    assert len(np.unique(samples, axis=0)) == len(samples)
    assert not samples.flags.writeable


def test_bridge_modules_do_not_import_curobo_in_parent_process() -> None:
    """Break caught: normal RoboSuite collection imports CuRobo/CUDA worker dependencies."""
    code = """
import sys
import latency_meta_mdp.expert_realization.robot_bridge
import latency_meta_mdp.expert_realization.planner_protocol
bad = [name for name in sys.modules if name == 'curobo' or name.startswith('curobo.')]
assert not bad, bad
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path.cwd(),
        env={"PYTHONPATH": str(Path.cwd() / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
