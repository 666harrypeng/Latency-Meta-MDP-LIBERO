"""Isolated CuRobo worker operations; never import this into the RoboSuite process."""

from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path

import numpy as np

from latency_meta_mdp.expert_realization.planner_protocol import (
    FkBatch,
    load_fk_request,
    write_fk_result,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _verify_curobo_assets(bridge: object) -> None:
    from latency_meta_mdp.expert_realization.robot_bridge import _asset_tree_sha256, _sha256_file

    root = _project_root()
    asset_root = root / "third_party/curobo/curobo/content/assets/robot/franka_description"
    urdf = asset_root / "franka_panda.urdf"
    config = root / "third_party/curobo/curobo/content/configs/robot/franka.yml"
    if _sha256_file(urdf) != bridge.curobo_urdf_sha256:
        raise ValueError("CuRobo URDF identity mismatch")
    if _sha256_file(config) != bridge.curobo_config_sha256:
        raise ValueError("CuRobo config identity mismatch")
    if _asset_tree_sha256(asset_root) != bridge.curobo_asset_tree_sha256:
        raise ValueError("CuRobo asset-tree identity mismatch")


def run_fk(request_root: Path, result_root: Path) -> None:
    import torch
    from curobo.kinematics import Kinematics, KinematicsCfg
    from curobo.types import JointState

    bridge, qpos, request_sha = load_fk_request(request_root)
    _verify_curobo_assets(bridge)
    if version("nvidia-curobo") != bridge.curobo_version:
        raise ValueError("installed CuRobo version does not match the bridge")
    robot = Kinematics(KinematicsCfg.from_robot_yaml_file("franka.yml"))
    if tuple(robot.joint_names) != bridge.joint_names or robot.tool_frames != [
        bridge.curobo_tool_frame
    ]:
        raise ValueError("CuRobo runtime joint/tool contract mismatch")
    positions, rotations = [], []
    for row in qpos:
        value = torch.tensor(row[None], device="cuda", dtype=torch.float32)
        state = robot.compute_kinematics(
            JointState.from_position(value, joint_names=list(bridge.joint_names))
        )
        hand = state.tool_poses.get_link_pose(bridge.curobo_tool_frame).get_matrix()[0]
        base_to_hand = hand.detach().cpu().numpy().astype(np.float64)
        hand_to_tcp = np.eye(4, dtype=np.float64)
        hand_to_tcp[:3, :3] = bridge.hand_to_tcp_rotation
        hand_to_tcp[:3, 3] = bridge.hand_to_tcp_position
        base_to_world = np.eye(4, dtype=np.float64)
        base_to_world[:3, :3] = bridge.base_rotation_world
        base_to_world[:3, 3] = bridge.base_position_world
        tcp_in_world = base_to_world @ base_to_hand @ hand_to_tcp
        positions.append(tcp_in_world[:3, 3])
        rotations.append(tcp_in_world[:3, :3])
    write_fk_result(
        result_root,
        request_sha256=request_sha,
        batch=FkBatch(
            positions_world=np.asarray(positions, dtype=np.float64),
            rotations_world=np.asarray(rotations, dtype=np.float64),
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one isolated CuRobo bridge operation.")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    run_fk(args.request, args.result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
