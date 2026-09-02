"""Isolated CuRobo worker operations; never import this into the RoboSuite process."""

from __future__ import annotations

import argparse
import json
import random
import time
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


def _tcp_world_to_hand_base(
    bridge: object, position: np.ndarray, rotation: np.ndarray
) -> np.ndarray:
    tcp_world = np.eye(4, dtype=np.float64)
    tcp_world[:3, :3] = rotation
    tcp_world[:3, 3] = position
    base_world = np.eye(4, dtype=np.float64)
    base_world[:3, :3] = bridge.base_rotation_world
    base_world[:3, 3] = bridge.base_position_world
    hand_tcp = np.eye(4, dtype=np.float64)
    hand_tcp[:3, :3] = bridge.hand_to_tcp_rotation
    hand_tcp[:3, 3] = bridge.hand_to_tcp_position
    return np.linalg.inv(base_world) @ tcp_world @ np.linalg.inv(hand_tcp)


def _eef_positions_world(robot: object, bridge: object, qpos_path: np.ndarray) -> np.ndarray:
    import torch
    from curobo.types import JointState

    qpos = torch.tensor(qpos_path, device="cuda", dtype=torch.float32)
    state = robot.compute_kinematics(JointState.from_position(qpos, joint_names=robot.joint_names))
    hand = state.tool_poses.get_link_pose(bridge.curobo_tool_frame).get_matrix()
    matrices = hand.detach().cpu().numpy().astype(np.float64)
    hand_tcp = np.eye(4, dtype=np.float64)
    hand_tcp[:3, :3] = bridge.hand_to_tcp_rotation
    hand_tcp[:3, 3] = bridge.hand_to_tcp_position
    base_world = np.eye(4, dtype=np.float64)
    base_world[:3, :3] = bridge.base_rotation_world
    base_world[:3, 3] = bridge.base_position_world
    return np.stack([(base_world @ matrix @ hand_tcp)[:3, 3] for matrix in matrices])


def run_planner_request(request_path: Path, result_root: Path) -> None:
    import torch
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import GoalToolPose, JointState
    from scipy.spatial.transform import Rotation

    from latency_meta_mdp.expert_realization.contracts import ExpertRealizationKey
    from latency_meta_mdp.expert_realization.planner import (
        PlannerCandidate,
        PlannerCandidateStatus,
        schedule_candidate_timestamps,
        validate_scheduled_joint_path,
        write_single_candidate_result,
    )
    from latency_meta_mdp.expert_realization.robot_bridge import PandaPlanningBridge

    raw = json.loads(Path(request_path).read_text())
    expected_fields = {
        "schema_version",
        "format_id",
        "expert_realization_key",
        "structured_expert_config_sha256",
        "candidate_index",
        "requested_seed",
        "bridge",
        "plan",
        "start_qpos",
        "timeout_seconds",
    }
    if type(raw) is not dict or set(raw) != expected_fields:
        raise ValueError("planner request fields are invalid")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1 or (
        raw["format_id"] != "structured_expert_planner_request_v1"
    ):
        raise ValueError("planner request format is invalid")
    bridge = PandaPlanningBridge.from_mapping(raw["bridge"])
    _verify_curobo_assets(bridge)
    key = ExpertRealizationKey.from_mapping(
        raw["expert_realization_key"],
        structured_expert_config_sha256=raw["structured_expert_config_sha256"],
    )
    index = raw["candidate_index"]
    requested_seed = raw["requested_seed"]
    from latency_meta_mdp.expert_realization.planner import planner_candidate_seed

    if requested_seed != planner_candidate_seed(key, index):
        raise ValueError("planner request seed mismatch")
    effective_seed = requested_seed
    random.seed(effective_seed)
    np.random.seed(effective_seed % 2**32)
    torch.manual_seed(effective_seed)
    timeout_seconds = raw["timeout_seconds"]
    start_qpos = np.asarray(raw["start_qpos"], dtype=np.float64)
    plan = raw["plan"]
    expected_plan_fields = {
        "task_instance_id",
        "family",
        "interception_tick",
        "pregrasp_arrival_tick",
        "reference_start_tick",
        "time_scaling_profile",
        "guide_positions_world",
        "pregrasp_position_world",
        "fixed_orientation_world",
    }
    if type(plan) is not dict or set(plan) != expected_plan_fields:
        raise ValueError("planner request plan fields are invalid")
    if (
        type(plan["reference_start_tick"]) is not int
        or plan["reference_start_tick"] != 5
        or type(plan["pregrasp_arrival_tick"]) is not int
        or plan["pregrasp_arrival_tick"] <= plan["reference_start_tick"]
    ):
        raise ValueError("planner request arrival schedule is invalid")
    targets = [*plan["guide_positions_world"], plan["pregrasp_position_world"]]
    fixed_rotation = np.asarray(plan["fixed_orientation_world"], dtype=np.float64)
    scene = {
        "cuboid": {
            "table": {
                "dims": [0.8, 0.8, 0.05],
                "pose": [0.56, 0.0, -0.137, 1.0, 0.0, 0.0, 0.0],
            }
        }
    }
    started = time.perf_counter()
    planner = MotionPlanner(
        MotionPlannerCfg.create(
            robot="franka.yml",
            scene_model=scene,
            collision_cache={"cuboid": 4},
            self_collision_check=True,
            use_cuda_graph=False,
            num_ik_seeds=16,
            num_trajopt_seeds=2,
            random_seed=effective_seed,
            position_tolerance=0.005,
            orientation_tolerance=0.05,
        )
    )
    planner.warmup(enable_graph=False, num_warmup_iterations=1)
    all_qpos: list[np.ndarray] = []
    all_timestamps: list[np.ndarray] = []
    elapsed_offset = 0.0
    costs, position_errors, rotation_errors = [], [], []
    current = start_qpos
    failure_reason = None
    for target in targets:
        hand_goal = _tcp_world_to_hand_base(
            bridge,
            np.asarray(target, dtype=np.float64),
            fixed_rotation,
        )
        xyzw = Rotation.from_matrix(hand_goal[:3, :3]).as_quat()
        quaternion = np.concatenate([[xyzw[3]], xyzw[:3]])
        current_state = JointState.from_position(
            torch.tensor(current[None], device="cuda", dtype=torch.float32),
            joint_names=planner.joint_names,
        )
        goal = GoalToolPose(
            tool_frames=planner.tool_frames,
            position=torch.tensor(
                hand_goal[:3, 3], device="cuda", dtype=torch.float32
            ).reshape(1, 1, 1, 1, 3),
            quaternion=torch.tensor(
                quaternion, device="cuda", dtype=torch.float32
            ).reshape(1, 1, 1, 1, 4),
        )
        result = planner.plan_pose(goal, current_state, enable_graph_attempt=0, max_attempts=5)
        if result is None or not bool(result.success.any()):
            failure_reason = "CuRobo returned no successful trajectory"
            break
        interpolated = result.get_interpolated_plan()
        segment = interpolated.position[0, 0, :, :7].detach().cpu().numpy().astype(np.float64)
        dt = float(interpolated.dt.reshape(-1)[0].detach().cpu())
        timestamps = elapsed_offset + np.arange(len(segment), dtype=np.float64) * dt
        if all_qpos:
            segment = segment[1:]
            timestamps = timestamps[1:]
        all_qpos.append(segment)
        all_timestamps.append(timestamps)
        elapsed_offset = float(timestamps[-1])
        current = segment[-1]
        costs.append(float(result.seed_cost.reshape(-1)[0].detach().cpu()))
        position_errors.append(float(result.position_error.reshape(-1)[0].detach().cpu()))
        rotation_errors.append(
            float(np.degrees(result.rotation_error.reshape(-1)[0].detach().cpu()))
        )
    planning_time = time.perf_counter() - started
    qpos_path = None
    scheduled_timestamps = None
    if failure_reason is None:
        qpos_path = np.concatenate(all_qpos, axis=0)
        raw_timestamps = np.concatenate(all_timestamps, axis=0)
        arrival_duration = (
            plan["pregrasp_arrival_tick"] - plan["reference_start_tick"]
        ) * 0.02
        try:
            scheduled_timestamps = schedule_candidate_timestamps(
                raw_timestamps,
                arrival_duration_seconds=float(arrival_duration),
                time_scaling_profile=plan["time_scaling_profile"],
            )
            validate_scheduled_joint_path(
                qpos_path,
                scheduled_timestamps,
                joint_lower=bridge.joint_lower,
                joint_upper=bridge.joint_upper,
                joint_velocity=bridge.joint_velocity,
                joint_acceleration=bridge.joint_acceleration,
            )
        except ValueError as error:
            failure_reason = str(error)
    if failure_reason is not None:
        candidate = PlannerCandidate.failure(
            expert_realization_key=key,
            candidate_index=index,
            requested_seed=requested_seed,
            effective_seed=effective_seed,
            status=PlannerCandidateStatus.PLANNER_FAILURE,
            reason=failure_reason,
            planning_time_seconds=planning_time,
        )
    elif planning_time > timeout_seconds:
        candidate = PlannerCandidate.failure(
            expert_realization_key=key,
            candidate_index=index,
            requested_seed=requested_seed,
            effective_seed=effective_seed,
            status=PlannerCandidateStatus.TIMEOUT,
            reason=f"planner call exceeded {timeout_seconds} seconds",
            planning_time_seconds=planning_time,
        )
    else:
        assert qpos_path is not None and scheduled_timestamps is not None
        eef = _eef_positions_world(planner, bridge, qpos_path)
        candidate = PlannerCandidate(
            expert_realization_key=key,
            candidate_index=index,
            requested_seed=requested_seed,
            effective_seed=effective_seed,
            status=PlannerCandidateStatus.SUCCESS,
            qpos_path=qpos_path,
            timestamps_seconds=scheduled_timestamps,
            eef_positions_world=eef,
            eef_path_length_m=float(np.linalg.norm(np.diff(eef, axis=0), axis=1).sum()),
            certified_clearance_lower_bound_m=0.0,
            goal_position_error_m=max(position_errors),
            goal_rotation_error_degrees=max(rotation_errors),
            planner_cost=sum(costs),
            planning_time_seconds=planning_time,
            failure_reason=None,
            deterministic_replay_verified=None,
        )
    write_single_candidate_result(result_root, candidate)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one isolated CuRobo bridge operation.")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--planner-request", type=Path)
    parser.add_argument("--planner-result", type=Path)
    args = parser.parse_args(argv)
    if args.request is not None and args.result is not None and args.planner_request is None:
        run_fk(args.request, args.result)
    elif (
        args.planner_request is not None
        and args.planner_result is not None
        and args.request is None
    ):
        run_planner_request(args.planner_request, args.planner_result)
    else:
        parser.error("choose exactly one complete FK or planner request/result pair")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
