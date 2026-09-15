"""Isolated CuRobo worker operations; never import this into the RoboSuite process."""

from __future__ import annotations

import argparse
import json
import random
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

from latency_meta_mdp.data.collection.planner_protocol import (
    FkBatch,
    load_fk_request,
    write_fk_result,
)
from latency_meta_mdp.io.paths import repository_root


def _project_root() -> Path:
    return repository_root()


def _verify_curobo_assets(bridge: object) -> None:
    from latency_meta_mdp.data.collection.robot_bridge import _asset_tree_sha256, _sha256_file

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


def _parse_approach_geometry(
    intent: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw_guides = intent["soft_guide_regions_world"]
    if type(raw_guides) is not list:
        raise ValueError("soft approach guides must be a JSON list")
    guides = np.asarray(raw_guides, dtype=np.float64)
    if not raw_guides:
        guides = np.empty((0, 3), dtype=np.float64)
    guide_radii = np.asarray(intent["soft_guide_radius_m"], dtype=np.float64)
    entry = np.asarray(intent["funnel_entry_position_world"], dtype=np.float64)
    entry_tangent = np.asarray(intent["funnel_entry_tangent_world"], dtype=np.float64)
    if (
        guides.ndim != 2
        or guides.shape[1:] != (3,)
        or guide_radii.shape != (len(guides),)
        or len(guides) > 2
        or entry.shape != (3,)
        or entry_tangent.shape != (3,)
        or not np.all(np.isfinite(guides))
        or not np.all(np.isfinite(guide_radii))
        or not np.all(np.isfinite(entry))
        or not np.all(np.isfinite(entry_tangent))
        or np.any(guide_radii <= 0.0)
        or not np.isclose(np.linalg.norm(entry_tangent), 1.0, atol=1.0e-9)
    ):
        raise ValueError("planner request approach geometry is invalid")
    return guides, guide_radii, entry, entry_tangent


def _planner_invocation_timeout_reason(
    invocation_seconds: tuple[float, ...], timeout_seconds: float
) -> str | None:
    for index, duration in enumerate(invocation_seconds):
        if duration > timeout_seconds:
            return f"CuRobo pose invocation {index} exceeded {timeout_seconds} seconds"
    return None


def run_planner_request(request_path: Path, result_root: Path) -> None:
    import torch
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import GoalToolPose, JointState
    from scipy.spatial.transform import Rotation

    from latency_meta_mdp.data.collection.contracts import ExpertRealizationKey
    from latency_meta_mdp.data.collection.planner import (
        PlannerCandidate,
        PlannerCandidateStatus,
        write_single_candidate_result,
    )
    from latency_meta_mdp.data.collection.robot_bridge import PandaPlanningBridge
    from latency_meta_mdp.data.collection.trajectory_smoothing import (
        build_cartesian_approach_reference,
    )

    raw = json.loads(Path(request_path).read_text())
    expected_fields = {
        "schema_version",
        "format_id",
        "expert_realization_key",
        "structured_expert_config_sha256",
        "candidate_index",
        "requested_seed",
        "bridge",
        "intent",
        "start_qpos",
        "timeout_seconds",
    }
    if type(raw) is not dict or set(raw) != expected_fields:
        raise ValueError("planner request fields are invalid")
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or (raw["format_id"] != "smooth_approach_planner_request_v1")
    ):
        raise ValueError("planner request format is invalid")
    bridge = PandaPlanningBridge.from_mapping(raw["bridge"])
    _verify_curobo_assets(bridge)
    key = ExpertRealizationKey.from_mapping(raw["expert_realization_key"])
    index = raw["candidate_index"]
    requested_seed = raw["requested_seed"]
    from latency_meta_mdp.data.collection.planner import planner_candidate_seed

    if requested_seed != planner_candidate_seed(key, index):
        raise ValueError("planner request seed mismatch")
    effective_seed = requested_seed
    random.seed(effective_seed)
    np.random.seed(effective_seed % 2**32)
    torch.manual_seed(effective_seed)
    timeout_seconds = raw["timeout_seconds"]
    start_qpos = np.asarray(raw["start_qpos"], dtype=np.float64)
    intent = raw["intent"]
    expected_intent_fields = {
        "task_instance_id",
        "family",
        "reference_start_tick",
        "funnel_entry_target_tick",
        "funnel_entry_deadline_tick",
        "time_scaling_profile",
        "soft_guide_regions_world",
        "soft_guide_radius_m",
        "funnel_entry_position_world",
        "funnel_entry_tangent_world",
        "fixed_orientation_world",
    }
    if type(intent) is not dict or set(intent) != expected_intent_fields:
        raise ValueError("planner request intent fields are invalid")
    if (
        type(intent["reference_start_tick"]) is not int
        or intent["reference_start_tick"] != 5
        or type(intent["funnel_entry_target_tick"]) is not int
        or intent["funnel_entry_target_tick"] <= intent["reference_start_tick"]
        or type(intent["funnel_entry_deadline_tick"]) is not int
        or intent["funnel_entry_deadline_tick"] < intent["funnel_entry_target_tick"]
    ):
        raise ValueError("planner request arrival schedule is invalid")
    guides, guide_radii, entry, entry_tangent = _parse_approach_geometry(intent)
    terminal_tangent_guide = entry - 0.04 * entry_tangent
    targets = [*guides, terminal_tangent_guide, entry]
    fixed_rotation = np.asarray(intent["fixed_orientation_world"], dtype=np.float64)
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
    costs, position_errors, rotation_errors = [], [], []
    invocation_seconds: list[float] = []
    current = start_qpos
    failure_reason = None
    invocation_timeout = False
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
            position=torch.tensor(hand_goal[:3, 3], device="cuda", dtype=torch.float32).reshape(
                1, 1, 1, 1, 3
            ),
            quaternion=torch.tensor(quaternion, device="cuda", dtype=torch.float32).reshape(
                1, 1, 1, 1, 4
            ),
        )
        invocation_started = time.perf_counter()
        result = planner.plan_pose(goal, current_state, enable_graph_attempt=0, max_attempts=5)
        invocation_seconds.append(time.perf_counter() - invocation_started)
        timeout_reason = _planner_invocation_timeout_reason(
            tuple(invocation_seconds),
            timeout_seconds,
        )
        if timeout_reason is not None:
            failure_reason = timeout_reason
            invocation_timeout = True
            break
        if result is None or not bool(result.success.any()):
            failure_reason = "CuRobo returned no successful trajectory"
            break
        interpolated = result.get_interpolated_plan()
        segment = interpolated.position[0, 0, :, :7].detach().cpu().numpy().astype(np.float64)
        if all_qpos:
            segment = segment[1:]
        all_qpos.append(segment)
        current = segment[-1]
        costs.append(float(result.seed_cost.reshape(-1)[0].detach().cpu()))
        position_errors.append(float(result.position_error.reshape(-1)[0].detach().cpu()))
        rotation_errors.append(
            float(np.degrees(result.rotation_error.reshape(-1)[0].detach().cpu()))
        )
    planning_time = time.perf_counter() - started
    geometric_seed_qpos_path = None
    cartesian_reference = None
    proposal_qpos = None
    if failure_reason is None:
        geometric_seed_qpos_path = np.concatenate(all_qpos, axis=0)
        arrival_duration = (
            intent["funnel_entry_target_tick"] - intent["reference_start_tick"]
        ) * 0.02
        try:
            cartesian_reference = build_cartesian_approach_reference(
                start_position_world=_eef_positions_world(
                    planner,
                    bridge,
                    geometric_seed_qpos_path[:1],
                )[0],
                soft_guide_regions_world=guides,
                funnel_entry_position_world=entry,
                funnel_entry_tangent_world=entry_tangent,
                duration_seconds=float(arrival_duration),
                sample_period_seconds=0.02,
            )
            raw_arc = np.concatenate(
                [
                    np.array([0.0]),
                    np.cumsum(
                        np.linalg.norm(
                            np.diff(geometric_seed_qpos_path, axis=0),
                            axis=1,
                        )
                    ),
                ]
            )
            if raw_arc[-1] <= 0.0:
                raise ValueError("CuRobo geometric proposal contains no motion")
            target_arc = np.linspace(
                0.0,
                raw_arc[-1],
                len(cartesian_reference.timestamps_seconds),
                dtype=np.float64,
            )
            proposal_qpos = np.stack(
                [
                    np.interp(target_arc, raw_arc, geometric_seed_qpos_path[:, joint])
                    for joint in range(7)
                ],
                axis=1,
            )
            if np.any(proposal_qpos < bridge.joint_lower) or np.any(
                proposal_qpos > bridge.joint_upper
            ):
                raise ValueError("CuRobo proposal interpolation violates joint position limits")
            eef = cartesian_reference.positions_world
            for guide, radius in zip(guides, guide_radii, strict=True):
                if float(np.linalg.norm(eef - guide, axis=1).min()) > float(radius):
                    raise ValueError("smoothed path misses a soft approach guide")
            if float(np.linalg.norm(eef[-1] - entry)) > 0.005:
                raise ValueError("smoothed path misses the funnel entry")
            terminal_delta = eef[-1] - eef[-2]
            terminal_norm = float(np.linalg.norm(terminal_delta))
            if (
                terminal_norm <= 1.0e-9
                or float(np.dot(terminal_delta / terminal_norm, entry_tangent)) < 0.75
            ):
                raise ValueError("smoothed path misses the funnel-entry tangent")
        except ValueError as error:
            failure_reason = str(error)
    if failure_reason is not None:
        candidate = PlannerCandidate.failure(
            expert_realization_key=key,
            candidate_index=index,
            requested_seed=requested_seed,
            effective_seed=effective_seed,
            status=(
                PlannerCandidateStatus.TIMEOUT
                if invocation_timeout
                else PlannerCandidateStatus.PLANNER_FAILURE
            ),
            reason=failure_reason,
            planning_time_seconds=planning_time,
        )
    else:
        assert (
            geometric_seed_qpos_path is not None
            and cartesian_reference is not None
            and proposal_qpos is not None
        )
        eef = cartesian_reference.positions_world
        candidate = PlannerCandidate(
            expert_realization_key=key,
            candidate_index=index,
            requested_seed=requested_seed,
            effective_seed=effective_seed,
            status=PlannerCandidateStatus.SUCCESS,
            geometric_seed_qpos_path=geometric_seed_qpos_path,
            qpos_path=proposal_qpos,
            timestamps_seconds=cartesian_reference.timestamps_seconds,
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
