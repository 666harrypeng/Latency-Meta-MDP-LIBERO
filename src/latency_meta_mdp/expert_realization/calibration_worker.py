"""Isolated historical-feedback worker for non-training timing calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from latency_meta_mdp.expert import (
    ExpertObservation,
    ExpertPhase,
    ScriptedBallExpert,
    load_expert_config,
)
from latency_meta_mdp.expert_realization.calibration import (
    TimingCalibrationAttemptRequest,
    TimingCalibrationRow,
    write_timing_calibration_attempt_result,
)
from latency_meta_mdp.expert_realization.shared_prefix import (
    _build_shared_prefix_anchor,
    _compare_shared_prefix_anchors,
)
from latency_meta_mdp.expert_realization.task_instance import (
    _build_task_instance_runtime,
    _sha256_bytes,
    materialize_task_instance,
)
from latency_meta_mdp.outcomes import OutcomeStatus


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_request(path: Path) -> TimingCalibrationAttemptRequest:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("calibration request must contain one JSON mapping") from error
    return TimingCalibrationAttemptRequest.from_mapping(value)


def _verify_request_sources(root: Path, request: TimingCalibrationAttemptRequest) -> None:
    expected = {
        "task_config_sha256": root / "configs/task/dynamic_grasp_lift_l0.yaml",
        "motion_config_sha256": root
        / f"configs/motion/dynamic_grasp_lift_l{request.level}.yaml",
        "runtime_config_sha256": root / "configs/runtime/robosuite_v1.yaml",
        "controller_config_sha256": root / "configs/control/panda_osc_pose_delta_v1.yaml",
        "expert_config_sha256": root / "configs/expert/panda_ball_feedback_v1.yaml",
    }
    mismatches = [
        name
        for name, path in expected.items()
        if _sha256(path) != getattr(request, name)
    ]
    if mismatches:
        raise ValueError(f"calibration request source hashes changed: {mismatches}")


def _kinematic_maxima(eef_positions: list[np.ndarray]) -> tuple[float, float, float]:
    positions = np.asarray(eef_positions, dtype=np.float64)
    velocity = np.diff(positions, axis=0) / 0.02
    acceleration = np.diff(velocity, axis=0) / 0.02
    jerk = np.diff(acceleration, axis=0) / 0.02

    def maximum_norm(value: np.ndarray) -> float:
        return 0.0 if not len(value) else float(np.linalg.norm(value, axis=1).max())

    return maximum_norm(velocity), maximum_norm(acceleration), maximum_norm(jerk)


def _relative_speed(previous: object, current: object) -> float:
    eef_velocity = (np.asarray(current.eef_pos) - np.asarray(previous.eef_pos)) / 0.02
    object_velocity = (
        np.asarray(current.object_body_pos) - np.asarray(previous.object_body_pos)
    ) / 0.02
    return float(np.linalg.norm(eef_velocity - object_velocity))


def run_matched_feedback_calibration_attempt(
    *,
    project_root: Path,
    request: TimingCalibrationAttemptRequest,
) -> TimingCalibrationRow:
    root = Path(project_root).resolve()
    _verify_request_sources(root, request)
    task = materialize_task_instance(
        project_root=root,
        level=request.level,
        task_instance_seed=request.master_task_seed,
    )
    runtime = _build_task_instance_runtime(task)
    expert = ScriptedBallExpert(
        action_contract=runtime.action_contract,
        config=load_expert_config(root / "configs/expert/panda_ball_feedback_v1.yaml"),
    )
    phase_ticks: dict[ExpertPhase, int] = {}
    close_distance = None
    close_relative_speed = None
    handoff_relative_speed = None
    saturated_count = 0
    decision_count = 0
    eef_positions: list[np.ndarray] = []
    try:
        snapshot = runtime.executor.initialize()
        boundaries = [snapshot]
        for action in task.expected_anchor.shared_actions:
            arm = runtime.env.robots[0].part_controllers["right"]
            expert.next_action(
                observation=ExpertObservation.from_snapshot(
                    snapshot,
                    world_to_base_rotation=arm.origin_ori.T,
                ),
                handoff_state=runtime.handoff.state,
            )
            snapshot = runtime.executor.step_formal(action)
            boundaries.append(snapshot)
        actual_anchor = _build_shared_prefix_anchor(
            runtime=runtime,
            boundaries=tuple(boundaries),
            shared_actions=task.expected_anchor.shared_actions,
            motion_profile_sha256=task.task_instance_id.motion_profile_sha256,
            shared_endpoint_sha256=_sha256_bytes(task.shared_endpoint_bytes),
        )
        _compare_shared_prefix_anchors(task.expected_anchor, actual_anchor)

        previous = boundaries[-2]
        eef_positions.append(np.asarray(snapshot.eef_pos, dtype=np.float64))
        maximum_steps = expert.config.collection_max_duration_us // 20_000
        while runtime.tracker.status is OutcomeStatus.RUNNING and (
            snapshot.formal_tick_index < maximum_steps
        ):
            arm = runtime.env.robots[0].part_controllers["right"]
            decision = expert.next_action(
                observation=ExpertObservation.from_snapshot(
                    snapshot,
                    world_to_base_rotation=arm.origin_ori.T,
                ),
                handoff_state=runtime.handoff.state,
            )
            phase_ticks.setdefault(decision.phase, snapshot.formal_tick_index)
            if decision.phase is ExpertPhase.CLOSE and close_distance is None:
                close_distance = float(
                    np.linalg.norm(
                        np.asarray(snapshot.eef_pos) - np.asarray(snapshot.object_body_pos)
                    )
                )
                close_relative_speed = _relative_speed(previous, snapshot)
            saturated_count += int(np.any(np.isclose(np.abs(decision.action[:6]), 1.0)))
            decision_count += 1
            previous = snapshot
            snapshot = runtime.executor.step_formal(decision.action)
            eef_positions.append(np.asarray(snapshot.eef_pos, dtype=np.float64))
            if runtime.tracker.handoff_us is not None and handoff_relative_speed is None:
                handoff_relative_speed = _relative_speed(previous, snapshot)

        speed, acceleration, jerk = _kinematic_maxima(eef_positions)
        lift_threshold = next(
            (event.time_us for event in runtime.tracker.events if event.kind == "lift_threshold"),
            None,
        )
        success = runtime.tracker.status is OutcomeStatus.SUCCESS
        return TimingCalibrationRow(
            logical_task_index=request.logical_task_index,
            level=request.level,
            terminal_status="success" if success else "failure",
            terminal_reason=(
                "collection_budget_exhausted"
                if runtime.tracker.terminal_reason is None
                else runtime.tracker.terminal_reason.value
            ),
            pregrasp_tick=phase_ticks.get(ExpertPhase.APPROACH),
            approach_tick=phase_ticks.get(ExpertPhase.APPROACH),
            close_tick=phase_ticks.get(ExpertPhase.CLOSE),
            lift_start_tick=phase_ticks.get(ExpertPhase.LIFT),
            first_contact_time_us=runtime.tracker.first_contact_us,
            stable_contact_time_us=runtime.tracker.stable_grasp_us,
            handoff_time_us=runtime.tracker.handoff_us,
            lift_threshold_time_us=lift_threshold,
            terminal_time_us=runtime.tracker.terminal_time_us,
            close_distance_m=close_distance,
            close_relative_speed_mps=close_relative_speed,
            handoff_relative_speed_mps=handoff_relative_speed,
            saturation_fraction=(
                0.0 if decision_count == 0 else float(saturated_count / decision_count)
            ),
            maximum_eef_speed_mps=speed,
            maximum_eef_acceleration_mps2=acceleration,
            maximum_eef_jerk_mps3=jerk,
            k6_planning_start_sha256=actual_anchor.planning_start_sha256,
        )
    finally:
        runtime.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one matched-K6 feedback calibration attempt")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    row = run_matched_feedback_calibration_attempt(
        project_root=Path.cwd(),
        request=_load_request(args.request),
    )
    write_timing_calibration_attempt_result(row, args.result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
