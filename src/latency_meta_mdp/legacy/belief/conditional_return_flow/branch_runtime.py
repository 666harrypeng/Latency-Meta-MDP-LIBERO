"""Fresh deterministic replay and branch execution primitives."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from latency_meta_mdp.data.expert_collection import (
    ExpertEpisodeRuntime,
    build_expert_episode_runtime,
)
from latency_meta_mdp.envs.motion import load_motion_config, motion_profile_from_mapping
from latency_meta_mdp.envs.outcomes import OutcomeStatus
from latency_meta_mdp.envs.snapshots import BoundarySnapshot
from latency_meta_mdp.legacy.belief.conditional_return_flow.branch_contracts import BranchRollout
from latency_meta_mdp.legacy.belief.conditional_return_flow.control_continuations import (
    ControlContinuation,
)
from latency_meta_mdp.legacy.belief.conditional_return_flow.source_corpus import (
    SelectedSourceContext,
    VerifiedSourceEpisode,
)


@dataclass(frozen=True)
class ReplayCertification:
    snapshot: BoundarySnapshot
    formal_tick: int
    maximum_physical_state_abs: float
    fingerprint_sha256: str


def _array_payload(value: Any) -> dict[str, object]:
    array = np.asarray(value)
    if array.dtype.kind == "O":
        raise ValueError("source replay fingerprint arrays cannot use object dtype")
    if array.dtype.kind in {"i", "u", "f", "c", "b"} and not np.all(np.isfinite(array)):
        raise ValueError("source replay fingerprint arrays must be finite")
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "values": array.tolist(),
    }


def _mapping_payload(value: dict[str, Any]) -> dict[str, object]:
    result = {}
    for name in sorted(value):
        item = value[name]
        result[name] = _array_payload(item) if isinstance(item, np.ndarray) else item
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _array_payload(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(name): _jsonable(value[name]) for name in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported source fingerprint value: {type(value).__name__}")


def source_replay_fingerprint(
    *,
    runtime: ExpertEpisodeRuntime,
    snapshot: BoundarySnapshot,
) -> str:
    """Hash the complete replay state needed before branch controls diverge."""

    arm = runtime.env.robots[0].part_controllers["right"]
    ledger = runtime.executor.ledger
    payload = {
        "snapshot": {
            "physics_step_index": snapshot.physics_step_index,
            "formal_tick_index": snapshot.formal_tick_index,
            "time_us": snapshot.time_us,
            "sim_time_seconds": snapshot.sim_time_seconds,
            "qpos": _array_payload(snapshot.qpos),
            "qvel": _array_payload(snapshot.qvel),
            "act": _array_payload(snapshot.act),
            "actuator_ctrl": _array_payload(snapshot.actuator_ctrl),
            "commanded_world": _mapping_payload(dict(snapshot.commanded_world)),
        },
        "controller": {
            "goal_pos": _array_payload(arm.goal_pos),
            "goal_ori": _array_payload(arm.goal_ori),
            "origin_pos": _array_payload(arm.origin_pos),
            "origin_ori": _array_payload(arm.origin_ori),
        },
        "clock": {
            "physics_dt_us": ledger.physics_dt_us,
            "formal_tick_us": ledger.formal_tick_us,
            "physics_step_index": ledger.physics_step_index,
            "formal_tick_index": ledger.formal_tick_index,
            "time_us": ledger.time_us,
            "at_formal_boundary": ledger.at_formal_boundary,
            "executor_boundary_prepared": runtime.executor._boundary_prepared,
        },
        "handoff": runtime.handoff.fingerprint_payload(),
        "environment": {
            "done": bool(runtime.env.done),
            "backend_goal_reached": bool(runtime.env.backend_goal_reached()),
            "task_failure": bool(runtime.env.task_failure),
            "terminal_reason": runtime.env.terminal_reason,
        },
        "motion_profile": dict(runtime.metadata.motion_profile),
    }
    encoded = json.dumps(
        _jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def build_branch_runtime(
    *,
    project_root: Path,
    episode: VerifiedSourceEpisode,
) -> ExpertEpisodeRuntime:
    motion_config = load_motion_config(episode.motion_config_path)
    profile = motion_profile_from_mapping(
        dict(episode.motion_profile_mapping),
        config=motion_config,
    )
    return build_expert_episode_runtime(
        project_root=project_root,
        spec=episode.to_expert_spec(),
        motion_profile_override=profile,
    )


def _recorded_source_max_abs(
    *,
    snapshot: BoundarySnapshot,
    arrays: dict[str, np.ndarray] | Any,
    source_tick: int,
) -> float:
    pairs = (
        (snapshot.robot_qpos, arrays["robot_qpos"][source_tick]),
        (snapshot.robot_qvel, arrays["robot_qvel"][source_tick]),
        (snapshot.robot_gripper_qpos, arrays["gripper_qpos"][source_tick]),
        (snapshot.robot_gripper_qvel, arrays["gripper_qvel"][source_tick]),
        (snapshot.object_qpos, arrays["object_pose"][source_tick]),
        (snapshot.object_qvel, arrays["object_velocity"][source_tick]),
        (snapshot.eef_pos, arrays["eef_position_world"][source_tick]),
        (snapshot.eef_xmat, arrays["eef_orientation_matrix_world"][source_tick]),
    )
    return max(
        float(np.max(np.abs(np.asarray(observed) - np.asarray(recorded))))
        for observed, recorded in pairs
    )


def replay_to_source(
    *,
    runtime: ExpertEpisodeRuntime,
    episode: VerifiedSourceEpisode,
    context: SelectedSourceContext,
) -> ReplayCertification:
    if context.episode != episode or context.identity.episode_id != episode.episode_id:
        raise ValueError("replay context and source episode disagree")
    arrays = episode.load_arrays(
        names=(
            "expert_action",
            "robot_qpos",
            "robot_qvel",
            "gripper_qpos",
            "gripper_qvel",
            "object_pose",
            "object_velocity",
            "eef_position_world",
            "eef_orientation_matrix_world",
            "boundary_formal_tick",
            "boundary_time_us",
            "boundary_outcome_status",
            "handoff_state",
        )
    )
    source_tick = context.identity.source_tick
    snapshot = runtime.executor.initialize()
    for tick in range(source_tick):
        snapshot = runtime.executor.step_formal(arrays["expert_action"][tick])
    if (
        snapshot.formal_tick_index != source_tick
        or snapshot.time_us != int(arrays["boundary_time_us"][source_tick])
        or snapshot.formal_tick_index != int(arrays["boundary_formal_tick"][source_tick])
        or runtime.tracker.status.value != str(arrays["boundary_outcome_status"][source_tick])
        or runtime.handoff.state.value != str(arrays["handoff_state"][source_tick])
    ):
        raise RuntimeError("fresh replay source semantic state does not match recorded source")
    maximum = _recorded_source_max_abs(
        snapshot=snapshot,
        arrays=arrays,
        source_tick=source_tick,
    )
    if maximum != 0.0:
        raise RuntimeError(f"fresh replay source physical state mismatch: {maximum}")
    return ReplayCertification(
        snapshot=snapshot,
        formal_tick=source_tick,
        maximum_physical_state_abs=maximum,
        fingerprint_sha256=source_replay_fingerprint(runtime=runtime, snapshot=snapshot),
    )


def snapshot_return_state(snapshot: Any) -> np.ndarray:
    """Pack one synchronized boundary into the public 22D state order."""

    state = np.concatenate(
        (
            np.asarray(snapshot.robot_qpos),
            np.asarray(snapshot.robot_qvel),
            np.asarray(
                [
                    snapshot.robot_gripper_qpos[0] - snapshot.robot_gripper_qpos[1],
                    snapshot.robot_gripper_qvel[0] - snapshot.robot_gripper_qvel[1],
                ]
            ),
            np.asarray(snapshot.object_qpos)[:3],
            np.asarray(snapshot.object_qvel)[:3],
        )
    ).astype(np.float32)
    if state.shape != (22,) or not np.all(np.isfinite(state)):
        raise ValueError("return state must be finite with shape (22,)")
    state.setflags(write=False)
    return state


def recorded_return_states(
    *,
    episode: VerifiedSourceEpisode,
    source_tick: int,
    maximum_delay_ticks: int,
) -> np.ndarray:
    if maximum_delay_ticks != 20:
        raise ValueError("recorded return targets require D20")
    arrays = episode.load_arrays(
        names=(
            "robot_qpos",
            "robot_qvel",
            "gripper_qpos",
            "gripper_qvel",
            "object_pose",
            "object_velocity",
        )
    )
    rows = []
    for tick in range(source_tick + 1, source_tick + maximum_delay_ticks + 1):
        rows.append(
            np.concatenate(
                (
                    arrays["robot_qpos"][tick],
                    arrays["robot_qvel"][tick],
                    np.asarray(
                        [
                            arrays["gripper_qpos"][tick, 0] - arrays["gripper_qpos"][tick, 1],
                            arrays["gripper_qvel"][tick, 0] - arrays["gripper_qvel"][tick, 1],
                        ]
                    ),
                    arrays["object_pose"][tick, :3],
                    arrays["object_velocity"][tick, :3],
                )
            )
        )
    result = np.asarray(rows, dtype=np.float32)
    if result.shape != (20, 22) or not np.all(np.isfinite(result)):
        raise ValueError("recorded return targets must be finite with shape (20, 22)")
    result.setflags(write=False)
    return result


def execute_control_branch(
    *,
    runtime: Any,
    source: SelectedSourceContext | Any,
    replay: ReplayCertification,
    canonical_source_fingerprint: str,
    continuation: ControlContinuation,
    recorded_nominal_targets: np.ndarray | None,
) -> BranchRollout:
    if runtime.tracker.status is not OutcomeStatus.RUNNING:
        raise ValueError("control branch source must be running")
    if continuation.source_context_id != source.identity.source_context_id:
        raise ValueError("control continuation and source context disagree")
    if len(canonical_source_fingerprint) != 64:
        raise ValueError("canonical source fingerprint must be a SHA256 digest")
    fingerprint_match = replay.fingerprint_sha256 == canonical_source_fingerprint
    targets = []
    absorbing = []
    handoff = []
    outcomes = []
    terminal_snapshot = replay.snapshot
    for delay_index in range(20):
        if runtime.tracker.status is OutcomeStatus.RUNNING:
            snapshot = runtime.executor.step_formal(continuation.prefix.controls[delay_index])
            terminal_snapshot = snapshot
            is_absorbing = runtime.tracker.status is not OutcomeStatus.RUNNING
        else:
            snapshot = terminal_snapshot
            is_absorbing = True
        targets.append(snapshot_return_state(snapshot))
        absorbing.append(is_absorbing)
        handoff.append(runtime.handoff.state.value)
        outcomes.append(runtime.tracker.status.value)
    target_array = np.asarray(targets, dtype=np.float32)
    nominal = continuation.spec.kind == "nominal"
    if nominal:
        reference = np.asarray(recorded_nominal_targets)
        if reference.shape != (20, 22) or not np.all(np.isfinite(reference)):
            raise ValueError("nominal control branch requires finite recorded D20 targets")
        nominal_error = float(np.max(np.abs(target_array - reference)))
    else:
        if recorded_nominal_targets is not None:
            raise ValueError("non-nominal branch cannot receive nominal target references")
        nominal_error = 0.0
    return BranchRollout(
        source=source.identity,
        branch=continuation.spec,
        prefix=continuation.prefix,
        target_states=target_array,
        target_absorbing=np.asarray(absorbing, dtype=bool),
        target_handoff_state=np.asarray(handoff),
        target_outcome_status=np.asarray(outcomes),
        source_replay_max_abs=replay.maximum_physical_state_abs,
        source_fingerprint_match=fingerprint_match,
        nominal_future_valid=nominal,
        nominal_future_max_abs=nominal_error,
    )
