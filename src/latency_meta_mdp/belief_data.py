"""Model-neutral belief views over synchronized BELIEF-profile episodes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from latency_meta_mdp.artifacts import sha256_file
from latency_meta_mdp.temporal_contract import TemporalContract

FORMAL_TICK_US = 20_000
PHYSICS_DT_US = 2_000
ACTION_DIM = 7

_RAW_SCHEMA_VERSION = 3
_RAW_FORMAT_ID = "synchronized_episode_npz_v3"
_REQUIRED_ARTIFACTS = frozenset({"arrays.npz", "events.json", "metadata.json"})
_FORBIDDEN_DEBUG_ARRAYS = frozenset(
    {
        "actuator_ctrl",
        "applied_reference",
        "applied_reference_source_tick",
        "control_reference_valid",
        "eef_orientation_error_rotvec",
        "eef_position_error",
        "nullspace_joint_position_error",
    }
)


def _readonly(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.array(source[name], copy=True) for name in source.files}


def _require_shape(
    arrays: dict[str, np.ndarray],
    name: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    try:
        value = arrays[name]
    except KeyError as error:
        raise ValueError(f"BELIEF episode is missing {name}") from error
    if value.shape != shape:
        raise ValueError(f"BELIEF {name} has shape {value.shape}, expected {shape}")
    return value


@dataclass(frozen=True)
class BeliefDeploymentStream:
    boundary_tick: np.ndarray
    boundary_time_us: np.ndarray
    agentview_rgb: np.ndarray
    wrist_rgb: np.ndarray
    robot_qpos: np.ndarray
    robot_qvel: np.ndarray
    gripper_qpos: np.ndarray
    gripper_qvel: np.ndarray
    eef_position_world: np.ndarray
    eef_orientation_matrix_world: np.ndarray


@dataclass(frozen=True)
class BeliefSupervisionStream:
    object_pose: np.ndarray
    object_velocity: np.ndarray
    commanded_motion_position: np.ndarray
    commanded_motion_velocity: np.ndarray
    commanded_motion_acceleration: np.ndarray
    commanded_motion_segment_index: np.ndarray
    left_pad_contact: np.ndarray
    right_pad_contact: np.ndarray
    handoff_state: np.ndarray
    relative_geometry: np.ndarray
    boundary_outcome_status: np.ndarray


@dataclass(frozen=True)
class BeliefEpisodeView:
    episode_id: str
    task_id: str
    instruction: str
    level: int
    scene_seed: int
    record_profile: str
    deployment: BeliefDeploymentStream
    supervision: BeliefSupervisionStream
    transition_source_tick: np.ndarray
    transition_target_tick: np.ndarray
    expert_actions: np.ndarray
    action_mask: np.ndarray
    expert_phase: np.ndarray

    @property
    def boundary_count(self) -> int:
        return len(self.deployment.boundary_tick)

    @property
    def transition_count(self) -> int:
        return len(self.transition_source_tick)


@dataclass(frozen=True)
class BeliefSampleIndex:
    episode_id: str
    history_start_tick: int
    source_tick: int
    branch_delay_tick: int
    target_tick: int


@dataclass(frozen=True)
class BeliefLaunchHistory:
    boundary_tick: np.ndarray
    boundary_time_us: np.ndarray
    agentview_rgb: np.ndarray
    wrist_rgb: np.ndarray
    robot_qpos: np.ndarray
    robot_qvel: np.ndarray
    gripper_qpos: np.ndarray
    gripper_qvel: np.ndarray
    eef_position_world: np.ndarray
    eef_orientation_matrix_world: np.ndarray


@dataclass(frozen=True)
class BeliefReturnTarget:
    branch_delay_tick: int
    target_tick: int
    agentview_rgb: np.ndarray
    wrist_rgb: np.ndarray
    robot_qpos: np.ndarray
    robot_qvel: np.ndarray
    gripper_qpos: np.ndarray
    gripper_qvel: np.ndarray
    eef_position_world: np.ndarray
    eef_orientation_matrix_world: np.ndarray
    object_pose: np.ndarray
    object_velocity: np.ndarray
    commanded_motion_position: np.ndarray
    commanded_motion_velocity: np.ndarray
    commanded_motion_acceleration: np.ndarray
    commanded_motion_segment_index: int
    left_pad_contact: bool
    right_pad_contact: bool
    handoff_state: str
    relative_geometry: np.ndarray
    outcome_status: str


class ActionBufferAdapter(Protocol):
    """Protocol boundary for teacher-forced deployment-buffer reconstruction."""

    def build(self, *, episode: BeliefEpisodeView, source_tick: int) -> Any: ...


@dataclass(frozen=True)
class SharpTeacherBuffer:
    protocol_id: str
    source_tick: int
    chunk_start_tick: int
    active_cursor: int
    full_chunk: np.ndarray
    remaining_actions: np.ndarray


class SharpTeacherBufferAdapter:
    """Reconstruct the H50 chunk state visible at the E25 launch boundary."""

    def __init__(self, temporal_contract: TemporalContract) -> None:
        if not isinstance(temporal_contract, TemporalContract):
            raise TypeError("temporal_contract must be a TemporalContract")
        self.temporal_contract = temporal_contract

    def build(
        self,
        *,
        episode: BeliefEpisodeView,
        source_tick: int,
    ) -> SharpTeacherBuffer:
        chunk_start, chunk_stop = self.temporal_contract.teacher_chunk_bounds(
            source_tick=source_tick,
            episode_action_count=episode.transition_count,
        )
        remaining_start, remaining_stop = (
            self.temporal_contract.teacher_remaining_bounds(
                source_tick=source_tick,
                episode_action_count=episode.transition_count,
            )
        )
        return SharpTeacherBuffer(
            protocol_id="sharp_return_time_chunk_v2",
            source_tick=source_tick,
            chunk_start_tick=chunk_start,
            active_cursor=self.temporal_contract.launch_trigger_horizon,
            full_chunk=_readonly(episode.expert_actions[chunk_start:chunk_stop]),
            remaining_actions=_readonly(
                episode.expert_actions[remaining_start:remaining_stop]
            ),
        )


def load_belief_episode(episode_dir: Path) -> BeliefEpisodeView:
    root = episode_dir.resolve()
    manifest = _load_json(root / "manifest.json")
    if (
        manifest.get("schema_version") != _RAW_SCHEMA_VERSION
        or manifest.get("format_id") != _RAW_FORMAT_ID
    ):
        raise ValueError("belief loading requires a synchronized raw-v3 episode")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != _REQUIRED_ARTIFACTS:
        raise ValueError("raw-v3 manifest has an invalid artifact inventory")
    for name, digest in artifacts.items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"raw-v3 artifact hash mismatch: {name}")

    metadata = _load_json(root / "metadata.json")
    if metadata.get("record_profile") != "belief":
        raise ValueError("belief loading requires the BELIEF record profile")
    if (
        metadata.get("formal_tick_us") != FORMAL_TICK_US
        or metadata.get("physics_dt_us") != PHYSICS_DT_US
    ):
        raise ValueError("BELIEF episode does not use the certified clock")
    boundary_count = manifest.get("boundary_count")
    transition_count = manifest.get("transition_count")
    if (
        isinstance(boundary_count, bool)
        or not isinstance(boundary_count, int)
        or isinstance(transition_count, bool)
        or not isinstance(transition_count, int)
        or boundary_count != transition_count + 1
        or transition_count <= 0
    ):
        raise ValueError("BELIEF episode has invalid boundary/transition counts")

    arrays = _load_arrays(root / "arrays.npz")
    forbidden = sorted(set(arrays) & _FORBIDDEN_DEBUG_ARRAYS)
    if forbidden:
        raise ValueError(f"BELIEF episode contains forbidden control-debug arrays: {forbidden}")
    boundary_tick = _require_shape(arrays, "boundary_formal_tick", (boundary_count,))
    boundary_time = _require_shape(arrays, "boundary_time_us", (boundary_count,))
    source_tick = _require_shape(arrays, "transition_source_tick", (transition_count,))
    target_tick = _require_shape(arrays, "transition_target_tick", (transition_count,))
    if not (
        np.array_equal(boundary_tick, np.arange(boundary_count))
        and np.array_equal(boundary_time, boundary_tick * FORMAL_TICK_US)
        and np.array_equal(source_tick, boundary_tick[:-1])
        and np.array_equal(target_tick, boundary_tick[1:])
    ):
        raise ValueError("BELIEF episode clocks are not contiguous and aligned")

    deployment = BeliefDeploymentStream(
        boundary_tick=_readonly(boundary_tick, dtype=np.int64),
        boundary_time_us=_readonly(boundary_time, dtype=np.int64),
        agentview_rgb=_readonly(
            _require_shape(
                arrays,
                "agentview_rgb",
                (boundary_count, *arrays["agentview_rgb"].shape[1:]),
            ),
            dtype=np.uint8,
        ),
        wrist_rgb=_readonly(
            _require_shape(
                arrays,
                "robot0_eye_in_hand_rgb",
                (boundary_count, *arrays["robot0_eye_in_hand_rgb"].shape[1:]),
            ),
            dtype=np.uint8,
        ),
        robot_qpos=_readonly(_require_shape(arrays, "robot_qpos", (boundary_count, 7))),
        robot_qvel=_readonly(_require_shape(arrays, "robot_qvel", (boundary_count, 7))),
        gripper_qpos=_readonly(_require_shape(arrays, "gripper_qpos", (boundary_count, 2))),
        gripper_qvel=_readonly(_require_shape(arrays, "gripper_qvel", (boundary_count, 2))),
        eef_position_world=_readonly(
            _require_shape(arrays, "eef_position_world", (boundary_count, 3))
        ),
        eef_orientation_matrix_world=_readonly(
            _require_shape(
                arrays,
                "eef_orientation_matrix_world",
                (boundary_count, 3, 3),
            )
        ),
    )
    if (
        deployment.agentview_rgb.ndim != 4
        or deployment.wrist_rgb.ndim != 4
        or deployment.agentview_rgb.shape[-1] != 3
        or deployment.wrist_rgb.shape[-1] != 3
    ):
        raise ValueError("BELIEF deployment cameras must be RGB boundary streams")

    supervision = BeliefSupervisionStream(
        object_pose=_readonly(_require_shape(arrays, "object_pose", (boundary_count, 7))),
        object_velocity=_readonly(
            _require_shape(arrays, "object_velocity", (boundary_count, 6))
        ),
        commanded_motion_position=_readonly(
            _require_shape(arrays, "commanded_motion_position", (boundary_count, 3))
        ),
        commanded_motion_velocity=_readonly(
            _require_shape(arrays, "commanded_motion_velocity", (boundary_count, 3))
        ),
        commanded_motion_acceleration=_readonly(
            _require_shape(arrays, "commanded_motion_acceleration", (boundary_count, 3))
        ),
        commanded_motion_segment_index=_readonly(
            _require_shape(arrays, "commanded_motion_segment_index", (boundary_count,)),
            dtype=np.int64,
        ),
        left_pad_contact=_readonly(
            _require_shape(arrays, "left_pad_contact", (boundary_count,)), dtype=np.bool_
        ),
        right_pad_contact=_readonly(
            _require_shape(arrays, "right_pad_contact", (boundary_count,)), dtype=np.bool_
        ),
        handoff_state=_readonly(_require_shape(arrays, "handoff_state", (boundary_count,))),
        relative_geometry=_readonly(
            _require_shape(arrays, "relative_geometry", (boundary_count, 3))
        ),
        boundary_outcome_status=_readonly(
            _require_shape(arrays, "boundary_outcome_status", (boundary_count,))
        ),
    )
    actions = _require_shape(arrays, "expert_action", (transition_count, ACTION_DIM))
    action_mask = _require_shape(arrays, "action_mask", (transition_count, ACTION_DIM))
    expert_phase = _require_shape(arrays, "expert_phase", (transition_count,))
    if not np.all(np.isfinite(actions)) or not np.all(action_mask):
        raise ValueError("BELIEF expert actions must be finite and fully valid")
    if not set(np.unique(expert_phase)) <= {
        "pregrasp",
        "approach",
        "close",
        "lift",
    }:
        raise ValueError("BELIEF expert phases are invalid")

    episode_id = metadata.get("episode_id")
    task_id = metadata.get("task_id")
    instruction = metadata.get("instruction")
    level = metadata.get("level")
    scene_seed = metadata.get("scene_seed")
    if (
        not isinstance(episode_id, str)
        or episode_id != manifest.get("episode_id")
        or not isinstance(task_id, str)
        or not isinstance(instruction, str)
        or not instruction.strip()
        or isinstance(level, bool)
        or not isinstance(level, int)
        or level not in (1, 2, 3)
        or isinstance(scene_seed, bool)
        or not isinstance(scene_seed, int)
        or scene_seed < 0
    ):
        raise ValueError("BELIEF episode identity is invalid")
    return BeliefEpisodeView(
        episode_id=episode_id,
        task_id=task_id,
        instruction=instruction.strip(),
        level=level,
        scene_seed=scene_seed,
        record_profile="belief",
        deployment=deployment,
        supervision=supervision,
        transition_source_tick=_readonly(source_tick, dtype=np.int64),
        transition_target_tick=_readonly(target_tick, dtype=np.int64),
        expert_actions=_readonly(actions, dtype=np.float32),
        action_mask=_readonly(action_mask, dtype=np.bool_),
        expert_phase=_readonly(expert_phase),
    )


def build_belief_sample_indices(
    *,
    episode: BeliefEpisodeView,
    temporal_contract: TemporalContract,
    delay_ticks: tuple[int, ...],
) -> tuple[BeliefSampleIndex, ...]:
    if not isinstance(temporal_contract, TemporalContract):
        raise TypeError("temporal_contract must be a TemporalContract")
    if (
        not delay_ticks
        or len(set(delay_ticks)) != len(delay_ticks)
        or any(
            isinstance(delay, bool)
            or not isinstance(delay, int)
            or not 1 <= delay <= temporal_contract.maximum_delay_ticks
            for delay in delay_ticks
        )
    ):
        raise ValueError("delay_ticks must be unique ticks within the temporal contract")
    source_interval = temporal_contract.belief_source_interval(
        episode_action_count=episode.transition_count
    )
    indices = []
    for source_tick in range(source_interval.minimum, source_interval.maximum + 1):
        for delay_tick in delay_ticks:
            target_tick = source_tick + delay_tick
            indices.append(
                BeliefSampleIndex(
                    episode_id=episode.episode_id,
                    history_start_tick=(
                        source_tick - temporal_contract.history_sample_count + 1
                    ),
                    source_tick=source_tick,
                    branch_delay_tick=delay_tick,
                    target_tick=target_tick,
                )
            )
    return tuple(indices)


def _validate_index(*, episode: BeliefEpisodeView, index: BeliefSampleIndex) -> None:
    if index.episode_id != episode.episode_id:
        raise ValueError("belief index episode identity does not match")
    if not (
        0 <= index.history_start_tick <= index.source_tick < index.target_tick
        and index.target_tick <= episode.transition_count
        and index.target_tick - index.source_tick == index.branch_delay_tick
    ):
        raise ValueError("belief index is outside the episode contract")


def build_launch_history(
    *, episode: BeliefEpisodeView, index: BeliefSampleIndex
) -> BeliefLaunchHistory:
    _validate_index(episode=episode, index=index)
    selection = slice(index.history_start_tick, index.source_tick + 1)
    source = episode.deployment
    return BeliefLaunchHistory(
        boundary_tick=_readonly(source.boundary_tick[selection]),
        boundary_time_us=_readonly(source.boundary_time_us[selection]),
        agentview_rgb=_readonly(source.agentview_rgb[selection]),
        wrist_rgb=_readonly(source.wrist_rgb[selection]),
        robot_qpos=_readonly(source.robot_qpos[selection]),
        robot_qvel=_readonly(source.robot_qvel[selection]),
        gripper_qpos=_readonly(source.gripper_qpos[selection]),
        gripper_qvel=_readonly(source.gripper_qvel[selection]),
        eef_position_world=_readonly(source.eef_position_world[selection]),
        eef_orientation_matrix_world=_readonly(
            source.eef_orientation_matrix_world[selection]
        ),
    )


def build_return_target(
    *, episode: BeliefEpisodeView, index: BeliefSampleIndex
) -> BeliefReturnTarget:
    _validate_index(episode=episode, index=index)
    tick = index.target_tick
    deployment = episode.deployment
    supervision = episode.supervision
    return BeliefReturnTarget(
        branch_delay_tick=index.branch_delay_tick,
        target_tick=tick,
        agentview_rgb=_readonly(deployment.agentview_rgb[tick]),
        wrist_rgb=_readonly(deployment.wrist_rgb[tick]),
        robot_qpos=_readonly(deployment.robot_qpos[tick]),
        robot_qvel=_readonly(deployment.robot_qvel[tick]),
        gripper_qpos=_readonly(deployment.gripper_qpos[tick]),
        gripper_qvel=_readonly(deployment.gripper_qvel[tick]),
        eef_position_world=_readonly(deployment.eef_position_world[tick]),
        eef_orientation_matrix_world=_readonly(
            deployment.eef_orientation_matrix_world[tick]
        ),
        object_pose=_readonly(supervision.object_pose[tick]),
        object_velocity=_readonly(supervision.object_velocity[tick]),
        commanded_motion_position=_readonly(supervision.commanded_motion_position[tick]),
        commanded_motion_velocity=_readonly(supervision.commanded_motion_velocity[tick]),
        commanded_motion_acceleration=_readonly(
            supervision.commanded_motion_acceleration[tick]
        ),
        commanded_motion_segment_index=int(
            supervision.commanded_motion_segment_index[tick]
        ),
        left_pad_contact=bool(supervision.left_pad_contact[tick]),
        right_pad_contact=bool(supervision.right_pad_contact[tick]),
        handoff_state=str(supervision.handoff_state[tick]),
        relative_geometry=_readonly(supervision.relative_geometry[tick]),
        outcome_status=str(supervision.boundary_outcome_status[tick]),
    )


def teacher_forced_pre_return_actions(
    *, episode: BeliefEpisodeView, index: BeliefSampleIndex
) -> np.ndarray:
    _validate_index(episode=episode, index=index)
    return _readonly(episode.expert_actions[index.source_tick : index.target_tick])
