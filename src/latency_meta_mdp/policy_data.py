"""Model-neutral policy views derived from synchronized raw episode artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from latency_meta_mdp.artifacts import sha256_file
from latency_meta_mdp.temporal_contract import load_temporal_contract

FORMAL_TICK_US = 20_000
_TEMPORAL_CONTRACT = load_temporal_contract(
    Path(__file__).resolve().parents[2]
    / "configs/temporal/h50_e25_d20_k6_v1.yaml"
)
ACTION_HORIZON = _TEMPORAL_CONTRACT.prediction_horizon
LAUNCH_TRIGGER_HORIZON = _TEMPORAL_CONTRACT.launch_trigger_horizon
TEMPORAL_CONTRACT_ID = _TEMPORAL_CONTRACT.contract_id
POLICY_STATE_DIM = 8
ACTION_DIM = 7
ACTION_CONTRACT_ID = "panda_osc_pose_delta_v1"

_RAW_SCHEMA_VERSION = 3
_RAW_FORMAT_ID = "synchronized_episode_npz_v3"
_REQUIRED_ARTIFACTS = frozenset({"arrays.npz", "events.json", "metadata.json"})


def _readonly(value: Any, *, dtype: np.dtype[Any] | type | None = None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _rotation_matrix(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("EEF orientation must be a finite 3x3 rotation matrix")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(matrix), 1.0, atol=1e-6
    ):
        raise ValueError("EEF orientation must be a proper rotation matrix")
    return matrix


def rotation_matrix_to_rotvec(value: Any) -> np.ndarray:
    """Convert a rotation matrix to a deterministic shortest axis-angle vector."""

    matrix = _rotation_matrix(value)
    m00, m01, m02 = matrix[0]
    m10, m11, m12 = matrix[1]
    m20, m21, m22 = matrix[2]
    quaternion_matrix = np.asarray(
        [
            [m00 - m11 - m22, 0.0, 0.0, 0.0],
            [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
            [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
            [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        ],
        dtype=np.float64,
    )
    quaternion_matrix /= 3.0
    _, eigenvectors = np.linalg.eigh(quaternion_matrix)
    quaternion_wxyz = eigenvectors[[3, 0, 1, 2], -1]

    if quaternion_wxyz[0] < 0.0:
        quaternion_wxyz = -quaternion_wxyz
    elif abs(quaternion_wxyz[0]) <= 1e-12:
        vector = quaternion_wxyz[1:]
        nonzero = np.flatnonzero(np.abs(vector) > 1e-12)
        if nonzero.size and vector[nonzero[0]] < 0.0:
            quaternion_wxyz = -quaternion_wxyz

    vector = quaternion_wxyz[1:]
    sine_half_angle = float(np.linalg.norm(vector))
    if sine_half_angle <= 1e-12:
        return _readonly(np.zeros(3, dtype=np.float64))
    angle = 2.0 * np.arctan2(sine_half_angle, quaternion_wxyz[0])
    return _readonly(vector * (angle / sine_half_angle))


@dataclass(frozen=True)
class PolicyEpisode:
    episode_id: str
    task_id: str
    instruction: str
    level: int
    action_horizon: int
    source_formal_tick: np.ndarray
    source_time_us: np.ndarray
    agentview_rgb: np.ndarray
    wrist_rgb: np.ndarray
    state: np.ndarray
    actions: np.ndarray
    valid_action_chunk_sources: np.ndarray


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _validate_manifest(episode_dir: Path) -> dict[str, Any]:
    manifest = _load_json_object(episode_dir / "manifest.json")
    if (
        manifest.get("schema_version") != _RAW_SCHEMA_VERSION
        or manifest.get("format_id") != _RAW_FORMAT_ID
    ):
        raise ValueError("policy conversion requires a synchronized raw-v3 episode")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != _REQUIRED_ARTIFACTS:
        raise ValueError("raw-v3 manifest has an invalid artifact inventory")
    for name, expected_hash in artifacts.items():
        if sha256_file(episode_dir / name) != expected_hash:
            raise ValueError(f"raw-v3 artifact hash mismatch: {name}")
    return manifest


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.array(source[name], copy=True) for name in source.files}


def _require_shape(arrays: dict[str, np.ndarray], name: str, shape: tuple[int, ...]) -> np.ndarray:
    try:
        value = arrays[name]
    except KeyError as exc:
        raise ValueError(f"raw-v3 episode is missing {name}") from exc
    if value.shape != shape:
        raise ValueError(f"raw-v3 {name} has shape {value.shape}, expected {shape}")
    return value


def load_policy_episode(
    episode_dir: Path,
    *,
    action_horizon: int = ACTION_HORIZON,
) -> PolicyEpisode:
    """Load one raw-v3 episode without exposing simulator-privileged fields."""

    root = episode_dir.resolve()
    manifest = _validate_manifest(root)
    metadata = _load_json_object(root / "metadata.json")
    if metadata.get("formal_tick_us") != FORMAL_TICK_US:
        raise ValueError("policy conversion requires the 20 ms formal clock")
    if metadata.get("action_contract_id") != ACTION_CONTRACT_ID:
        raise ValueError("policy conversion requires the canonical delta-EEF action contract")
    if metadata.get("action_dim") != ACTION_DIM:
        raise ValueError("policy conversion requires 7D actions")
    if (
        isinstance(action_horizon, bool)
        or not isinstance(action_horizon, int)
        or action_horizon <= 0
    ):
        raise ValueError("action_horizon must be a positive integer")

    boundary_count = manifest.get("boundary_count")
    transition_count = manifest.get("transition_count")
    if (
        isinstance(boundary_count, bool)
        or not isinstance(boundary_count, int)
        or isinstance(transition_count, bool)
        or not isinstance(transition_count, int)
        or boundary_count != transition_count + 1
        or transition_count < action_horizon
    ):
        raise ValueError("raw-v3 episode cannot provide a complete action horizon")

    arrays = _load_arrays(root / "arrays.npz")
    boundary_tick = _require_shape(
        arrays, "boundary_formal_tick", (boundary_count,)
    ).astype(np.int64, copy=False)
    boundary_time = _require_shape(
        arrays, "boundary_time_us", (boundary_count,)
    ).astype(np.int64, copy=False)
    source_tick = _require_shape(
        arrays, "transition_source_tick", (transition_count,)
    ).astype(np.int64, copy=False)
    target_tick = _require_shape(
        arrays, "transition_target_tick", (transition_count,)
    ).astype(np.int64, copy=False)
    if not (
        np.array_equal(boundary_tick, np.arange(boundary_count, dtype=np.int64))
        and np.array_equal(boundary_time, boundary_tick * FORMAL_TICK_US)
        and np.array_equal(source_tick, boundary_tick[:-1])
        and np.array_equal(target_tick, boundary_tick[1:])
    ):
        raise ValueError("raw-v3 transition and boundary clocks are not exactly aligned")

    try:
        agentview = arrays["agentview_rgb"]
        wrist = arrays["robot0_eye_in_hand_rgb"]
    except KeyError as exc:
        raise ValueError("raw-v3 episode is missing a policy camera") from exc
    if (
        agentview.dtype != np.uint8
        or wrist.dtype != np.uint8
        or agentview.ndim != 4
        or wrist.ndim != 4
        or agentview.shape[0] != boundary_count
        or wrist.shape[0] != boundary_count
        or agentview.shape[-1] != 3
        or wrist.shape[-1] != 3
    ):
        raise ValueError("policy cameras must be RGB uint8 boundary streams")

    eef_position = _require_shape(
        arrays, "eef_position_world", (boundary_count, 3)
    )
    eef_orientation = _require_shape(
        arrays, "eef_orientation_matrix_world", (boundary_count, 3, 3)
    )
    gripper_qpos = _require_shape(arrays, "gripper_qpos", (boundary_count, 2))
    actions = _require_shape(arrays, "expert_action", (transition_count, ACTION_DIM))
    action_mask = _require_shape(arrays, "action_mask", (transition_count, ACTION_DIM))
    if not np.all(action_mask):
        raise ValueError("SFT policy conversion requires every action dimension to be valid")

    rotation_vectors = np.stack(
        [rotation_matrix_to_rotvec(matrix) for matrix in eef_orientation[:-1]]
    )
    state = np.concatenate(
        [eef_position[:-1], rotation_vectors, gripper_qpos[:-1]], axis=-1
    )
    if state.shape != (transition_count, POLICY_STATE_DIM) or not np.all(np.isfinite(state)):
        raise ValueError("derived policy state is not a finite 8D transition stream")
    if not np.all(np.isfinite(actions)):
        raise ValueError("policy actions must be finite")

    instruction = metadata.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("policy conversion requires a non-empty instruction")
    episode_id = metadata.get("episode_id")
    task_id = metadata.get("task_id")
    level = metadata.get("level")
    if (
        not isinstance(episode_id, str)
        or not isinstance(task_id, str)
        or isinstance(level, bool)
        or not isinstance(level, int)
        or episode_id != manifest.get("episode_id")
    ):
        raise ValueError("raw-v3 episode identity is invalid")

    return PolicyEpisode(
        episode_id=episode_id,
        task_id=task_id,
        instruction=instruction.strip(),
        level=level,
        action_horizon=action_horizon,
        source_formal_tick=_readonly(source_tick, dtype=np.int64),
        source_time_us=_readonly(boundary_time[:-1], dtype=np.int64),
        agentview_rgb=_readonly(agentview[:-1], dtype=np.uint8),
        wrist_rgb=_readonly(wrist[:-1], dtype=np.uint8),
        state=_readonly(state, dtype=np.float32),
        actions=_readonly(actions, dtype=np.float32),
        valid_action_chunk_sources=_readonly(
            np.arange(transition_count - action_horizon + 1), dtype=np.int64
        ),
    )
