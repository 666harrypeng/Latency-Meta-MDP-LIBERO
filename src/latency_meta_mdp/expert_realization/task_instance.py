"""Deterministic Task 4 task-instance materialization and boundary-0 identity."""

from __future__ import annotations

import hashlib
import io
import json
import struct
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import yaml

from latency_meta_mdp.backend import FormalStepExecutor, RoboSuitePlant
from latency_meta_mdp.control import ActionContract, load_action_contract
from latency_meta_mdp.expert_realization.config import (
    PilotConfig,
    StructuredExpertConfig,
    load_pilot_config,
    load_structured_expert_config,
)
from latency_meta_mdp.expert_realization.contracts import TaskInstanceId
from latency_meta_mdp.handoff import (
    HandoffAwareBallWorld,
    OneWayHandoff,
    PandaBallContactDetector,
)
from latency_meta_mdp.motion import (
    DrivenBallWorld,
    MotionConfig,
    MotionProfile,
    build_motion_profile,
    load_motion_config,
    motion_profile_from_mapping,
)
from latency_meta_mdp.outcomes import EpisodeOutcomeTracker, OutcomeCriteria
from latency_meta_mdp.snapshots import BoundarySnapshot, BoundarySnapshotter
from latency_meta_mdp.task import TaskSpec, load_task_spec, make_dynamic_grasp_lift_environment
from latency_meta_mdp.timing import ClockLedger

_CAMERAS = ("agentview", "robot0_eye_in_hand")
_EXPECTED_RUNTIME = {
    "runtime_version": "robosuite_native_v1",
    "physics_dt_us": 2_000,
    "formal_tick_us": 20_000,
    "camera_stride_ticks": 1,
    "compatibility_stride_ticks": 5,
    "control_freq_hz": 50,
    "lite_physics": True,
}
_SHA_FIELDS = (
    "shared_actions_sha256",
    "k6_camera_sha256",
    "k6_proprio_sha256",
    "planning_start_sha256",
)

_environment_factory: Callable[..., Any] = make_dynamic_grasp_lift_environment
_snapshotter_factory: Callable[..., Any] = BoundarySnapshotter
_profile_builder: Callable[..., MotionProfile] = build_motion_profile


class TaskInstanceReplayMismatch(RuntimeError):
    """An exact same-host replay diverged before family-specific control."""

    def __init__(self, *, field: str, detail: str = "bitwise replay mismatch") -> None:
        self.field = field
        self.detail = detail
        super().__init__(f"Task instance replay mismatch at {field}: {detail}")


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _readonly_exact(
    value: Any, *, dtype: np.dtype[Any], shape: tuple[int | None, ...], name: str
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if array.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(array.shape, shape, strict=True)
    ):
        raise ValueError(f"{name} has invalid shape {array.shape}")
    copied = np.array(array, copy=True, order="C")
    copied.setflags(write=False)
    return copied


_INITIAL_ARRAY_CONTRACT: dict[str, tuple[np.dtype[Any], tuple[int | None, ...]]] = {
    "mujoco_qpos": (np.dtype(np.float64), (None,)),
    "mujoco_qvel": (np.dtype(np.float64), (None,)),
    "mujoco_act": (np.dtype(np.float64), (None,)),
    "mujoco_ctrl": (np.dtype(np.float64), (9,)),
    "sim_time_seconds": (np.dtype(np.float64), ()),
    "robot_qpos": (np.dtype(np.float64), (7,)),
    "robot_qvel": (np.dtype(np.float64), (7,)),
    "gripper_qpos": (np.dtype(np.float64), (2,)),
    "gripper_qvel": (np.dtype(np.float64), (2,)),
    "object_qpos": (np.dtype(np.float64), (7,)),
    "object_qvel": (np.dtype(np.float64), (6,)),
    "eef_position_world": (np.dtype(np.float64), (3,)),
    "eef_orientation_matrix_world": (np.dtype(np.float64), (3, 3)),
    "osc_origin_position_world": (np.dtype(np.float64), (3,)),
    "osc_origin_orientation_world": (np.dtype(np.float64), (3, 3)),
    "osc_goal_position_base": (np.dtype(np.float64), (3,)),
    "osc_goal_orientation_matrix_base": (np.dtype(np.float64), (3, 3)),
    "clock_physics_step": (np.dtype(np.int64), ()),
    "clock_formal_tick": (np.dtype(np.int64), ()),
    "clock_time_us": (np.dtype(np.int64), ()),
    "clock_physics_dt_us": (np.dtype(np.int64), ()),
    "clock_formal_tick_us": (np.dtype(np.int64), ()),
    "handoff_state_json_utf8": (np.dtype(np.uint8), (None,)),
    "outcome_state_json_utf8": (np.dtype(np.uint8), (None,)),
    "agentview_rgb_sha256": (np.dtype(np.uint8), (32,)),
    "robot0_eye_in_hand_rgb_sha256": (np.dtype(np.uint8), (32,)),
    "motion_time_origin_us": (np.dtype(np.int64), ()),
}


@dataclass(frozen=True)
class InitialStateSnapshot:
    mujoco_qpos: np.ndarray
    mujoco_qvel: np.ndarray
    mujoco_act: np.ndarray
    mujoco_ctrl: np.ndarray
    sim_time_seconds: np.ndarray
    robot_qpos: np.ndarray
    robot_qvel: np.ndarray
    gripper_qpos: np.ndarray
    gripper_qvel: np.ndarray
    object_qpos: np.ndarray
    object_qvel: np.ndarray
    eef_position_world: np.ndarray
    eef_orientation_matrix_world: np.ndarray
    osc_origin_position_world: np.ndarray
    osc_origin_orientation_world: np.ndarray
    osc_goal_position_base: np.ndarray
    osc_goal_orientation_matrix_base: np.ndarray
    clock_physics_step: np.ndarray
    clock_formal_tick: np.ndarray
    clock_time_us: np.ndarray
    clock_physics_dt_us: np.ndarray
    clock_formal_tick_us: np.ndarray
    handoff_state_json_utf8: np.ndarray
    outcome_state_json_utf8: np.ndarray
    agentview_rgb_sha256: np.ndarray
    robot0_eye_in_hand_rgb_sha256: np.ndarray
    motion_time_origin_us: np.ndarray

    def __post_init__(self) -> None:
        for name, (dtype, shape) in _INITIAL_ARRAY_CONTRACT.items():
            object.__setattr__(
                self,
                name,
                _readonly_exact(getattr(self, name), dtype=dtype, shape=shape, name=name),
            )
        locked_scalars = {
            "sim_time_seconds": 0.0,
            "clock_physics_step": 0,
            "clock_formal_tick": 0,
            "clock_time_us": 0,
            "clock_physics_dt_us": 2_000,
            "clock_formal_tick_us": 20_000,
            "motion_time_origin_us": 0,
        }
        for name, expected in locked_scalars.items():
            if getattr(self, name).item() != expected:
                raise ValueError(f"{name} must equal {expected}")
        for name in ("handoff_state_json_utf8", "outcome_state_json_utf8"):
            try:
                json.loads(bytes(getattr(self, name)).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"{name} must contain UTF-8 JSON") from error


def _npy_bytes(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.lib.format.write_array(output, array, version=(2, 0), allow_pickle=False)
    return output.getvalue()


def encode_initial_state_npz(state: InitialStateSnapshot) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output, mode="w", compression=zipfile.ZIP_STORED, strict_timestamps=True
    ) as archive:
        for name in sorted(_INITIAL_ARRAY_CONTRACT):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, _npy_bytes(getattr(state, name)))
    return output.getvalue()


def decode_initial_state_npz(payload: bytes) -> InitialStateSnapshot:
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if set(archive.files) != set(_INITIAL_ARRAY_CONTRACT):
            raise ValueError("initial-state NPZ member inventory is invalid")
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    return InitialStateSnapshot(**arrays)


def _freeze_json(value: Any) -> Any:
    if type(value) is dict:
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if type(value) is list:
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _frame_array(name: str, value: Any) -> bytes:
    name_bytes = name.encode("utf-8")
    array = np.ascontiguousarray(np.asarray(value))
    dtype_bytes = array.dtype.str.encode("ascii")
    raw = array.tobytes(order="C")
    output = bytearray()
    output += struct.pack(">Q", len(name_bytes)) + name_bytes
    output += struct.pack(">Q", len(dtype_bytes)) + dtype_bytes
    output += struct.pack(">Q", array.ndim)
    for size in array.shape:
        output += struct.pack(">Q", size)
    output += struct.pack(">Q", len(raw)) + raw
    return bytes(output)


def _framed_arrays_sha256(*items: tuple[str, Any], json_values: tuple[Any, ...] = ()) -> str:
    digest = hashlib.sha256()
    for name, value in items:
        digest.update(_frame_array(name, value))
    for value in json_values:
        payload = _canonical_json_bytes(value)
        digest.update(struct.pack(">Q", len(payload)))
        digest.update(payload)
    return digest.hexdigest()


@dataclass(frozen=True)
class _MaterializedMotion:
    config: MotionConfig
    profile: MotionProfile
    motion_profile_mapping: Mapping[str, Any]
    motion_profile_bytes: bytes
    shared_endpoints_xy: np.ndarray
    shared_endpoint_bytes: bytes


def _materialize_motion_profile(
    *, project_root: Path, level: int, task_instance_seed: int
) -> _MaterializedMotion:
    if type(level) is not int or level not in (1, 2, 3):
        raise ValueError("level must be one of 1, 2, or 3")
    if type(task_instance_seed) is not int or task_instance_seed < 0:
        raise ValueError("task_instance_seed must be a non-negative integer")
    root = project_root.resolve()
    task_spec = load_task_spec(root / "configs/task/dynamic_grasp_lift_l0.yaml")
    config = load_motion_config(root / f"configs/motion/dynamic_grasp_lift_l{level}.yaml")
    generated = _profile_builder(
        config=config,
        seed=task_instance_seed,
        workspace_z=task_spec.ball_initial_position[2],
    )
    mapping = generated.to_mapping()
    payload = _canonical_json_bytes(mapping)
    parsed_mapping = json.loads(payload)
    profile = motion_profile_from_mapping(parsed_mapping, config=config)
    endpoints = np.stack(
        [profile.sample(0).position[:2], profile.sample(config.anchor_time_us).position[:2]]
    ).astype(np.float64, copy=False)
    endpoints = _readonly_exact(
        endpoints, dtype=np.dtype(np.float64), shape=(2, 2), name="shared_endpoints_xy"
    )
    endpoint_payload = _canonical_json_bytes({"shared_endpoints_xy": endpoints.tolist()})
    return _MaterializedMotion(
        config=config,
        profile=profile,
        motion_profile_mapping=_freeze_json(parsed_mapping),
        motion_profile_bytes=payload,
        shared_endpoints_xy=endpoints,
        shared_endpoint_bytes=endpoint_payload,
    )


@dataclass(frozen=True)
class MaterializedTaskInstance:
    project_root: Path
    task_instance_id: TaskInstanceId
    task_id: str
    instruction: str
    decision_source_tick: int
    shared_prefix_policy: str
    camera_height: int
    camera_width: int
    motion_profile_mapping: Mapping[str, Any]
    motion_profile_bytes: bytes
    shared_endpoints_xy: np.ndarray
    shared_endpoint_bytes: bytes
    initial_state: InitialStateSnapshot
    initial_state_bytes: bytes
    expected_anchor: Any
    expected_anchor_fingerprints: Mapping[str, str]
    task_config_sha256: str
    motion_config_sha256: str
    runtime_config_sha256: str
    controller_config_sha256: str

    def __post_init__(self) -> None:
        root = self.project_root.resolve()
        if root != self.project_root:
            raise ValueError("project_root must be resolved")
        if (
            self.task_id != "dynamic_grasp_lift"
            or self.instruction != "Grasp the moving ball and lift it."
        ):
            raise ValueError("unsupported moving-ball task identity")
        if self.decision_source_tick != 5 or self.shared_prefix_policy != "settle_open_hold_v1":
            raise ValueError("unsupported shared-prefix contract")
        if self.camera_height != 256 or self.camera_width != 256:
            raise ValueError("Task 4 cameras must be 256x256")
        object.__setattr__(
            self,
            "shared_endpoints_xy",
            _readonly_exact(
                self.shared_endpoints_xy,
                dtype=np.dtype(np.float64),
                shape=(2, 2),
                name="shared_endpoints_xy",
            ),
        )
        if set(self.expected_anchor_fingerprints) != set(_SHA_FIELDS):
            raise ValueError("expected anchor fingerprints are incomplete")
        for name, digest in {
            **dict(self.expected_anchor_fingerprints),
            "task_config_sha256": self.task_config_sha256,
            "motion_config_sha256": self.motion_config_sha256,
            "runtime_config_sha256": self.runtime_config_sha256,
            "controller_config_sha256": self.controller_config_sha256,
        }.items():
            if type(digest) is not str or len(digest) != 64 or digest != digest.lower():
                raise ValueError(f"{name} must be a lowercase SHA-256")
        object.__setattr__(
            self, "motion_profile_mapping", _freeze_json(_thaw_json(self.motion_profile_mapping))
        )
        object.__setattr__(
            self,
            "expected_anchor_fingerprints",
            MappingProxyType(dict(self.expected_anchor_fingerprints)),
        )
        self.validate_publication_consistency()

    def validate_publication_consistency(self) -> None:
        """Bind detached values to the exact bytes and hashes published by Task 3."""
        if not isinstance(self.task_instance_id, TaskInstanceId):
            raise TypeError("task_instance_id must be a TaskInstanceId")
        for name in ("motion_profile_bytes", "initial_state_bytes", "shared_endpoint_bytes"):
            if type(getattr(self, name)) is not bytes:
                raise TaskInstanceReplayMismatch(
                    field=name, detail="publication value is not bytes"
                )
        if _sha256_bytes(self.motion_profile_bytes) != self.task_instance_id.motion_profile_sha256:
            raise TaskInstanceReplayMismatch(field="motion_profile_bytes")
        try:
            decoded_motion = json.loads(self.motion_profile_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise TaskInstanceReplayMismatch(
                field="motion_profile_bytes", detail="invalid canonical JSON"
            ) from error
        if decoded_motion != _thaw_json(self.motion_profile_mapping):
            raise TaskInstanceReplayMismatch(field="motion_profile_mapping")
        if _canonical_json_bytes(decoded_motion) != self.motion_profile_bytes:
            raise TaskInstanceReplayMismatch(field="motion_profile_bytes")

        if _sha256_bytes(self.initial_state_bytes) != self.task_instance_id.initial_state_sha256:
            raise TaskInstanceReplayMismatch(field="initial_state_bytes")
        try:
            decoded_initial = decode_initial_state_npz(self.initial_state_bytes)
        except (OSError, ValueError, TypeError) as error:
            raise TaskInstanceReplayMismatch(
                field="initial_state_bytes", detail="invalid deterministic NPZ"
            ) from error
        try:
            _compare_initial_states(self.initial_state, decoded_initial)
        except TaskInstanceReplayMismatch as error:
            raise TaskInstanceReplayMismatch(field="initial_state", detail=error.field) from error
        if encode_initial_state_npz(self.initial_state) != self.initial_state_bytes:
            raise TaskInstanceReplayMismatch(field="initial_state_bytes")

        try:
            endpoint_mapping = json.loads(self.shared_endpoint_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise TaskInstanceReplayMismatch(
                field="shared_endpoint_bytes", detail="invalid canonical JSON"
            ) from error
        if _canonical_json_bytes(endpoint_mapping) != self.shared_endpoint_bytes:
            raise TaskInstanceReplayMismatch(field="shared_endpoint_bytes")
        if endpoint_mapping != {"shared_endpoints_xy": self.shared_endpoints_xy.tolist()}:
            raise TaskInstanceReplayMismatch(field="shared_endpoints_xy")

        from latency_meta_mdp.expert_realization.shared_prefix import SharedPrefixAnchor

        if not isinstance(self.expected_anchor, SharedPrefixAnchor):
            raise TypeError("expected_anchor must be a SharedPrefixAnchor")
        for name in _SHA_FIELDS:
            if getattr(self.expected_anchor, name) != self.expected_anchor_fingerprints[name]:
                raise TaskInstanceReplayMismatch(field=f"expected_anchor.{name}")


@dataclass(frozen=True)
class _LoadedConfiguration:
    root: Path
    task_spec: TaskSpec
    motion_config: MotionConfig
    action_contract: ActionContract
    structured: StructuredExpertConfig
    pilot: PilotConfig
    hashes: Mapping[str, str]


def _load_configuration(project_root: Path, level: int) -> _LoadedConfiguration:
    root = project_root.resolve()
    paths = {
        "task": root / "configs/task/dynamic_grasp_lift_l0.yaml",
        "motion": root / f"configs/motion/dynamic_grasp_lift_l{level}.yaml",
        "runtime": root / "configs/runtime/robosuite_v1.yaml",
        "controller": root / "configs/control/panda_osc_pose_delta_v1.yaml",
        "structured": root / "configs/expert_realization/panda_ball_structured.yaml",
        "pilot": root / "configs/collection/panda_ball_structured_pilot.yaml",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Task 4 configuration is incomplete: {missing}")
    task_spec = load_task_spec(paths["task"])
    motion_config = load_motion_config(paths["motion"])
    action_contract = load_action_contract(paths["controller"])
    structured = load_structured_expert_config(paths["structured"])
    pilot = load_pilot_config(paths["pilot"])
    runtime_mapping = yaml.safe_load(paths["runtime"].read_text(encoding="utf-8"))
    if runtime_mapping != _EXPECTED_RUNTIME:
        raise ValueError("runtime config does not match the certified Task 4 clock")
    if (
        level not in pilot.levels
        or motion_config.level != level
        or action_contract.physics_dt_us != 2_000
        or action_contract.formal_tick_us != 20_000
        or structured.decision_source_tick != 5
        or structured.shared_prefix_policy != "settle_open_hold_v1"
        or pilot.camera_height != 256
        or pilot.camera_width != 256
    ):
        raise ValueError("Task 4 configs do not share the locked task/prefix/camera/clock contract")
    return _LoadedConfiguration(
        root=root,
        task_spec=task_spec,
        motion_config=motion_config,
        action_contract=action_contract,
        structured=structured,
        pilot=pilot,
        hashes=MappingProxyType(
            {
                f"{name}_config_sha256": _sha256_bytes(paths[name].read_bytes())
                for name in ("task", "motion", "runtime", "controller")
            }
        ),
    )


@dataclass
class _TaskInstanceRuntime:
    env: Any
    action_contract: Any
    tracker: Any
    handoff: Any
    world: Any
    executor: Any
    closed: bool = False

    def require_open(self) -> None:
        if self.closed:
            raise RuntimeError("task-instance runtime is closed")

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.env.close()


def _construct_runtime(
    *, config: _LoadedConfiguration, profile: MotionProfile, seed: int
) -> _TaskInstanceRuntime:
    env: Any | None = None
    try:
        env = _environment_factory(
            spec=config.task_spec,
            seed=seed,
            offscreen=True,
            controller_config=config.action_contract.to_robosuite_config(),
        )
        config.action_contract.verify_runtime(env)
        tracker = EpisodeOutcomeTracker(
            OutcomeCriteria(
                physics_dt_us=2_000,
                formal_tick_us=20_000,
                stable_grasp_dwell_us=40_000,
                lift_height_m=config.task_spec.lift_success_height_m,
                lift_dwell_us=100_000,
                grasp_deadline_us=config.motion_config.anchor_time_us,
                lift_timeout_us=10_000_000,
            )
        )
        handoff = OneWayHandoff(
            env=env,
            action_contract=config.action_contract,
            contact_detector=PandaBallContactDetector(env),
            outcome_tracker=tracker,
        )
        world = HandoffAwareBallWorld(
            driver=DrivenBallWorld(profile=profile, motion_level=config.motion_config.level),
            handoff=handoff,
        )
        executor = FormalStepExecutor(
            plant=RoboSuitePlant(
                env=env,
                snapshotter=_snapshotter_factory(
                    camera_names=_CAMERAS,
                    width=256,
                    height=256,
                ),
                world_writer=world,
                physics_point_observer=handoff.on_physics_point,
                control_observer=handoff.on_control_applied,
            ),
            ledger=ClockLedger(physics_dt_us=2_000, formal_tick_us=20_000),
        )
        return _TaskInstanceRuntime(
            env=env,
            action_contract=config.action_contract,
            tracker=tracker,
            handoff=handoff,
            world=world,
            executor=executor,
        )
    except BaseException:
        if env is not None:
            env.close()
        raise


def _build_task_instance_runtime(task_instance: MaterializedTaskInstance) -> _TaskInstanceRuntime:
    if not isinstance(task_instance, MaterializedTaskInstance):
        raise TypeError("task_instance must be a MaterializedTaskInstance")
    config = _load_configuration(task_instance.project_root, task_instance.task_instance_id.level)
    for name in ("task", "motion", "runtime", "controller"):
        field = f"{name}_config_sha256"
        if config.hashes[field] != getattr(task_instance, field):
            raise TaskInstanceReplayMismatch(field=field, detail="configuration bytes changed")
    parsed = motion_profile_from_mapping(
        _thaw_json(task_instance.motion_profile_mapping), config=config.motion_config
    )
    return _construct_runtime(
        config=config,
        profile=parsed,
        seed=task_instance.task_instance_id.task_instance_seed,
    )


def _json_uint8(value: Any) -> np.ndarray:
    return np.frombuffer(_canonical_json_bytes(value), dtype=np.uint8).copy()


def _validate_camera_sample_clock(
    boundary: BoundarySnapshot, camera_name: str, *, context: str
) -> None:
    if camera_name not in boundary.cameras:
        raise TaskInstanceReplayMismatch(field=f"{context}.{camera_name}.missing")
    camera = boundary.cameras[camera_name]
    if camera.name != camera_name:
        raise TaskInstanceReplayMismatch(field=f"{context}.{camera_name}.name")
    expected = {
        "source_physics_step": boundary.physics_step_index,
        "source_formal_tick": boundary.formal_tick_index,
        "source_time_us": boundary.time_us,
    }
    for field, value in expected.items():
        if getattr(camera, field) != value:
            raise TaskInstanceReplayMismatch(field=f"{context}.{camera_name}.{field}")


def _initial_state_from_runtime(
    runtime: _TaskInstanceRuntime, boundary: BoundarySnapshot
) -> InitialStateSnapshot:
    runtime.require_open()
    for camera_name in _CAMERAS:
        _validate_camera_sample_clock(boundary, camera_name, context="boundary0")
    arm = runtime.env.robots[0].part_controllers["right"]
    handoff_payload = runtime.handoff.fingerprint_payload()
    handoff_payload.pop("outcome", None)
    camera_hashes = {}
    for name in _CAMERAS:
        camera = boundary.cameras[name]
        camera_hashes[name] = bytes.fromhex(
            _framed_arrays_sha256(
                (f"{name}_rgb", camera.rgb),
                (
                    f"{name}_source_physics_step",
                    np.array(camera.source_physics_step, dtype=np.int64),
                ),
                (f"{name}_source_formal_tick", np.array(camera.source_formal_tick, dtype=np.int64)),
                (f"{name}_source_time_us", np.array(camera.source_time_us, dtype=np.int64)),
            )
        )
    return InitialStateSnapshot(
        mujoco_qpos=np.asarray(boundary.qpos, dtype=np.float64),
        mujoco_qvel=np.asarray(boundary.qvel, dtype=np.float64),
        mujoco_act=np.asarray(boundary.act, dtype=np.float64),
        mujoco_ctrl=np.asarray(boundary.actuator_ctrl, dtype=np.float64),
        sim_time_seconds=np.array(boundary.sim_time_seconds, dtype=np.float64),
        robot_qpos=np.asarray(boundary.robot_qpos, dtype=np.float64),
        robot_qvel=np.asarray(boundary.robot_qvel, dtype=np.float64),
        gripper_qpos=np.asarray(boundary.robot_gripper_qpos, dtype=np.float64),
        gripper_qvel=np.asarray(boundary.robot_gripper_qvel, dtype=np.float64),
        object_qpos=np.asarray(boundary.object_qpos, dtype=np.float64),
        object_qvel=np.asarray(boundary.object_qvel, dtype=np.float64),
        eef_position_world=np.asarray(boundary.eef_pos, dtype=np.float64),
        eef_orientation_matrix_world=np.asarray(boundary.eef_xmat, dtype=np.float64),
        osc_origin_position_world=np.asarray(arm.origin_pos, dtype=np.float64),
        osc_origin_orientation_world=np.asarray(arm.origin_ori, dtype=np.float64),
        osc_goal_position_base=np.asarray(arm.goal_pos, dtype=np.float64),
        osc_goal_orientation_matrix_base=np.asarray(arm.goal_ori, dtype=np.float64),
        clock_physics_step=np.array(runtime.executor.ledger.physics_step_index, dtype=np.int64),
        clock_formal_tick=np.array(runtime.executor.ledger.formal_tick_index, dtype=np.int64),
        clock_time_us=np.array(runtime.executor.ledger.time_us, dtype=np.int64),
        clock_physics_dt_us=np.array(runtime.executor.ledger.physics_dt_us, dtype=np.int64),
        clock_formal_tick_us=np.array(runtime.executor.ledger.formal_tick_us, dtype=np.int64),
        handoff_state_json_utf8=_json_uint8(handoff_payload),
        outcome_state_json_utf8=_json_uint8(runtime.tracker.fingerprint_payload()),
        agentview_rgb_sha256=np.frombuffer(camera_hashes["agentview"], dtype=np.uint8),
        robot0_eye_in_hand_rgb_sha256=np.frombuffer(
            camera_hashes["robot0_eye_in_hand"], dtype=np.uint8
        ),
        motion_time_origin_us=np.array(0, dtype=np.int64),
    )


def _compare_initial_states(expected: InitialStateSnapshot, actual: InitialStateSnapshot) -> None:
    for field in fields(InitialStateSnapshot):
        name = field.name
        if not np.array_equal(getattr(expected, name), getattr(actual, name)):
            raise TaskInstanceReplayMismatch(field=name)


def materialize_task_instance(
    *, project_root: Path, level: int, task_instance_seed: int
) -> MaterializedTaskInstance:
    config = _load_configuration(project_root, level)
    motion = _materialize_motion_profile(
        project_root=config.root, level=level, task_instance_seed=task_instance_seed
    )
    runtime = _construct_runtime(
        config=config,
        profile=motion.profile,
        seed=task_instance_seed,
    )
    try:
        boundary_zero = runtime.executor.initialize()
        initial_state = _initial_state_from_runtime(runtime, boundary_zero)
        initial_bytes = encode_initial_state_npz(initial_state)
        task_id = TaskInstanceId(
            level=level,
            task_instance_seed=task_instance_seed,
            motion_profile_sha256=_sha256_bytes(motion.motion_profile_bytes),
            initial_state_sha256=_sha256_bytes(initial_bytes),
        )
        from latency_meta_mdp.expert_realization.shared_prefix import (
            _execute_shared_prefix,
        )

        dry_anchor = _execute_shared_prefix(
            runtime=runtime,
            boundary_zero=boundary_zero,
            decision_source_tick=5,
            motion_profile_sha256=task_id.motion_profile_sha256,
            shared_endpoint_sha256=_sha256_bytes(motion.shared_endpoint_bytes),
        )
        return MaterializedTaskInstance(
            project_root=config.root,
            task_instance_id=task_id,
            task_id=config.task_spec.task_id,
            instruction="Grasp the moving ball and lift it.",
            decision_source_tick=5,
            shared_prefix_policy=config.structured.shared_prefix_policy,
            camera_height=config.pilot.camera_height,
            camera_width=config.pilot.camera_width,
            motion_profile_mapping=motion.motion_profile_mapping,
            motion_profile_bytes=motion.motion_profile_bytes,
            shared_endpoints_xy=motion.shared_endpoints_xy,
            shared_endpoint_bytes=motion.shared_endpoint_bytes,
            initial_state=initial_state,
            initial_state_bytes=initial_bytes,
            expected_anchor=dry_anchor,
            expected_anchor_fingerprints=MappingProxyType(
                {name: getattr(dry_anchor, name) for name in _SHA_FIELDS}
            ),
            task_config_sha256=config.hashes["task_config_sha256"],
            motion_config_sha256=config.hashes["motion_config_sha256"],
            runtime_config_sha256=config.hashes["runtime_config_sha256"],
            controller_config_sha256=config.hashes["controller_config_sha256"],
        )
    finally:
        runtime.close()
