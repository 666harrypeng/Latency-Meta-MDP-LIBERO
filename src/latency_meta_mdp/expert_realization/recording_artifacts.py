"""Durable NPZ publication and verified loading for structured expert episodes."""

from __future__ import annotations

import io
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from latency_meta_mdp.expert_realization.artifacts import (
    _check_final_root,
    _hash_bytes,
    _json_bytes,
    _load_json_bytes,
    _publish_tree,
    _reject_legacy_payload,
    _strict,
)
from latency_meta_mdp.expert_realization.contracts import (
    AttemptId,
    ExpertRealizationId,
    TaskInstanceId,
)
from latency_meta_mdp.expert_realization.recording_contracts import (
    StructuredBoundaryRecord,
    StructuredDeploymentRecord,
    StructuredEpisodeMetadata,
    StructuredExpertAuditRecord,
    StructuredPhysicalEventRecord,
    StructuredQualificationRecord,
    StructuredSynchronizedEpisode,
    StructuredTransitionRecord,
)

EPISODE_FORMAT = "structured_expert_episode_npz_v1"
_EPISODE_FILES = frozenset({"arrays.npz", "metadata.json", "events.json", "manifest.json"})
_OUTCOME_CODE = {"running": 0, "success": 1, "failure": 2}
_OUTCOME_VALUE = {value: key for key, value in _OUTCOME_CODE.items()}
_HANDOFF_CODE = {"driven": 0, "physical": 1, "failure": 2}
_HANDOFF_VALUE = {value: key for key, value in _HANDOFF_CODE.items()}
_ARRAY_NAMES = frozenset(
    {
        "boundary_formal_tick",
        "boundary_physics_step",
        "boundary_time_us",
        "agentview_rgb",
        "robot0_eye_in_hand_rgb",
        "robot_qpos",
        "robot_qvel",
        "gripper_qpos",
        "gripper_qvel",
        "eef_position_world",
        "eef_orientation_matrix_world",
        "boundary_outcome_status_code",
        "object_pose",
        "object_velocity",
        "commanded_motion_position",
        "commanded_motion_velocity",
        "commanded_motion_acceleration",
        "commanded_motion_segment_index",
        "left_pad_contact",
        "right_pad_contact",
        "handoff_state_code",
        "relative_geometry",
        "actuator_ctrl",
        "control_reference_valid",
        "applied_reference",
        "applied_reference_source_tick",
        "nullspace_joint_position_error",
        "eef_position_error",
        "eef_orientation_error_rotvec",
        "transition_source_tick",
        "transition_target_tick",
        "expert_action",
        "action_mask",
        "audit_source_physics_step",
        "audit_source_formal_tick",
        "audit_source_time_us",
        "audit_selected_reference_index",
        "audit_target_eef_position_world",
        "audit_target_eef_orientation_world",
        "audit_estimated_object_velocity",
    }
)


def _stack(records: tuple[Any, ...], name: str, *, dtype: Any) -> np.ndarray:
    return np.asarray([getattr(item, name) for item in records], dtype=dtype)


def _episode_arrays(episode: StructuredSynchronizedEpisode) -> dict[str, np.ndarray]:
    boundaries = episode.boundaries
    deployment = tuple(item.deployment for item in boundaries)
    qualification = tuple(item.qualification for item in boundaries)
    transitions = episode.transitions
    audits = tuple(item.expert_audit for item in transitions)
    boundary_count = len(boundaries)
    valid = np.asarray(
        [item.applied_reference is not None for item in qualification], dtype=np.bool_
    )
    optional = {
        "applied_reference": np.full((boundary_count, 7), np.nan, dtype=np.float64),
        "nullspace_joint_position_error": np.full((boundary_count, 7), np.nan, dtype=np.float64),
        "eef_position_error": np.full((boundary_count, 3), np.nan, dtype=np.float64),
        "eef_orientation_error_rotvec": np.full((boundary_count, 3), np.nan, dtype=np.float64),
    }
    source_ticks = np.full(boundary_count, -1, dtype=np.int64)
    for index, item in enumerate(qualification):
        if item.applied_reference is not None:
            for name in optional:
                optional[name][index] = getattr(item, name)
            source_ticks[index] = item.applied_reference_source_tick
    arrays = {
        "boundary_formal_tick": _stack(boundaries, "formal_tick_index", dtype=np.int64),
        "boundary_physics_step": _stack(boundaries, "physics_step_index", dtype=np.int64),
        "boundary_time_us": _stack(boundaries, "time_us", dtype=np.int64),
        "agentview_rgb": _stack(deployment, "agentview_rgb", dtype=np.uint8),
        "robot0_eye_in_hand_rgb": _stack(deployment, "robot0_eye_in_hand_rgb", dtype=np.uint8),
        "robot_qpos": _stack(deployment, "robot_qpos", dtype=np.float64),
        "robot_qvel": _stack(deployment, "robot_qvel", dtype=np.float64),
        "gripper_qpos": _stack(deployment, "gripper_qpos", dtype=np.float64),
        "gripper_qvel": _stack(deployment, "gripper_qvel", dtype=np.float64),
        "eef_position_world": _stack(deployment, "eef_position_world", dtype=np.float64),
        "eef_orientation_matrix_world": _stack(
            deployment, "eef_orientation_matrix_world", dtype=np.float64
        ),
        "boundary_outcome_status_code": np.asarray(
            [_OUTCOME_CODE[item.outcome_status] for item in boundaries], dtype=np.int8
        ),
        "object_pose": _stack(qualification, "object_pose", dtype=np.float64),
        "object_velocity": _stack(qualification, "object_velocity", dtype=np.float64),
        "commanded_motion_position": _stack(
            qualification, "commanded_motion_position", dtype=np.float64
        ),
        "commanded_motion_velocity": _stack(
            qualification, "commanded_motion_velocity", dtype=np.float64
        ),
        "commanded_motion_acceleration": _stack(
            qualification, "commanded_motion_acceleration", dtype=np.float64
        ),
        "commanded_motion_segment_index": _stack(
            qualification, "commanded_motion_segment_index", dtype=np.int64
        ),
        "left_pad_contact": _stack(qualification, "left_pad_contact", dtype=np.bool_),
        "right_pad_contact": _stack(qualification, "right_pad_contact", dtype=np.bool_),
        "handoff_state_code": np.asarray(
            [_HANDOFF_CODE[item.handoff_state] for item in qualification], dtype=np.int8
        ),
        "relative_geometry": _stack(qualification, "relative_geometry", dtype=np.float64),
        "actuator_ctrl": _stack(qualification, "actuator_ctrl", dtype=np.float64),
        "control_reference_valid": valid,
        "applied_reference": optional["applied_reference"],
        "applied_reference_source_tick": source_ticks,
        "nullspace_joint_position_error": optional["nullspace_joint_position_error"],
        "eef_position_error": optional["eef_position_error"],
        "eef_orientation_error_rotvec": optional["eef_orientation_error_rotvec"],
        "transition_source_tick": _stack(transitions, "source_formal_tick", dtype=np.int64),
        "transition_target_tick": _stack(transitions, "target_formal_tick", dtype=np.int64),
        "expert_action": _stack(transitions, "expert_action", dtype=np.float64),
        "action_mask": _stack(transitions, "action_mask", dtype=np.bool_),
        "audit_source_physics_step": _stack(audits, "source_physics_step", dtype=np.int64),
        "audit_source_formal_tick": _stack(audits, "source_formal_tick", dtype=np.int64),
        "audit_source_time_us": _stack(audits, "source_time_us", dtype=np.int64),
        "audit_selected_reference_index": np.asarray(
            [
                item.selected_reference_index if item.selected_reference_index is not None else -1
                for item in audits
            ],
            dtype=np.int64,
        ),
        "audit_target_eef_position_world": _stack(
            audits, "target_eef_position_world", dtype=np.float64
        ),
        "audit_target_eef_orientation_world": _stack(
            audits, "target_eef_orientation_matrix_world", dtype=np.float64
        ),
        "audit_estimated_object_velocity": _stack(
            audits, "estimated_object_velocity_world", dtype=np.float64
        ),
    }
    if not transitions:
        arrays.update(
            {
                "transition_source_tick": np.empty((0,), dtype=np.int64),
                "transition_target_tick": np.empty((0,), dtype=np.int64),
                "expert_action": np.empty((0, 7), dtype=np.float64),
                "action_mask": np.empty((0, 7), dtype=np.bool_),
                "audit_source_physics_step": np.empty((0,), dtype=np.int64),
                "audit_source_formal_tick": np.empty((0,), dtype=np.int64),
                "audit_source_time_us": np.empty((0,), dtype=np.int64),
                "audit_selected_reference_index": np.empty((0,), dtype=np.int64),
                "audit_target_eef_position_world": np.empty((0, 3), dtype=np.float64),
                "audit_target_eef_orientation_world": np.empty((0, 3, 3), dtype=np.float64),
                "audit_estimated_object_velocity": np.empty((0, 3), dtype=np.float64),
            }
        )
    return arrays


def _npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    np.savez(output, **arrays)
    return output.getvalue()


def write_structured_episode(episode: StructuredSynchronizedEpisode, target: Path) -> Path:
    if not isinstance(episode, StructuredSynchronizedEpisode):
        raise TypeError("episode must be a StructuredSynchronizedEpisode")
    arrays = _npz_bytes(_episode_arrays(episode))
    metadata = _json_bytes(
        {
            "schema_version": 1,
            "metadata": episode.metadata.to_mapping(),
            "terminal_status": episode.terminal_status,
            "terminal_reason": episode.terminal_reason,
            "audit_phase_id": [item.expert_audit.phase_id for item in episode.transitions],
            "audit_reference_kind": [
                item.expert_audit.reference_kind for item in episode.transitions
            ],
        }
    )
    events = _json_bytes(
        {
            "schema_version": 1,
            "episode_id": episode.metadata.episode_id,
            "attempt_id": episode.metadata.attempt_id.to_mapping(),
            "events": [item.to_mapping() for item in episode.physical_events],
        }
    )
    artifacts = {
        "arrays.npz": _hash_bytes(arrays),
        "metadata.json": _hash_bytes(metadata),
        "events.json": _hash_bytes(events),
    }
    manifest = _json_bytes(
        {
            "schema_version": 1,
            "format_id": EPISODE_FORMAT,
            "task_instance_id": episode.metadata.task_instance_id.to_mapping(),
            "expert_realization_id": episode.metadata.expert_realization_id.to_mapping(),
            "attempt_id": episode.metadata.attempt_id.to_mapping(),
            "record_profile": episode.metadata.record_profile,
            "boundary_count": len(episode.boundaries),
            "transition_count": len(episode.transitions),
            "event_count": len(episode.physical_events),
            "terminal_status": episode.terminal_status,
            "terminal_reason": episode.terminal_reason,
            "artifacts": artifacts,
        }
    )
    return _publish_tree(
        Path(target),
        {
            "arrays.npz": arrays,
            "metadata.json": metadata,
            "events.json": events,
            "manifest.json": manifest,
        },
        expected_hashes=artifacts,
    )


def _require_array(
    arrays: dict[str, np.ndarray],
    name: str,
    *,
    dtype: Any,
    shape: tuple[int, ...],
) -> np.ndarray:
    value = arrays[name]
    if value.dtype != np.dtype(dtype) or value.shape != shape:
        raise ValueError(f"{name} has wrong dtype or shape")
    return value


@dataclass(frozen=True)
class VerifiedStructuredEpisodeSnapshot:
    episode: StructuredSynchronizedEpisode
    files: MappingProxyType
    manifest_sha256: str


def _read_exact_episode_files(root: Path) -> dict[str, bytes]:
    root = _check_final_root(Path(root))
    if root.is_symlink():
        raise ValueError("episode root must be a regular final directory, not a symlink")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_fd = os.open(root, flags)
    except OSError as error:
        raise ValueError("episode root must be a regular final directory") from error
    descriptors: dict[str, int] = {}
    try:
        names = set(os.listdir(directory_fd))
        if names != _EPISODE_FILES:
            raise ValueError("episode filesystem inventory must contain exactly four files")
        for name in sorted(_EPISODE_FILES):
            try:
                descriptor = os.open(
                    name,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=directory_fd,
                )
            except OSError as error:
                raise ValueError(
                    f"episode child must be a regular non-symlink file: {name}"
                ) from error
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                os.close(descriptor)
                raise ValueError(f"episode child must be a regular non-symlink file: {name}")
            descriptors[name] = descriptor
        payloads: dict[str, bytes] = {}
        for name, descriptor in descriptors.items():
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            payloads[name] = b"".join(chunks)
        return payloads
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)
        os.close(directory_fd)


def _episode_from_payloads(payloads: dict[str, bytes]) -> StructuredSynchronizedEpisode:
    if set(payloads) != _EPISODE_FILES or any(
        type(value) is not bytes for value in payloads.values()
    ):
        raise ValueError("episode snapshot must contain exactly four byte payloads")
    raw_manifest = _load_json_bytes(payloads["manifest.json"], name="episode manifest")
    if raw_manifest.get("format_id") != EPISODE_FORMAT:
        raise ValueError("unsupported episode format")
    manifest = _strict(
        raw_manifest,
        {
            "schema_version",
            "format_id",
            "task_instance_id",
            "expert_realization_id",
            "attempt_id",
            "record_profile",
            "boundary_count",
            "transition_count",
            "event_count",
            "terminal_status",
            "terminal_reason",
            "artifacts",
        },
        name="episode manifest",
    )
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ValueError("unsupported episode format")
    if type(manifest["artifacts"]) is not dict or set(manifest["artifacts"]) != {
        "arrays.npz",
        "metadata.json",
        "events.json",
    }:
        raise ValueError("episode artifact inventory is invalid")
    for name, digest in manifest["artifacts"].items():
        if _hash_bytes(payloads[name]) != digest:
            raise ValueError(f"{name} hash mismatch")
        _reject_legacy_payload(name, payloads[name])
    metadata_envelope = _strict(
        _load_json_bytes(payloads["metadata.json"], name="metadata.json"),
        {
            "schema_version",
            "metadata",
            "terminal_status",
            "terminal_reason",
            "audit_phase_id",
            "audit_reference_kind",
        },
        name="metadata.json",
    )
    if (
        type(metadata_envelope["schema_version"]) is not int
        or metadata_envelope["schema_version"] != 1
    ):
        raise ValueError("metadata schema version is invalid")
    metadata = StructuredEpisodeMetadata.from_mapping(metadata_envelope["metadata"])
    task = TaskInstanceId.from_mapping(manifest["task_instance_id"])
    realization = ExpertRealizationId.from_mapping(
        manifest["expert_realization_id"],
        structured_expert_config_sha256=metadata.structured_expert_config_sha256,
    )
    attempt = AttemptId.from_mapping(
        manifest["attempt_id"],
        structured_expert_config_sha256=metadata.structured_expert_config_sha256,
    )
    if (task, realization, attempt) != (
        metadata.task_instance_id,
        metadata.expert_realization_id,
        metadata.attempt_id,
    ):
        raise ValueError("episode manifest identities do not match metadata")
    if manifest["record_profile"] != metadata.record_profile:
        raise ValueError("episode record profile mismatch")
    boundary_count = manifest["boundary_count"]
    transition_count = manifest["transition_count"]
    event_count = manifest["event_count"]
    if (
        type(boundary_count) is not int
        or type(transition_count) is not int
        or type(event_count) is not int
        or boundary_count < 1
        or transition_count < 0
        or event_count < 0
        or boundary_count != transition_count + 1
    ):
        raise ValueError("episode counts are invalid")
    try:
        with np.load(io.BytesIO(payloads["arrays.npz"]), allow_pickle=False) as source:
            if set(source.files) != _ARRAY_NAMES:
                raise ValueError("episode NPZ array set is invalid")
            arrays = {name: np.array(source[name], copy=True) for name in source.files}
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error) == "episode NPZ array set is invalid":
            raise
        raise ValueError("arrays.npz is invalid") from error
    height, width = metadata.camera_height, metadata.camera_width
    shapes = {
        "boundary_formal_tick": (boundary_count,),
        "boundary_physics_step": (boundary_count,),
        "boundary_time_us": (boundary_count,),
        "agentview_rgb": (boundary_count, height, width, 3),
        "robot0_eye_in_hand_rgb": (boundary_count, height, width, 3),
        "robot_qpos": (boundary_count, 7),
        "robot_qvel": (boundary_count, 7),
        "gripper_qpos": (boundary_count, 2),
        "gripper_qvel": (boundary_count, 2),
        "eef_position_world": (boundary_count, 3),
        "eef_orientation_matrix_world": (boundary_count, 3, 3),
        "boundary_outcome_status_code": (boundary_count,),
        "object_pose": (boundary_count, 7),
        "object_velocity": (boundary_count, 6),
        "commanded_motion_position": (boundary_count, 3),
        "commanded_motion_velocity": (boundary_count, 3),
        "commanded_motion_acceleration": (boundary_count, 3),
        "commanded_motion_segment_index": (boundary_count,),
        "left_pad_contact": (boundary_count,),
        "right_pad_contact": (boundary_count,),
        "handoff_state_code": (boundary_count,),
        "relative_geometry": (boundary_count, 3),
        "actuator_ctrl": (boundary_count, 9),
        "control_reference_valid": (boundary_count,),
        "applied_reference": (boundary_count, 7),
        "applied_reference_source_tick": (boundary_count,),
        "nullspace_joint_position_error": (boundary_count, 7),
        "eef_position_error": (boundary_count, 3),
        "eef_orientation_error_rotvec": (boundary_count, 3),
        "transition_source_tick": (transition_count,),
        "transition_target_tick": (transition_count,),
        "expert_action": (transition_count, 7),
        "action_mask": (transition_count, 7),
        "audit_source_physics_step": (transition_count,),
        "audit_source_formal_tick": (transition_count,),
        "audit_source_time_us": (transition_count,),
        "audit_selected_reference_index": (transition_count,),
        "audit_target_eef_position_world": (transition_count, 3),
        "audit_target_eef_orientation_world": (transition_count, 3, 3),
        "audit_estimated_object_velocity": (transition_count, 3),
    }
    int64_names = {
        "boundary_formal_tick",
        "boundary_physics_step",
        "boundary_time_us",
        "commanded_motion_segment_index",
        "applied_reference_source_tick",
        "transition_source_tick",
        "transition_target_tick",
        "audit_source_physics_step",
        "audit_source_formal_tick",
        "audit_source_time_us",
        "audit_selected_reference_index",
    }
    bool_names = {"left_pad_contact", "right_pad_contact", "control_reference_valid", "action_mask"}
    uint8_names = {"agentview_rgb", "robot0_eye_in_hand_rgb"}
    int8_names = {"boundary_outcome_status_code", "handoff_state_code"}
    for name, shape in shapes.items():
        dtype = (
            np.int64
            if name in int64_names
            else np.bool_
            if name in bool_names
            else np.uint8
            if name in uint8_names
            else np.int8
            if name in int8_names
            else np.float64
        )
        _require_array(arrays, name, dtype=dtype, shape=shape)
    valid = arrays["control_reference_valid"]
    optional_names = (
        "applied_reference",
        "nullspace_joint_position_error",
        "eef_position_error",
        "eef_orientation_error_rotvec",
    )
    if not np.all(np.isnan(arrays["applied_reference"][~valid])) or not np.all(
        arrays["applied_reference_source_tick"][~valid] == -1
    ):
        raise ValueError("invalid control-reference sentinel encoding")
    if any(not np.all(np.isnan(arrays[name][~valid])) for name in optional_names[1:]):
        raise ValueError("invalid control-error sentinel encoding")
    if any(not np.all(np.isfinite(arrays[name][valid])) for name in optional_names):
        raise ValueError("valid control-reference rows must be finite")
    phases = metadata_envelope["audit_phase_id"]
    references = metadata_envelope["audit_reference_kind"]
    if (
        type(phases) is not list
        or type(references) is not list
        or len(phases) != transition_count
        or len(references) != transition_count
    ):
        raise ValueError("audit string lists do not match transition count")
    boundaries = []
    for index in range(boundary_count):
        deployment = StructuredDeploymentRecord(
            source_physics_step=int(arrays["boundary_physics_step"][index]),
            source_formal_tick=int(arrays["boundary_formal_tick"][index]),
            source_time_us=int(arrays["boundary_time_us"][index]),
            agentview_rgb=arrays["agentview_rgb"][index],
            robot0_eye_in_hand_rgb=arrays["robot0_eye_in_hand_rgb"][index],
            robot_qpos=arrays["robot_qpos"][index],
            robot_qvel=arrays["robot_qvel"][index],
            gripper_qpos=arrays["gripper_qpos"][index],
            gripper_qvel=arrays["gripper_qvel"][index],
            eef_position_world=arrays["eef_position_world"][index],
            eef_orientation_matrix_world=arrays["eef_orientation_matrix_world"][index],
        )
        present = bool(valid[index])
        qualification = StructuredQualificationRecord(
            object_pose=arrays["object_pose"][index],
            object_velocity=arrays["object_velocity"][index],
            commanded_motion_position=arrays["commanded_motion_position"][index],
            commanded_motion_velocity=arrays["commanded_motion_velocity"][index],
            commanded_motion_acceleration=arrays["commanded_motion_acceleration"][index],
            commanded_motion_segment_index=int(arrays["commanded_motion_segment_index"][index]),
            left_pad_contact=bool(arrays["left_pad_contact"][index]),
            right_pad_contact=bool(arrays["right_pad_contact"][index]),
            handoff_state=_HANDOFF_VALUE.get(int(arrays["handoff_state_code"][index]), "invalid"),
            relative_geometry=arrays["relative_geometry"][index],
            actuator_ctrl=arrays["actuator_ctrl"][index],
            applied_reference=arrays["applied_reference"][index] if present else None,
            applied_reference_source_tick=int(arrays["applied_reference_source_tick"][index])
            if present
            else None,
            nullspace_joint_position_error=arrays["nullspace_joint_position_error"][index]
            if present
            else None,
            eef_position_error=arrays["eef_position_error"][index] if present else None,
            eef_orientation_error_rotvec=arrays["eef_orientation_error_rotvec"][index]
            if present
            else None,
        )
        boundaries.append(
            StructuredBoundaryRecord(
                formal_tick_index=int(arrays["boundary_formal_tick"][index]),
                physics_step_index=int(arrays["boundary_physics_step"][index]),
                time_us=int(arrays["boundary_time_us"][index]),
                deployment=deployment,
                qualification=qualification,
                outcome_status=_OUTCOME_VALUE.get(
                    int(arrays["boundary_outcome_status_code"][index]), "invalid"
                ),
            )
        )
    transitions = []
    for index in range(transition_count):
        selected = int(arrays["audit_selected_reference_index"][index])
        audit = StructuredExpertAuditRecord(
            expert_realization_id=metadata.expert_realization_id,
            source_physics_step=int(arrays["audit_source_physics_step"][index]),
            source_formal_tick=int(arrays["audit_source_formal_tick"][index]),
            source_time_us=int(arrays["audit_source_time_us"][index]),
            phase_id=phases[index],
            reference_kind=references[index],
            selected_reference_index=None if selected == -1 else selected,
            target_eef_position_world=arrays["audit_target_eef_position_world"][index],
            target_eef_orientation_matrix_world=arrays["audit_target_eef_orientation_world"][index],
            estimated_object_velocity_world=arrays["audit_estimated_object_velocity"][index],
        )
        transitions.append(
            StructuredTransitionRecord(
                source_formal_tick=int(arrays["transition_source_tick"][index]),
                target_formal_tick=int(arrays["transition_target_tick"][index]),
                expert_action=arrays["expert_action"][index],
                action_mask=arrays["action_mask"][index],
                expert_audit=audit,
            )
        )
    event_envelope = _strict(
        _load_json_bytes(payloads["events.json"], name="events.json"),
        {"schema_version", "episode_id", "attempt_id", "events"},
        name="events.json",
    )
    if (
        type(event_envelope["schema_version"]) is not int
        or event_envelope["schema_version"] != 1
        or event_envelope["episode_id"] != metadata.episode_id
    ):
        raise ValueError("event envelope identity is invalid")
    event_attempt = AttemptId.from_mapping(
        event_envelope["attempt_id"],
        structured_expert_config_sha256=metadata.structured_expert_config_sha256,
    )
    if event_attempt != metadata.attempt_id or type(event_envelope["events"]) is not list:
        raise ValueError("event attempt identity is invalid")
    events = tuple(
        StructuredPhysicalEventRecord.from_mapping(item) for item in event_envelope["events"]
    )
    if len(events) != manifest["event_count"]:
        raise ValueError("event count mismatch")
    if (
        metadata_envelope["terminal_status"] != manifest["terminal_status"]
        or metadata_envelope["terminal_reason"] != manifest["terminal_reason"]
    ):
        raise ValueError("terminal metadata mismatch")
    return StructuredSynchronizedEpisode(
        metadata=metadata,
        boundaries=tuple(boundaries),
        transitions=tuple(transitions),
        physical_events=events,
        terminal_status=manifest["terminal_status"],
        terminal_reason=manifest["terminal_reason"],
    )


def load_verified_structured_episode_snapshot(root: Path) -> VerifiedStructuredEpisodeSnapshot:
    payloads = _read_exact_episode_files(root)
    episode = _episode_from_payloads(payloads)
    return VerifiedStructuredEpisodeSnapshot(
        episode=episode,
        files=MappingProxyType(dict(payloads)),
        manifest_sha256=_hash_bytes(payloads["manifest.json"]),
    )


def load_verified_structured_episode(root: Path) -> StructuredSynchronizedEpisode:
    return load_verified_structured_episode_snapshot(root).episode
