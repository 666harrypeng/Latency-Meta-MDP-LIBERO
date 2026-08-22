"""Lossless, no-overwrite storage for synchronized episode records."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from latency_meta_mdp.artifacts import sha256_file
from latency_meta_mdp.recording import SynchronizedEpisode


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(_jsonable(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _episode_arrays(episode: SynchronizedEpisode) -> dict[str, np.ndarray]:
    boundaries = episode.boundaries
    transitions = episode.transitions
    arrays: dict[str, np.ndarray] = {
        "boundary_formal_tick": np.asarray(
            [record.formal_tick_index for record in boundaries], dtype=np.int64
        ),
        "boundary_physics_step": np.asarray(
            [record.physics_step_index for record in boundaries], dtype=np.int64
        ),
        "boundary_time_us": np.asarray(
            [record.time_us for record in boundaries], dtype=np.int64
        ),
        "boundary_outcome_status": np.asarray(
            [record.outcome_status.value for record in boundaries]
        ),
        "agentview_rgb": np.stack(
            [record.deployment.images["agentview"].rgb for record in boundaries]
        ),
        "robot0_eye_in_hand_rgb": np.stack(
            [record.deployment.images["robot0_eye_in_hand"].rgb for record in boundaries]
        ),
        "robot_qpos": np.stack([record.deployment.robot_qpos for record in boundaries]),
        "robot_qvel": np.stack([record.deployment.robot_qvel for record in boundaries]),
        "gripper_qpos": np.stack(
            [record.deployment.gripper_qpos for record in boundaries]
        ),
        "gripper_qvel": np.stack(
            [record.deployment.gripper_qvel for record in boundaries]
        ),
        "transition_source_tick": np.asarray(
            [record.source_formal_tick for record in transitions], dtype=np.int64
        ),
        "transition_target_tick": np.asarray(
            [record.target_formal_tick for record in transitions], dtype=np.int64
        ),
        "expert_action": np.stack([record.expert_action for record in transitions]),
        "action_mask": np.stack([record.action_mask for record in transitions]),
        "expert_phase": np.asarray(
            [record.expert_audit.phase.value for record in transitions]
        ),
        "expert_history_start_time_us": np.asarray(
            [record.expert_audit.history_start_time_us for record in transitions],
            dtype=np.int64,
        ),
        "expert_history_sample_count": np.asarray(
            [record.expert_audit.history_sample_count for record in transitions],
            dtype=np.int64,
        ),
        "expert_target_eef_position_world": np.stack(
            [record.expert_audit.target_eef_position_world for record in transitions]
        ),
        "expert_estimated_object_velocity_world": np.stack(
            [record.expert_audit.estimated_object_velocity_world for record in transitions]
        ),
    }

    if episode.metadata.record_profile.includes_privileged:
        privileged = [record.privileged for record in boundaries]
        if any(record is None for record in privileged):
            raise ValueError("privileged episode contains a missing boundary record")
        records = [record for record in privileged if record is not None]
        arrays.update(
            {
                "object_pose": np.stack([record.object_pose for record in records]),
                "object_velocity": np.stack(
                    [record.object_velocity for record in records]
                ),
                "commanded_motion_position": np.stack(
                    [record.commanded_motion.position for record in records]
                ),
                "commanded_motion_velocity": np.stack(
                    [record.commanded_motion.velocity for record in records]
                ),
                "commanded_motion_acceleration": np.stack(
                    [record.commanded_motion.acceleration for record in records]
                ),
                "commanded_motion_segment_index": np.asarray(
                    [record.commanded_motion.segment_index for record in records],
                    dtype=np.int64,
                ),
                "left_pad_contact": np.asarray(
                    [record.contact.left for record in records], dtype=bool
                ),
                "right_pad_contact": np.asarray(
                    [record.contact.right for record in records], dtype=bool
                ),
                "handoff_state": np.asarray(
                    [record.handoff_state.value for record in records]
                ),
                "relative_geometry": np.stack(
                    [record.relative_geometry for record in records]
                ),
            }
        )

    if episode.metadata.record_profile.includes_control_debug:
        debug = [record.control_debug for record in boundaries]
        if any(record is None for record in debug):
            raise ValueError("debug episode contains a missing boundary record")
        records = [record for record in debug if record is not None]
        valid = np.asarray(
            [record.applied_reference is not None for record in records], dtype=bool
        )
        references = np.full(
            (len(records), episode.metadata.action_dim), np.nan, dtype=float
        )
        source_ticks = np.full(len(records), -1, dtype=np.int64)
        nullspace_error = np.full((len(records), 7), np.nan, dtype=float)
        eef_position_error = np.full((len(records), 3), np.nan, dtype=float)
        eef_orientation_error = np.full((len(records), 3), np.nan, dtype=float)
        for index, record in enumerate(records):
            if not valid[index]:
                continue
            references[index] = record.applied_reference
            source_ticks[index] = record.applied_reference_source_formal_tick
            nullspace_error[index] = record.nullspace_joint_position_error
            eef_position_error[index] = record.eef_position_error
            eef_orientation_error[index] = record.eef_orientation_error_rotvec
        arrays.update(
            {
                "actuator_ctrl": np.stack(
                    [record.actuator_ctrl for record in records]
                ),
                "control_reference_valid": valid,
                "applied_reference": references,
                "applied_reference_source_tick": source_ticks,
                "nullspace_joint_position_error": nullspace_error,
                "eef_position_error": eef_position_error,
                "eef_orientation_error_rotvec": eef_orientation_error,
            }
        )
    return arrays


def _metadata_payload(episode: SynchronizedEpisode) -> dict[str, Any]:
    metadata = episode.metadata
    return {
        "schema_version": metadata.schema_version,
        "episode_id": metadata.episode_id,
        "task_id": metadata.task_id,
        "instruction": metadata.instruction,
        "level": metadata.level,
        "scene_seed": metadata.scene_seed,
        "motion_seed": metadata.motion_seed,
        "expert_seed": metadata.expert_seed,
        "physics_dt_us": metadata.physics_dt_us,
        "formal_tick_us": metadata.formal_tick_us,
        "action_contract_id": metadata.action_contract_id,
        "action_dim": metadata.action_dim,
        "actuator_dim": metadata.actuator_dim,
        "expert_id": metadata.expert_id,
        "record_profile": metadata.record_profile,
        "config_sha256": metadata.config_sha256,
        "motion_profile": metadata.motion_profile,
        "terminal_status": episode.terminal_status,
        "terminal_reason": episode.terminal_reason,
    }


def _events_payload(episode: SynchronizedEpisode) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "episode_id": episode.metadata.episode_id,
        "events": [
            {
                "kind": event.kind,
                "physics_step_index": event.physics_step_index,
                "time_us": event.time_us,
                "payload": event.payload,
                "terminal_reason": event.terminal_reason,
            }
            for event in episode.physical_events
        ],
    }


def write_synchronized_episode_artifact(
    *,
    episode: SynchronizedEpisode,
    output_dir: Path,
) -> Path:
    """Write one validated episode atomically without overwriting existing data."""

    episode.validate_complete()
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"episode output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.building-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"episode staging output already exists: {staging}")
    try:
        staging.mkdir()
        arrays_path = staging / "arrays.npz"
        with arrays_path.open("xb") as handle:
            np.savez_compressed(handle, **_episode_arrays(episode))
            handle.flush()
            os.fsync(handle.fileno())
        _write_json(staging / "metadata.json", _metadata_payload(episode))
        _write_json(staging / "events.json", _events_payload(episode))
        artifacts = {
            name: sha256_file(staging / name)
            for name in ("arrays.npz", "events.json", "metadata.json")
        }
        _write_json(
            staging / "manifest.json",
            {
                "schema_version": 2,
                "format_id": "synchronized_episode_npz_v2",
                "episode_id": episode.metadata.episode_id,
                "record_profile": episode.metadata.record_profile,
                "boundary_count": len(episode.boundaries),
                "transition_count": len(episode.transitions),
                "physical_event_count": len(episode.physical_events),
                "terminal_status": episode.terminal_status,
                "terminal_reason": episode.terminal_reason,
                "artifacts": artifacts,
            },
        )
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target / "manifest.json"
