"""Immutable resumable payload for one qualified formal-source success."""

from __future__ import annotations

import io
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from latency_meta_mdp.data.collection.artifacts import (
    _check_final_root,
    _hash_bytes,
    _load_json_bytes,
    _publish_tree,
    _read_regular_file_nofollow,
    _strict,
)
from latency_meta_mdp.data.collection.recording_contracts import (
    StructuredBoundaryRecord,
    StructuredDeploymentRecord,
    StructuredExpertAuditRecord,
    StructuredPhysicalEventRecord,
    StructuredQualificationRecord,
    StructuredTransitionRecord,
    json_thaw,
)
from latency_meta_mdp.data.source.config import SourceCorpusConfig
from latency_meta_mdp.data.source.contracts import (
    FormalSourceEpisodeMetadata,
    FormalSourceSynchronizedEpisode,
)
from latency_meta_mdp.data.source.parquet import (
    decode_png,
    episode_to_frame_table,
)
from latency_meta_mdp.data.source.recording import (
    QualifiedSourceRecording,
)
from latency_meta_mdp.data.source.schema import SOURCE_FRAME_SCHEMA

_FORMAT = "structured_expert_source_success_payload_v1"
_FILES = frozenset({"episode.parquet", "record.json", "manifest.json"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _parquet_bytes(episode: FormalSourceSynchronizedEpisode, config: SourceCorpusConfig) -> bytes:
    output = io.BytesIO()
    pq.write_table(
        episode_to_frame_table(episode, config=config),
        output,
        compression=config.parquet_compression,
        compression_level=config.parquet_compression_level,
    )
    return output.getvalue()


@dataclass(frozen=True)
class LoadedSourceSuccess:
    episode: FormalSourceSynchronizedEpisode
    strategy_parameters: Mapping[str, Any]
    selected_planner_fingerprint: str
    qualification: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.episode, FormalSourceSynchronizedEpisode):
            raise TypeError("episode must be FormalSourceSynchronizedEpisode")
        if (
            type(self.selected_planner_fingerprint) is not str
            or _SHA256.fullmatch(self.selected_planner_fingerprint) is None
        ):
            raise ValueError("selected_planner_fingerprint must be a SHA-256 digest")
        for name in ("strategy_parameters", "qualification"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")
            detached = dict(json_thaw(value))
            json.dumps(detached, allow_nan=False)
            object.__setattr__(self, name, MappingProxyType(detached))
        if (
            self.qualification.get("eligible") is not True
            or self.qualification.get("failures") != []
            or type(self.qualification.get("actual_rollout_safety")) is not dict
        ):
            raise ValueError("source success requires complete eligible qualification")


def write_source_success_payload(
    *,
    target: Path,
    recording: QualifiedSourceRecording,
    source_config: SourceCorpusConfig,
    strategy_parameters: Mapping[str, Any],
    selected_planner_fingerprint: str,
) -> Path:
    if not isinstance(recording, QualifiedSourceRecording):
        raise TypeError("recording must be QualifiedSourceRecording")
    if not isinstance(source_config, SourceCorpusConfig):
        raise TypeError("source_config must be SourceCorpusConfig")
    loaded = LoadedSourceSuccess(
        episode=recording.episode,
        strategy_parameters=strategy_parameters,
        selected_planner_fingerprint=selected_planner_fingerprint,
        qualification=recording.qualification_mapping(),
    )
    parquet = _parquet_bytes(loaded.episode, source_config)
    record = _json_bytes(
        {
            "schema_version": 1,
            "metadata": loaded.episode.metadata.to_mapping(),
            "terminal_reason": loaded.episode.terminal_reason,
            "physical_events": [event.to_mapping() for event in loaded.episode.physical_events],
            "strategy_parameters": json_thaw(loaded.strategy_parameters),
            "selected_planner_fingerprint": loaded.selected_planner_fingerprint,
            "qualification": json_thaw(loaded.qualification),
            "source_config_sha256": source_config.sha256,
        }
    )
    artifacts = {
        "episode.parquet": _hash_bytes(parquet),
        "record.json": _hash_bytes(record),
    }
    manifest = _json_bytes(
        {
            "schema_version": 1,
            "format_id": _FORMAT,
            "episode_id": loaded.episode.metadata.episode_id,
            "boundary_count": len(loaded.episode.boundaries),
            "transition_count": len(loaded.episode.transitions),
            "source_config_sha256": source_config.sha256,
            "artifacts": artifacts,
        }
    )
    return _publish_tree(
        Path(target),
        {
            "episode.parquet": parquet,
            "record.json": record,
            "manifest.json": manifest,
        },
        expected_hashes=artifacts,
    )


def _optional_vector(value: Any, *, length: int) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64).reshape(length)


def _episode_from_frame_table(
    table: Any,
    *,
    metadata: FormalSourceEpisodeMetadata,
    events: tuple[StructuredPhysicalEventRecord, ...],
    terminal_reason: str,
) -> FormalSourceSynchronizedEpisode:
    if table.schema != SOURCE_FRAME_SCHEMA or table.num_rows < 2:
        raise ValueError("source success Parquet schema or row count is invalid")
    rows = table.to_pylist()
    boundaries = []
    transitions = []
    for index, row in enumerate(rows):
        deployment = StructuredDeploymentRecord(
            source_physics_step=row["physics_step"],
            source_formal_tick=row["formal_tick"],
            source_time_us=row["time_us"],
            agentview_rgb=decode_png(row["agentview_rgb"]["bytes"]),
            robot0_eye_in_hand_rgb=decode_png(row["wrist_rgb"]["bytes"]),
            robot_qpos=np.asarray(row["robot_qpos"], dtype=np.float64),
            robot_qvel=np.asarray(row["robot_qvel"], dtype=np.float64),
            gripper_qpos=np.asarray(row["gripper_qpos"], dtype=np.float64),
            gripper_qvel=np.asarray(row["gripper_qvel"], dtype=np.float64),
            eef_position_world=np.asarray(row["eef_position_world"], dtype=np.float64),
            eef_orientation_matrix_world=np.asarray(
                row["eef_orientation_matrix_world"], dtype=np.float64
            ).reshape(3, 3),
        )
        qualification = StructuredQualificationRecord(
            object_pose=np.asarray(row["object_pose"], dtype=np.float64),
            object_velocity=np.asarray(row["object_velocity"], dtype=np.float64),
            commanded_motion_position=np.asarray(
                row["commanded_motion_position"], dtype=np.float64
            ),
            commanded_motion_velocity=np.asarray(
                row["commanded_motion_velocity"], dtype=np.float64
            ),
            commanded_motion_acceleration=np.asarray(
                row["commanded_motion_acceleration"], dtype=np.float64
            ),
            commanded_motion_segment_index=row["commanded_motion_segment_index"],
            left_pad_contact=row["left_pad_contact"],
            right_pad_contact=row["right_pad_contact"],
            handoff_state=row["handoff_state"],
            relative_geometry=np.asarray(row["relative_geometry"], dtype=np.float64),
            actuator_ctrl=np.asarray(row["actuator_ctrl"], dtype=np.float64),
            applied_reference=_optional_vector(row["applied_reference"], length=7),
            applied_reference_source_tick=row["applied_reference_source_tick"],
            nullspace_joint_position_error=_optional_vector(
                row["nullspace_joint_position_error"], length=7
            ),
            eef_position_error=_optional_vector(row["eef_position_error"], length=3),
            eef_orientation_error_rotvec=_optional_vector(
                row["eef_orientation_error_rotvec"], length=3
            ),
        )
        boundaries.append(
            StructuredBoundaryRecord(
                formal_tick_index=row["formal_tick"],
                physics_step_index=row["physics_step"],
                time_us=row["time_us"],
                deployment=deployment,
                qualification=qualification,
                outcome_status=row["outcome_status"],
            )
        )
        if index == len(rows) - 1:
            continue
        audit = StructuredExpertAuditRecord(
            expert_realization_id=metadata.expert_realization_id,
            source_physics_step=row["physics_step"],
            source_formal_tick=row["formal_tick"],
            source_time_us=row["time_us"],
            phase_id=row["phase_id"],
            reference_kind=row["reference_kind"],
            selected_reference_index=row["selected_reference_index"],
            target_eef_position_world=np.asarray(
                row["target_eef_position_world"], dtype=np.float64
            ),
            target_eef_orientation_matrix_world=np.asarray(
                row["target_eef_orientation_matrix_world"], dtype=np.float64
            ).reshape(3, 3),
            estimated_object_velocity_world=np.asarray(
                row["estimated_object_velocity_world"], dtype=np.float64
            ),
        )
        transitions.append(
            StructuredTransitionRecord(
                source_formal_tick=row["formal_tick"],
                target_formal_tick=row["formal_tick"] + 1,
                expert_action=np.asarray(row["expert_action"], dtype=np.float64),
                action_mask=np.asarray(row["action_mask"], dtype=np.bool_),
                expert_audit=audit,
            )
        )
    return FormalSourceSynchronizedEpisode(
        metadata=metadata,
        boundaries=tuple(boundaries),
        transitions=tuple(transitions),
        physical_events=events,
        terminal_reason=terminal_reason,
    )


def load_source_success_payload(
    root: Path,
    *,
    source_config: SourceCorpusConfig,
) -> LoadedSourceSuccess:
    if not isinstance(source_config, SourceCorpusConfig):
        raise TypeError("source_config must be SourceCorpusConfig")
    root = _check_final_root(Path(root))
    if {path.name for path in root.iterdir()} != _FILES:
        raise ValueError("source success payload inventory is invalid")
    payloads = {name: _read_regular_file_nofollow(root / name, name=name) for name in _FILES}
    manifest = _strict(
        _load_json_bytes(payloads["manifest.json"], name="success manifest"),
        {
            "schema_version",
            "format_id",
            "episode_id",
            "boundary_count",
            "transition_count",
            "source_config_sha256",
            "artifacts",
        },
        name="source success manifest",
    )
    if manifest["schema_version"] != 1 or manifest["format_id"] != _FORMAT:
        raise ValueError("source success payload format is invalid")
    if manifest["source_config_sha256"] != source_config.sha256:
        raise ValueError("source success storage config changed")
    expected_hashes = {
        name: _hash_bytes(payloads[name]) for name in ("episode.parquet", "record.json")
    }
    if manifest["artifacts"] != expected_hashes:
        raise ValueError("source success payload hash mismatch")
    record = _strict(
        _load_json_bytes(payloads["record.json"], name="success record"),
        {
            "schema_version",
            "metadata",
            "terminal_reason",
            "physical_events",
            "strategy_parameters",
            "selected_planner_fingerprint",
            "qualification",
            "source_config_sha256",
        },
        name="source success record",
    )
    if record["schema_version"] != 1 or record["source_config_sha256"] != (source_config.sha256):
        raise ValueError("source success record config is invalid")
    metadata = FormalSourceEpisodeMetadata.from_mapping(record["metadata"])
    if metadata.episode_id != manifest["episode_id"]:
        raise ValueError("source success episode identity mismatch")
    events = tuple(
        StructuredPhysicalEventRecord.from_mapping(value) for value in record["physical_events"]
    )
    table = pq.read_table(io.BytesIO(payloads["episode.parquet"]))
    if table.num_rows != manifest["boundary_count"] or manifest["transition_count"] != (
        table.num_rows - 1
    ):
        raise ValueError("source success episode counts changed")
    episode = _episode_from_frame_table(
        table,
        metadata=metadata,
        events=events,
        terminal_reason=record["terminal_reason"],
    )
    return LoadedSourceSuccess(
        episode=episode,
        strategy_parameters=record["strategy_parameters"],
        selected_planner_fingerprint=record["selected_planner_fingerprint"],
        qualification=record["qualification"],
    )
