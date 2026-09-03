"""Central relational metadata for the formal structured-expert source corpus."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

import pyarrow as pa

from latency_meta_mdp.expert_realization.contracts import TaskInstanceId
from latency_meta_mdp.expert_realization.recording_contracts import json_thaw
from latency_meta_mdp.expert_realization.source_corpus.contracts import (
    FormalSourceEpisodeMetadata,
    FormalSourceSynchronizedEpisode,
)
from latency_meta_mdp.expert_realization.source_corpus.parquet import EpisodeLocation
from latency_meta_mdp.expert_realization.source_corpus.schema import (
    EPISODE_SCHEMA,
    EVENT_SCHEMA,
    TASK_INSTANCE_SCHEMA,
    source_schema_document,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SHARD = re.compile(r"^data/level-([123])/shard-[0-9]{5}[.]parquet$")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _normalized_text(value: Any, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty normalized string")
    return value


@dataclass(frozen=True)
class SourceTaskMetadataEntry:
    task_instance_id: TaskInstanceId
    corpus_id: str
    logical_master_task_index: int
    instruction: str
    motion_profile_json: str
    initial_state_npz: bytes
    admitted_realization_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.task_instance_id, TaskInstanceId):
            raise TypeError("task_instance_id must be a TaskInstanceId")
        _normalized_text(self.corpus_id, name="corpus_id")
        _normalized_text(self.instruction, name="instruction")
        if type(self.logical_master_task_index) is not int or self.logical_master_task_index < 0:
            raise ValueError("logical_master_task_index must be non-negative")
        if type(self.motion_profile_json) is not str:
            raise TypeError("motion_profile_json must be a string")
        try:
            json.loads(self.motion_profile_json)
        except json.JSONDecodeError as error:
            raise ValueError("motion_profile_json must contain valid JSON") from error
        if hashlib.sha256(self.motion_profile_json.encode()).hexdigest() != (
            self.task_instance_id.motion_profile_sha256
        ):
            raise ValueError("motion profile hash does not match task identity")
        if type(self.initial_state_npz) is not bytes:
            raise TypeError("initial_state_npz must be bytes")
        if hashlib.sha256(self.initial_state_npz).hexdigest() != (
            self.task_instance_id.initial_state_sha256
        ):
            raise ValueError("initial state hash does not match task identity")
        if type(self.admitted_realization_count) is not int or self.admitted_realization_count <= 0:
            raise ValueError("admitted_realization_count must be positive")

    def to_row(self) -> dict[str, Any]:
        return {
            "task_instance_id": self.task_instance_id.canonical_json(),
            "corpus_id": self.corpus_id,
            "logical_master_task_index": self.logical_master_task_index,
            "task_instance_seed": self.task_instance_id.task_instance_seed,
            "level": self.task_instance_id.level,
            "instruction": self.instruction,
            "motion_profile_sha256": self.task_instance_id.motion_profile_sha256,
            "initial_state_sha256": self.task_instance_id.initial_state_sha256,
            "motion_profile_json": self.motion_profile_json,
            "initial_state_npz": self.initial_state_npz,
            "admitted_realization_count": self.admitted_realization_count,
        }


@dataclass(frozen=True)
class SourceEpisodeMetadataEntry:
    episode: FormalSourceSynchronizedEpisode
    logical_master_task_index: int
    strategy_parameters: Mapping[str, Any]
    selected_planner_fingerprint: str
    qualification: Mapping[str, Any]
    location: EpisodeLocation
    data_shard: str

    def __post_init__(self) -> None:
        if not isinstance(self.episode, FormalSourceSynchronizedEpisode):
            raise TypeError("episode must be FormalSourceSynchronizedEpisode")
        if (
            type(self.logical_master_task_index) is not int
            or self.logical_master_task_index != self.episode.metadata.logical_master_task_index
        ):
            raise ValueError("logical_master_task_index does not match episode metadata")
        for name in ("strategy_parameters", "qualification"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")
            _canonical_json(json_thaw(value))
        qualification = json_thaw(self.qualification)
        if qualification.get("eligible") is not True or qualification.get("failures") != []:
            raise ValueError("source episode qualification must be an eligible success")
        if (
            type(self.selected_planner_fingerprint) is not str
            or _SHA256.fullmatch(self.selected_planner_fingerprint) is None
        ):
            raise ValueError("selected_planner_fingerprint must be a SHA-256 digest")
        if not isinstance(self.location, EpisodeLocation):
            raise TypeError("location must be EpisodeLocation")
        if (
            self.location.episode_id != self.episode.metadata.episode_id
            or self.location.row_count != len(self.episode.boundaries)
        ):
            raise ValueError("episode location does not match source episode")
        path = PurePosixPath(self.data_shard)
        match = _SHARD.fullmatch(self.data_shard)
        if path.as_posix() != self.data_shard or match is None:
            raise ValueError("data_shard must use the canonical level partition")
        if int(match.group(1)) != self.episode.metadata.task_instance_id.level:
            raise ValueError("data_shard level partition does not match episode")

    def to_row(self) -> dict[str, Any]:
        metadata = self.episode.metadata
        events = {event.kind: event.time_us for event in self.episode.physical_events}
        required = {
            "first_contact",
            "stable_grasp",
            "handoff",
            "lift_threshold",
            "success",
        }
        if len(self.episode.physical_events) != len(required) or set(events) != required:
            raise ValueError("successful source episode has an invalid physical-event inventory")
        key = metadata.expert_realization_id.expert_realization_key
        return {
            "episode_id": metadata.episode_id,
            "task_instance_id": metadata.task_instance_id.canonical_json(),
            "logical_master_task_index": self.logical_master_task_index,
            "level": metadata.task_instance_id.level,
            "realization_index": key.realization_index,
            "realization_seed": key.realization_seed,
            "strategy_family": metadata.strategy_family.value,
            "strategy_parameters_json": _canonical_json(json_thaw(self.strategy_parameters)),
            "frame_count": len(self.episode.boundaries),
            "terminal_tick": self.episode.boundaries[-1].formal_tick_index,
            "first_contact_time_us": events["first_contact"],
            "stable_grasp_time_us": events["stable_grasp"],
            "handoff_time_us": events["handoff"],
            "lift_threshold_time_us": events["lift_threshold"],
            "success_time_us": events["success"],
            "selected_planner_fingerprint": self.selected_planner_fingerprint,
            "qualification_json": _canonical_json(json_thaw(self.qualification)),
            "data_shard": self.data_shard,
            "row_group_index": self.location.row_group_index,
            "row_offset": self.location.row_offset,
            "row_count": self.location.row_count,
        }


def _nonempty_tuple(value: Any, item_type: type, *, name: str) -> tuple[Any, ...]:
    if type(value) is not tuple or not value or any(
        not isinstance(item, item_type) for item in value
    ):
        raise ValueError(f"{name} must be a non-empty tuple of {item_type.__name__}")
    return value


def build_task_instance_table(entries: tuple[SourceTaskMetadataEntry, ...]) -> pa.Table:
    rows = _nonempty_tuple(entries, SourceTaskMetadataEntry, name="task metadata")
    identities = [row.task_instance_id.canonical_json() for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("task instance IDs must be unique")
    return pa.Table.from_pylist([row.to_row() for row in rows], schema=TASK_INSTANCE_SCHEMA)


def build_episode_table(entries: tuple[SourceEpisodeMetadataEntry, ...]) -> pa.Table:
    rows = _nonempty_tuple(entries, SourceEpisodeMetadataEntry, name="episode metadata")
    identities = [row.episode.metadata.episode_id for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("episode IDs must be unique")
    return pa.Table.from_pylist([row.to_row() for row in rows], schema=EPISODE_SCHEMA)


def build_event_table(episodes: tuple[FormalSourceSynchronizedEpisode, ...]) -> pa.Table:
    values = _nonempty_tuple(
        episodes,
        FormalSourceSynchronizedEpisode,
        name="event episodes",
    )
    episode_ids = [episode.metadata.episode_id for episode in values]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("event episode IDs must be unique")
    rows = []
    for episode in values:
        for index, event in enumerate(episode.physical_events):
            rows.append(
                {
                    "episode_id": episode.metadata.episode_id,
                    "event_index": index,
                    "kind": event.kind,
                    "physics_step": event.physics_step_index,
                    "time_us": event.time_us,
                    "payload_json": _canonical_json(json_thaw(event.payload)),
                    "terminal_reason": event.terminal_reason,
                }
            )
    if not rows:
        raise ValueError("event table must be non-empty")
    return pa.Table.from_pylist(rows, schema=EVENT_SCHEMA)


def build_schema_document() -> dict[str, Any]:
    return source_schema_document()


_PROVENANCE_FIELDS = (
    "corpus_id",
    "record_profile",
    "task_id",
    "instruction",
    "physics_dt_us",
    "formal_tick_us",
    "camera_height",
    "camera_width",
    "action_contract_id",
    "action_dim",
    "actuator_dim",
    "expert_id",
    "formal_corpus_config_sha256",
    "source_corpus_config_sha256",
    "task_config_sha256",
    "runtime_config_sha256",
    "controller_config_sha256",
    "structured_expert_config_sha256",
    "curobo_planner_config_sha256",
    "implementation",
)


def build_provenance_document(
    metadata: tuple[FormalSourceEpisodeMetadata, ...],
) -> dict[str, Any]:
    values = _nonempty_tuple(
        metadata,
        FormalSourceEpisodeMetadata,
        name="source metadata",
    )
    first = values[0]
    motion_config_by_level: dict[str, str] = {}
    for item in values:
        level = str(item.task_instance_id.level)
        existing = motion_config_by_level.get(level)
        if existing is not None and existing != item.motion_config_sha256:
            raise ValueError("one level contains multiple motion config identities")
        motion_config_by_level[level] = item.motion_config_sha256
    expected = {
        name: (
            first.implementation.to_mapping()
            if name == "implementation"
            else getattr(first, name)
        )
        for name in _PROVENANCE_FIELDS
    }
    for item in values[1:]:
        actual = {
            name: (
                item.implementation.to_mapping()
                if name == "implementation"
                else getattr(item, name)
            )
            for name in _PROVENANCE_FIELDS
        }
        if actual != expected:
            raise ValueError("source episodes do not share one dataset-level provenance contract")
    return {
        "schema_version": 2,
        "format_id": "structured_expert_source_provenance_v2",
        "motion_config_sha256_by_level": dict(sorted(motion_config_by_level.items())),
        **expected,
    }
