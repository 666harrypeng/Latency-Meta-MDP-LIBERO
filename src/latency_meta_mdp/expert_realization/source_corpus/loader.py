"""Verified relational loader for the formal structured-expert source corpus."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from latency_meta_mdp.expert_realization.artifacts import _check_final_root, _hash_file
from latency_meta_mdp.expert_realization.contracts import FormalRequestUniverse, TaskInstanceId
from latency_meta_mdp.expert_realization.source_corpus.config import (
    MasterTaskSplitPlan,
    SourceCorpusConfig,
)
from latency_meta_mdp.expert_realization.source_corpus.schema import (
    EPISODE_SCHEMA,
    EPISODE_SCHEMA_V1_SPLIT,
    EVENT_SCHEMA,
    SOURCE_FRAME_FIELDS,
    SOURCE_FRAME_SCHEMA,
    TASK_INSTANCE_SCHEMA,
    TASK_INSTANCE_SCHEMA_V1_SPLIT,
    SourceFieldRole,
    legacy_split_source_schema_document,
    source_schema_document,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SHARD = re.compile(r"^data/level-[123]/shard-[0-9]{5}[.]parquet$")
_FIXED_ARTIFACTS = {
    "README.md",
    "schema.json",
    "provenance.json",
    "meta/task_instances.parquet",
    "meta/episodes.parquet",
    "meta/events.parquet",
    "meta/collection_summary.json",
}
_TRANSITION_FIELDS = {
    "expert_action",
    "action_mask",
    "phase_id",
    "reference_kind",
    "selected_reference_index",
    "target_eef_position_world",
    "target_eef_orientation_matrix_world",
    "estimated_object_velocity_world",
}


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} must contain valid JSON") from error
    if type(value) is not dict:
        raise ValueError(f"{name} must contain a JSON mapping")
    return value


def _strict(value: Any, expected: set[str], *, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be a mapping")
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown or missing:
        raise ValueError(f"{name} fields mismatch; unknown={unknown}, missing={missing}")
    return value


@dataclass(frozen=True)
class SourceCorpusManifest:
    corpus_id: str
    request_sha256: str
    source_config_sha256: str
    split_plan_sha256: str | None
    admitted_master_task_indices: tuple[int, ...]
    master_task_count: int
    level_task_instance_count: int
    episode_count: int
    episodes_by_level: Mapping[str, int]
    frame_count: int
    shard_count: int
    artifacts: Mapping[str, Mapping[str, Any]]
    format_id: str = "structured_expert_source_corpus_v2"

    @classmethod
    def from_mapping(cls, mapping: Any) -> SourceCorpusManifest:
        if type(mapping) is not dict:
            raise TypeError("source manifest must be a mapping")
        format_id = mapping.get("format_id")
        if format_id == "structured_expert_source_corpus_v1":
            version_fields = {"split_plan_sha256"}
            schema_version = 1
        elif format_id == "structured_expert_source_corpus_v2":
            version_fields = set()
            schema_version = 2
        else:
            raise ValueError("source manifest is not a complete supported corpus")
        raw = _strict(
            mapping,
            {
                "schema_version",
                "format_id",
                "complete",
                "corpus_id",
                "request_sha256",
                "source_config_sha256",
                "admitted_master_task_indices",
                "master_task_count",
                "level_task_instance_count",
                "episode_count",
                "episodes_by_level",
                "frame_count",
                "shard_count",
                "artifacts",
            }
            | version_fields,
            name="source manifest",
        )
        if (
            raw["schema_version"] != schema_version
            or raw["complete"] is not True
        ):
            raise ValueError("source manifest is not a complete supported corpus")
        if type(raw["corpus_id"]) is not str or not raw["corpus_id"]:
            raise ValueError("source manifest corpus_id is invalid")
        for name in ("request_sha256", "source_config_sha256"):
            if type(raw[name]) is not str or _SHA256.fullmatch(raw[name]) is None:
                raise ValueError(f"source manifest {name} is invalid")
        split_sha = raw.get("split_plan_sha256")
        if split_sha is not None and (
            type(split_sha) is not str or _SHA256.fullmatch(split_sha) is None
        ):
            raise ValueError("source manifest split_plan_sha256 is invalid")
        indices = raw["admitted_master_task_indices"]
        if (
            type(indices) is not list
            or any(type(value) is not int or value < 0 for value in indices)
            or indices != sorted(set(indices))
        ):
            raise ValueError("admitted master-task indices are invalid")
        for name in (
            "master_task_count",
            "level_task_instance_count",
            "episode_count",
            "frame_count",
            "shard_count",
        ):
            value = raw[name]
            if type(value) is not int or value <= 0:
                raise ValueError(f"source manifest {name} must be positive")
        if raw["master_task_count"] != len(indices):
            raise ValueError("source manifest master-task count is inconsistent")
        by_level = raw["episodes_by_level"]
        if type(by_level) is not dict or any(
            key not in {"1", "2", "3"} or type(value) is not int or value <= 0
            for key, value in by_level.items()
        ):
            raise ValueError("source manifest per-level counts are invalid")
        if sum(by_level.values()) != raw["episode_count"]:
            raise ValueError("source manifest episode count is inconsistent")
        artifacts = raw["artifacts"]
        if type(artifacts) is not dict:
            raise ValueError("source manifest artifacts must be a mapping")
        frozen_artifacts = {}
        for relative, metadata in artifacts.items():
            if type(relative) is not str:
                raise ValueError("source artifact path is invalid")
            child = _strict(metadata, {"sha256", "bytes"}, name="source artifact")
            if type(child["sha256"]) is not str or _SHA256.fullmatch(child["sha256"]) is None:
                raise ValueError("source artifact SHA-256 is invalid")
            if type(child["bytes"]) is not int or child["bytes"] < 0:
                raise ValueError("source artifact byte count is invalid")
            frozen_artifacts[relative] = MappingProxyType(dict(child))
        return cls(
            corpus_id=raw["corpus_id"],
            request_sha256=raw["request_sha256"],
            source_config_sha256=raw["source_config_sha256"],
            split_plan_sha256=split_sha,
            admitted_master_task_indices=tuple(indices),
            master_task_count=raw["master_task_count"],
            level_task_instance_count=raw["level_task_instance_count"],
            episode_count=raw["episode_count"],
            episodes_by_level=MappingProxyType(dict(by_level)),
            frame_count=raw["frame_count"],
            shard_count=raw["shard_count"],
            artifacts=MappingProxyType(frozen_artifacts),
            format_id=format_id,
        )

    @property
    def legacy_split(self) -> bool:
        return self.format_id == "structured_expert_source_corpus_v1"


@dataclass(frozen=True)
class VerifiedSourceEpisode:
    metadata: Mapping[str, Any]
    frames: pa.Table


class VerifiedSourceCorpus:
    def __init__(
        self,
        *,
        root: Path,
        manifest: SourceCorpusManifest,
        episode_rows: tuple[dict[str, Any], ...],
    ) -> None:
        self.root = root
        self.manifest = manifest
        self._episodes = MappingProxyType({row["episode_id"]: row for row in episode_rows})

    def episode_ids(self, *, level: int, split: str | None = None) -> tuple[str, ...]:
        if type(level) is not int or level not in (1, 2, 3):
            raise ValueError("level must be one of 1, 2, or 3")
        if split is not None and split not in {"train", "validation"}:
            raise ValueError("split must be train, validation, or None")
        if split is not None and any("split" not in row for row in self._episodes.values()):
            raise ValueError("canonical source corpus has no embedded split")
        return tuple(
            sorted(
                row["episode_id"]
                for row in self._episodes.values()
                if row["level"] == level
                and (split is None or row["split"] == split)
            )
        )

    def read_episode(self, episode_id: str) -> VerifiedSourceEpisode:
        if episode_id not in self._episodes:
            raise KeyError(f"unknown source episode: {episode_id}")
        metadata = self._episodes[episode_id]
        parquet = pq.ParquetFile(self.root / metadata["data_shard"])
        frames = parquet.read_row_group(metadata["row_group_index"])
        if frames.num_rows != metadata["row_count"]:
            raise ValueError("source episode row count changed after verification")
        return VerifiedSourceEpisode(MappingProxyType(dict(metadata)), frames)

    def read_fields(
        self,
        episode_id: str,
        *,
        fields: tuple[str, ...],
        allowed_roles: frozenset[SourceFieldRole],
    ) -> pa.Table:
        if type(fields) is not tuple or not fields or any(type(name) is not str for name in fields):
            raise TypeError("fields must be a non-empty tuple of strings")
        if len(set(fields)) != len(fields):
            raise ValueError("fields must be unique")
        if type(allowed_roles) is not frozenset or not allowed_roles or any(
            not isinstance(role, SourceFieldRole) for role in allowed_roles
        ):
            raise TypeError("allowed_roles must be a non-empty frozenset of SourceFieldRole")
        specs = {field.name: field for field in SOURCE_FRAME_FIELDS}
        for name in fields:
            if name not in specs:
                raise KeyError(f"unknown source field: {name}")
            if not set(specs[name].roles) & set(allowed_roles):
                raise PermissionError(f"source field is not allowed: {name}")
        return self.read_episode(episode_id).frames.select(fields)


def _verify_inventory(root: Path, manifest: SourceCorpusManifest) -> None:
    actual = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        info = os.lstat(path)
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"source corpus contains a symlink: {relative}")
        if stat.S_ISREG(info.st_mode):
            actual.add(relative)
    expected = set(manifest.artifacts) | {"manifest.json"}
    if actual != expected:
        raise ValueError("source corpus file inventory does not match manifest")
    data_paths = set(manifest.artifacts) - _FIXED_ARTIFACTS
    if not data_paths or any(_SHARD.fullmatch(path) is None for path in data_paths):
        raise ValueError("source corpus contains a noncanonical artifact path")
    if len(data_paths) != manifest.shard_count:
        raise ValueError("source manifest shard count is inconsistent")
    for relative, metadata in manifest.artifacts.items():
        path = root / relative
        if path.stat().st_size != metadata["bytes"]:
            raise ValueError(f"source artifact size mismatch: {relative}")
        if _hash_file(path) != metadata["sha256"]:
            raise ValueError(f"source artifact hash mismatch: {relative}")


def _load_metadata_tables(
    root: Path, *, legacy_split: bool
) -> tuple[pa.Table, pa.Table, pa.Table]:
    tasks = pq.read_table(root / "meta/task_instances.parquet")
    episodes = pq.read_table(root / "meta/episodes.parquet")
    events = pq.read_table(root / "meta/events.parquet")
    expected_tasks = TASK_INSTANCE_SCHEMA_V1_SPLIT if legacy_split else TASK_INSTANCE_SCHEMA
    expected_episodes = EPISODE_SCHEMA_V1_SPLIT if legacy_split else EPISODE_SCHEMA
    if tasks.schema != expected_tasks:
        raise ValueError("task metadata schema is invalid")
    if episodes.schema != expected_episodes:
        raise ValueError("episode metadata schema is invalid")
    if events.schema != EVENT_SCHEMA:
        raise ValueError("event metadata schema is invalid")
    return tasks, episodes, events


def _validate_frame_group(table: pa.Table, metadata: Mapping[str, Any]) -> None:
    if table.schema != SOURCE_FRAME_SCHEMA:
        raise ValueError("source data shard schema is invalid")
    expected_rows = metadata["row_count"]
    if table.num_rows != expected_rows:
        raise ValueError("source episode row group has wrong row count")
    if table["episode_id"].to_pylist() != [metadata["episode_id"]] * expected_rows:
        raise ValueError("source row group episode identity is invalid")
    if table["formal_tick"].to_pylist() != list(range(expected_rows)):
        raise ValueError("source row group formal ticks are not contiguous")
    if table["physics_step"].to_pylist() != [10 * tick for tick in range(expected_rows)]:
        raise ValueError("source row group physics ticks are not contiguous")
    if table["time_us"].to_pylist() != [20_000 * tick for tick in range(expected_rows)]:
        raise ValueError("source row group timestamps are not contiguous")
    for field in SOURCE_FRAME_SCHEMA:
        if not field.nullable and table[field.name].null_count:
            raise ValueError(f"nonnull source field contains nulls: {field.name}")
    for field in SOURCE_FRAME_FIELDS:
        if field.shape is None or field.dtype == "image/png":
            continue
        expected_size = math.prod(field.shape)
        for value in table[field.name].to_pylist():
            if value is not None and isinstance(value, list) and len(value) != expected_size:
                raise ValueError(f"source field {field.name} violates its declared shape")
    for name in _TRANSITION_FIELDS:
        values = table[name].to_pylist()
        if values[-1] is not None:
            raise ValueError(f"terminal transition field is not null: {name}")
        if name != "selected_reference_index" and any(value is None for value in values[:-1]):
            raise ValueError(f"nonterminal transition field contains null: {name}")
    if table["outcome_status"][-1].as_py() != "success":
        raise ValueError("source terminal boundary is not successful")
    if any(value != "running" for value in table["outcome_status"].to_pylist()[:-1]):
        raise ValueError("source contains a terminal outcome before its final boundary")
    control_fields = {
        "applied_reference",
        "applied_reference_source_tick",
        "nullspace_joint_position_error",
        "eef_position_error",
        "eef_orientation_error_rotvec",
    }
    for name in control_fields:
        values = table[name].to_pylist()
        if values[0] is not None or any(value is None for value in values[1:]):
            raise ValueError(f"source control-debug nullability is invalid: {name}")
    for camera in ("agentview_rgb", "wrist_rgb"):
        for image in table[camera].to_pylist():
            if not image["bytes"].startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError(f"source {camera} does not contain PNG bytes")
            if not image["path"].endswith(".png"):
                raise ValueError(f"source {camera} path is not PNG")


def _validate_relations(
    *,
    root: Path,
    manifest: SourceCorpusManifest,
    split_plan: MasterTaskSplitPlan | None,
    tasks: pa.Table,
    episodes: pa.Table,
    events: pa.Table,
) -> tuple[dict[str, Any], ...]:
    task_rows = tasks.to_pylist()
    episode_rows = episodes.to_pylist()
    event_rows = events.to_pylist()
    if len(task_rows) != manifest.level_task_instance_count:
        raise ValueError("source task metadata count does not match manifest")
    if len(episode_rows) != manifest.episode_count:
        raise ValueError("source episode metadata count does not match manifest")
    task_by_id: dict[str, dict[str, Any]] = {}
    master_indices: set[int] = set()
    for row in task_rows:
        serialized_id = row["task_instance_id"]
        task_id = TaskInstanceId.from_mapping(json.loads(serialized_id))
        if (
            row["level"] != task_id.level
            or row["task_instance_seed"] != task_id.task_instance_seed
            or row["motion_profile_sha256"] != task_id.motion_profile_sha256
            or row["initial_state_sha256"] != task_id.initial_state_sha256
        ):
            raise ValueError("task metadata identity columns are inconsistent")
        if hashlib.sha256(row["motion_profile_json"].encode()).hexdigest() != (
            task_id.motion_profile_sha256
        ):
            raise ValueError("task motion profile payload hash does not match identity")
        if hashlib.sha256(row["initial_state_npz"]).hexdigest() != task_id.initial_state_sha256:
            raise ValueError("task initial state payload hash does not match identity")
        logical = row["logical_master_task_index"]
        if split_plan is not None and row["split"] != split_plan.split_for(logical):
            raise ValueError("task metadata split join is invalid")
        master_indices.add(logical)
        if serialized_id in task_by_id:
            raise ValueError("task metadata contains duplicate identities")
        task_by_id[serialized_id] = row
    if master_indices != set(manifest.admitted_master_task_indices):
        raise ValueError("admitted master-task inventory does not match task metadata")
    episode_ids = [row["episode_id"] for row in episode_rows]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("episode metadata contains duplicate identities")
    by_level: dict[str, int] = {}
    locations: dict[str, list[dict[str, Any]]] = {}
    for row in episode_rows:
        task = task_by_id.get(row["task_instance_id"])
        if task is None:
            raise ValueError("episode metadata has no task join")
        if row["logical_master_task_index"] != task["logical_master_task_index"] or (
            row["level"] != task["level"]
        ):
            raise ValueError("episode metadata task join is invalid")
        if split_plan is not None and row["split"] != task["split"]:
            raise ValueError("episode metadata split join is invalid")
        try:
            strategy = json.loads(row["strategy_parameters_json"])
            qualification = json.loads(row["qualification_json"])
        except json.JSONDecodeError as error:
            raise ValueError("episode metadata contains invalid JSON payload") from error
        if (
            type(strategy) is not dict
            or type(qualification) is not dict
            or qualification.get("eligible") is not True
            or qualification.get("failures") != []
        ):
            raise ValueError("episode metadata qualification is not an admitted success")
        shard = row["data_shard"]
        if shard not in manifest.artifacts or _SHARD.fullmatch(shard) is None:
            raise ValueError("episode metadata references an invalid data shard")
        locations.setdefault(shard, []).append(row)
        level = str(row["level"])
        by_level[level] = by_level.get(level, 0) + 1
    if by_level != dict(manifest.episodes_by_level):
        raise ValueError("episode level counts do not match manifest")
    if sum(row["frame_count"] for row in episode_rows) != manifest.frame_count:
        raise ValueError("episode frame counts do not match manifest")
    episode_counts = {}
    for row in episode_rows:
        episode_counts[row["task_instance_id"]] = episode_counts.get(row["task_instance_id"], 0) + 1
    if any(
        episode_counts.get(task_id, 0) != row["admitted_realization_count"]
        for task_id, row in task_by_id.items()
    ):
        raise ValueError("task admitted realization count does not match episode metadata")
    for shard, rows in locations.items():
        parquet = pq.ParquetFile(root / shard)
        if parquet.schema_arrow != SOURCE_FRAME_SCHEMA:
            raise ValueError("source data shard schema is invalid")
        ordered = sorted(rows, key=lambda row: row["row_group_index"])
        if parquet.metadata.num_row_groups != len(ordered):
            raise ValueError("source data shard row group count is invalid")
        offset = 0
        for index, row in enumerate(ordered):
            if row["row_group_index"] != index or row["row_offset"] != offset:
                raise ValueError("episode metadata row group location is invalid")
            table = parquet.read_row_group(index)
            _validate_frame_group(table, row)
            if row["frame_count"] != table.num_rows:
                raise ValueError("episode frame count does not match row group")
            offset += table.num_rows
    events_by_episode: dict[str, list[dict[str, Any]]] = {}
    for row in event_rows:
        try:
            payload = json.loads(row["payload_json"])
        except json.JSONDecodeError as error:
            raise ValueError("event metadata contains invalid payload JSON") from error
        if type(payload) is not dict:
            raise ValueError("event payload must decode to a mapping")
        events_by_episode.setdefault(row["episode_id"], []).append(row)
    if set(events_by_episode) != set(episode_ids):
        raise ValueError("event metadata episode join is invalid")
    required_events = {
        "first_contact",
        "stable_grasp",
        "handoff",
        "lift_threshold",
        "success",
    }
    episode_by_id = {row["episode_id"]: row for row in episode_rows}
    for episode_id, rows in events_by_episode.items():
        ordered = sorted(rows, key=lambda row: row["event_index"])
        if [row["event_index"] for row in ordered] != list(range(len(ordered))):
            raise ValueError("event metadata indices are not contiguous")
        kinds = {row["kind"] for row in ordered}
        if len(ordered) != len(required_events) or kinds != required_events:
            raise ValueError("event metadata inventory is invalid")
        success = next(row for row in ordered if row["kind"] == "success")
        if success["time_us"] != episode_by_id[episode_id]["success_time_us"]:
            raise ValueError("event success time does not match episode metadata")
    return tuple(episode_rows)


def load_verified_source_corpus(root: Path) -> VerifiedSourceCorpus:
    root = _check_final_root(Path(root))
    manifest = SourceCorpusManifest.from_mapping(
        _load_json(root / "manifest.json", name="source manifest")
    )
    _verify_inventory(root, manifest)
    expected_schema = (
        legacy_split_source_schema_document()
        if manifest.legacy_split
        else source_schema_document()
    )
    if _load_json(root / "schema.json", name="source schema") != expected_schema:
        raise ValueError("source schema document does not match the canonical schema")
    provenance = _load_json(root / "provenance.json", name="source provenance")
    required_provenance = {"formal_request", "source_config"}
    if manifest.legacy_split:
        required_provenance.add("split_plan")
    for field in required_provenance:
        if field not in provenance:
            raise ValueError(f"source provenance is missing {field}")
    request = FormalRequestUniverse.from_mapping(provenance["formal_request"])
    split_plan = None
    if manifest.legacy_split:
        legacy_config = provenance["source_config"]
        payload = json.dumps(
            legacy_config, sort_keys=True, separators=(",", ":")
        ).encode()
        if (
            type(legacy_config) is not dict
            or legacy_config.get("format_id") != "structured_expert_source_parquet_v1"
            or legacy_config.get("split_unit") != "master_task_index"
            or hashlib.sha256(payload).hexdigest() != manifest.source_config_sha256
        ):
            raise ValueError("legacy source storage config does not match manifest")
        split_plan = MasterTaskSplitPlan.from_mapping(provenance["split_plan"])
    else:
        source_config = SourceCorpusConfig.from_mapping(provenance["source_config"])
        if source_config.sha256 != manifest.source_config_sha256:
            raise ValueError("source storage config does not match manifest")
    if request.request_sha256 != manifest.request_sha256:
        raise ValueError("source request identity does not match manifest")
    if split_plan is not None and split_plan.sha256 != manifest.split_plan_sha256:
        raise ValueError("source split plan does not match manifest")
    if request.config.corpus_id != manifest.corpus_id or (
        split_plan is not None and split_plan.corpus_id != manifest.corpus_id
    ):
        raise ValueError("source corpus identity is inconsistent")
    requested_indices = tuple(
        task.logical_task_index for task in request.primary_tasks + request.reserve_tasks
    )
    if split_plan is not None:
        split_plan.require_exact_indices(requested_indices)
    if manifest.master_task_count != request.config.task_instance_count:
        raise ValueError("source master-task count does not match formal request")
    expected_task_instances = manifest.master_task_count * len(request.config.levels)
    expected_episodes = expected_task_instances * request.config.realizations_per_task
    if manifest.level_task_instance_count != expected_task_instances:
        raise ValueError("source level-task count does not match formal request")
    if manifest.episode_count != expected_episodes:
        raise ValueError("source episode count does not match formal request")
    summary = _load_json(root / "meta/collection_summary.json", name="collection summary")
    if (
        summary.get("format_id") != "structured_expert_collection_summary_v1"
        or summary.get("admitted_realizations") != manifest.episode_count
    ):
        raise ValueError("collection summary does not match source manifest")
    tasks, episodes, events = _load_metadata_tables(
        root, legacy_split=manifest.legacy_split
    )
    episode_rows = _validate_relations(
        root=root,
        manifest=manifest,
        split_plan=split_plan,
        tasks=tasks,
        episodes=episodes,
        events=events,
    )
    return VerifiedSourceCorpus(
        root=root,
        manifest=manifest,
        episode_rows=episode_rows,
    )
