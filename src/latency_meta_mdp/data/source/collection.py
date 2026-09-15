"""Success-only atomic publication for the formal structured-expert source corpus."""

from __future__ import annotations

import json
import os
import uuid
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from latency_meta_mdp.data.collection.artifacts import (
    _cleanup_owned_staging,
    _fsync_directory,
    _hash_file,
    _rename_noreplace,
    _write_file_fsynced,
)
from latency_meta_mdp.data.collection.contracts import (
    FailureClass,
    FormalRequestUniverse,
    StrategyFamily,
    build_formal_realization_draw_request,
    build_formal_realization_requests,
)
from latency_meta_mdp.data.collection.recording_contracts import json_thaw
from latency_meta_mdp.data.source.config import SourceCorpusConfig
from latency_meta_mdp.data.source.contracts import (
    FormalSourceSynchronizedEpisode,
)
from latency_meta_mdp.data.source.metadata import (
    SourceEpisodeMetadataEntry,
    SourceTaskMetadataEntry,
    build_provenance_document,
    build_schema_document,
    build_task_instance_table,
)
from latency_meta_mdp.data.source.parquet import (
    SourceParquetShardWriter,
)
from latency_meta_mdp.data.source.schema import (
    EPISODE_SCHEMA,
    EVENT_SCHEMA,
)


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


@dataclass(frozen=True)
class CollectionSummary:
    requested_realizations: int
    planned_realizations: int
    executed_attempts: int
    successful_realizations: int
    admitted_realizations: int
    failures_by_class: Mapping[str, int]
    attempted_family_counts: Mapping[str, int] | None = None
    admitted_family_counts: Mapping[str, int] | None = None

    def __post_init__(self) -> None:
        for name in (
            "requested_realizations",
            "planned_realizations",
            "executed_attempts",
            "successful_realizations",
            "admitted_realizations",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.planned_realizations > self.requested_realizations:
            raise ValueError("collection counts must be monotonic within requested work")
        if self.successful_realizations > self.executed_attempts:
            raise ValueError("collection counts must be monotonic within executed work")
        if self.admitted_realizations > self.successful_realizations:
            raise ValueError("collection counts must be monotonic within successful work")
        if not isinstance(self.failures_by_class, Mapping):
            raise TypeError("failures_by_class must be a mapping")
        allowed = {value.value for value in FailureClass}
        failures = dict(self.failures_by_class)
        if any(key not in allowed for key in failures):
            raise ValueError("collection summary contains an unknown failure class")
        if any(type(value) is not int or value < 0 for value in failures.values()):
            raise ValueError("failure class counts must be non-negative integers")
        object.__setattr__(self, "failures_by_class", MappingProxyType(failures))
        family_values = {value.value for value in StrategyFamily}
        family_counts = (self.attempted_family_counts, self.admitted_family_counts)
        if (family_counts[0] is None) != (family_counts[1] is None):
            raise ValueError("quota family counts must be provided together")
        if family_counts[0] is not None:
            frozen = []
            for name, value, expected_total in (
                ("attempted_family_counts", family_counts[0], self.requested_realizations),
                ("admitted_family_counts", family_counts[1], self.admitted_realizations),
            ):
                if not isinstance(value, Mapping):
                    raise TypeError(f"{name} must be a mapping")
                detached = dict(value)
                if any(key not in family_values for key in detached) or any(
                    type(count) is not int or count < 0 for count in detached.values()
                ):
                    raise ValueError(f"{name} contains invalid family counts")
                if sum(detached.values()) != expected_total:
                    raise ValueError(f"{name} does not match its aggregate total")
                frozen.append(MappingProxyType(detached))
            object.__setattr__(self, "attempted_family_counts", frozen[0])
            object.__setattr__(self, "admitted_family_counts", frozen[1])

    def to_mapping(self) -> dict[str, Any]:
        if self.attempted_family_counts is not None:
            return {
                "schema_version": 2,
                "format_id": "structured_expert_quota_collection_summary_v2",
                "semantic_draws_attempted": self.requested_realizations,
                "planner_qualified_draws": self.planned_realizations,
                "rollout_attempts": self.executed_attempts,
                "task_successful_draws": self.successful_realizations,
                "admitted_realizations": self.admitted_realizations,
                "failures_by_class": dict(sorted(self.failures_by_class.items())),
                "attempted_family_counts": dict(sorted(self.attempted_family_counts.items())),
                "admitted_family_counts": dict(sorted(self.admitted_family_counts.items())),
                "draws_per_admitted_realization": (
                    self.requested_realizations / self.admitted_realizations
                ),
            }
        return {
            "schema_version": 1,
            "format_id": "structured_expert_collection_summary_v1",
            "requested_realizations": self.requested_realizations,
            "planned_realizations": self.planned_realizations,
            "executed_attempts": self.executed_attempts,
            "successful_realizations": self.successful_realizations,
            "admitted_realizations": self.admitted_realizations,
            "failures_by_class": dict(sorted(self.failures_by_class.items())),
        }


@dataclass(frozen=True)
class AdmittedSourceEpisode:
    episode: FormalSourceSynchronizedEpisode
    strategy_parameters: Mapping[str, Any]
    selected_planner_fingerprint: str
    qualification: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.episode, FormalSourceSynchronizedEpisode):
            raise TypeError("episode must be FormalSourceSynchronizedEpisode")
        for name in ("strategy_parameters", "qualification"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")
            json.dumps(json_thaw(value), allow_nan=False)


def _validate_task_inventory(
    *,
    request: FormalRequestUniverse,
    task_entries: tuple[SourceTaskMetadataEntry, ...],
) -> tuple[tuple[int, ...], dict[str, SourceTaskMetadataEntry]]:
    if not isinstance(request, FormalRequestUniverse):
        raise TypeError("request must be FormalRequestUniverse")
    all_tasks = request.primary_tasks + request.reserve_tasks
    if type(task_entries) is not tuple or any(
        not isinstance(entry, SourceTaskMetadataEntry) for entry in task_entries
    ):
        raise TypeError("task_entries must be a tuple of SourceTaskMetadataEntry")
    tasks_by_key = {
        (row.logical_master_task_index, row.task_instance_id.level): row for row in task_entries
    }
    if len(tasks_by_key) != len(task_entries):
        raise ValueError("source task metadata contains duplicate task identities")
    complete_blocks = []
    candidate_indices = sorted({logical for logical, _level in tasks_by_key})
    request_tasks = {row.logical_task_index: row for row in all_tasks}
    for logical in candidate_indices:
        if logical not in request_tasks:
            raise ValueError("admitted task is outside formal request")
        complete = True
        for level in request.config.levels:
            task = tasks_by_key.get((logical, level))
            if task is None or (
                task.admitted_realization_count != request.config.realizations_per_task
            ):
                complete = False
                break
            if task.corpus_id != request.config.corpus_id:
                raise ValueError("source task metadata does not match formal request")
            if task.task_instance_id.task_instance_seed != request_tasks[logical].master_task_seed:
                raise ValueError("source task seed does not match formal request")
        if complete:
            complete_blocks.append(logical)
    if len(complete_blocks) != request.config.task_instance_count:
        raise ValueError("publication requires complete admitted master-task blocks")
    admitted_keys = {
        (logical, level) for logical in complete_blocks for level in request.config.levels
    }
    if set(tasks_by_key) != admitted_keys:
        raise ValueError("publication contains rows outside complete admitted master-task blocks")
    return (
        tuple(complete_blocks),
        {row.task_instance_id.canonical_json(): row for row in task_entries},
    )


def _validate_admitted_episode(
    admitted: AdmittedSourceEpisode,
    *,
    request: FormalRequestUniverse,
    source_config: SourceCorpusConfig,
    tasks_by_id: Mapping[str, SourceTaskMetadataEntry],
) -> tuple[int, int, int]:
    if not isinstance(admitted, AdmittedSourceEpisode):
        raise TypeError("admitted episode stream must contain AdmittedSourceEpisode values")
    metadata = admitted.episode.metadata
    logical = metadata.logical_master_task_index
    level = metadata.task_instance_id.level
    if metadata.corpus_id != request.config.corpus_id:
        raise ValueError("source episode corpus does not match formal request")
    if metadata.formal_corpus_config_sha256 != request.corpus_config_sha256:
        raise ValueError("source episode formal config does not match request")
    if metadata.source_corpus_config_sha256 != source_config.sha256:
        raise ValueError("source episode storage config does not match source config")
    task = tasks_by_id.get(metadata.task_instance_id.canonical_json())
    if task is None or metadata.task_instance_id != task.task_instance_id:
        raise ValueError("source episode does not match exact task metadata")
    key = metadata.expert_realization_id.expert_realization_key
    if metadata.schema_version == 3:
        expected_request = build_formal_realization_draw_request(
            request,
            metadata.task_instance_id,
            key.realization_index,
        )
        accepted_slot = metadata.accepted_slot
    else:
        expected_requests = build_formal_realization_requests(request, metadata.task_instance_id)
        if key.realization_index >= len(expected_requests):
            raise ValueError("source realization is outside the formal request")
        expected_request = expected_requests[key.realization_index]
        accepted_slot = key.realization_index
    if key != expected_request.to_expert_realization_key() or (
        metadata.strategy_family is not expected_request.assigned_family
    ):
        raise ValueError("source realization identity does not match formal request")
    if type(accepted_slot) is not int:
        raise ValueError("source accepted slot is invalid")
    return logical, level, accepted_slot


def _write_parquet(path: Path, table: Any, *, config: SourceCorpusConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table,
        path,
        compression=config.parquet_compression,
        compression_level=config.parquet_compression_level,
    )
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def publish_source_corpus(
    *,
    target: Path,
    request: FormalRequestUniverse,
    source_config: SourceCorpusConfig,
    task_entries: tuple[SourceTaskMetadataEntry, ...],
    admitted_episodes: Iterable[AdmittedSourceEpisode],
    collection_summary: CollectionSummary,
) -> Path:
    if not isinstance(collection_summary, CollectionSummary):
        raise TypeError("collection_summary must be CollectionSummary")
    complete_blocks, tasks_by_id = _validate_task_inventory(
        request=request,
        task_entries=task_entries,
    )
    total_requested = (
        len(request.primary_tasks + request.reserve_tasks)
        * len(request.config.levels)
        * request.config.realizations_per_task
    )
    if source_config.schema_version == 2 and (
        collection_summary.requested_realizations != total_requested
    ):
        raise ValueError("collection summary requested count does not match formal universe")
    if source_config.schema_version == 3 and (
        collection_summary.requested_realizations < collection_summary.admitted_realizations
    ):
        raise ValueError("quota collection attempted fewer draws than it admitted")
    ordered_tasks = tuple(
        sorted(
            task_entries,
            key=lambda item: (
                item.logical_master_task_index,
                item.task_instance_id.level,
            ),
        )
    )
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    parent_stat = os.lstat(target.parent)
    building = target.with_name(f".{target.name}.building-{uuid.uuid4().hex}")
    building.mkdir()
    building_stat = os.lstat(building)
    try:
        episode_rows = []
        event_rows = []
        provenance_metadata = []
        published_shards: list[tuple[str, Any]] = []
        episodes_by_level: Counter[int] = Counter()
        slots_by_task: dict[tuple[int, int], list[int]] = {}
        frame_count = 0
        episode_count = 0
        previous_key: tuple[int, int, int] | None = None
        current_level: int | None = None
        shard_index = 0
        shard_relative = ""
        writer: SourceParquetShardWriter | None = None
        for admitted in admitted_episodes:
            logical, level, realization = _validate_admitted_episode(
                admitted,
                request=request,
                source_config=source_config,
                tasks_by_id=tasks_by_id,
            )
            order_key = (level, logical, realization)
            if previous_key is not None and order_key <= previous_key:
                raise ValueError("admitted episode stream must use canonical order")
            previous_key = order_key
            if level != current_level:
                if writer is not None:
                    published_shards.append((shard_relative, writer.close()))
                current_level = level
                shard_index = 0
                writer = None
            if writer is None:
                shard_relative = f"data/level-{level}/shard-{shard_index:05d}.parquet"
                writer = SourceParquetShardWriter(
                    target=building / shard_relative,
                    level=level,
                    config=source_config,
                )
            episode = admitted.episode
            location = writer.add_episode(episode)
            entry = SourceEpisodeMetadataEntry(
                episode=episode,
                logical_master_task_index=logical,
                strategy_parameters=admitted.strategy_parameters,
                selected_planner_fingerprint=admitted.selected_planner_fingerprint,
                qualification=admitted.qualification,
                location=location,
                data_shard=shard_relative,
            )
            episode_rows.append(entry.to_row())
            for event_index, event in enumerate(episode.physical_events):
                event_rows.append(
                    {
                        "episode_id": episode.metadata.episode_id,
                        "event_index": event_index,
                        "kind": event.kind,
                        "physics_step": event.physics_step_index,
                        "time_us": event.time_us,
                        "payload_json": json.dumps(
                            json_thaw(event.payload),
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        ),
                        "terminal_reason": event.terminal_reason,
                    }
                )
            provenance_metadata.append(episode.metadata)
            slots_by_task.setdefault((logical, level), []).append(realization)
            episode_count += 1
            frame_count += len(episode.boundaries)
            episodes_by_level[level] += 1
            if writer.current_byte_count >= source_config.target_shard_bytes:
                published_shards.append((shard_relative, writer.close()))
                writer = None
                shard_index += 1
            del entry, episode, admitted
        if writer is not None:
            published_shards.append((shard_relative, writer.close()))
        expected_slots = list(range(request.config.realizations_per_task))
        expected_task_keys = {
            (logical, level) for logical in complete_blocks for level in request.config.levels
        }
        if set(slots_by_task) != expected_task_keys or any(
            sorted(values) != expected_slots for values in slots_by_task.values()
        ):
            raise ValueError("publication requires complete admitted master-task blocks")
        if collection_summary.admitted_realizations != episode_count:
            raise ValueError("collection summary admitted count does not match source episodes")
        meta = building / "meta"
        _write_parquet(
            meta / "task_instances.parquet",
            build_task_instance_table(ordered_tasks),
            config=source_config,
        )
        _write_parquet(
            meta / "episodes.parquet",
            pa.Table.from_pylist(episode_rows, schema=EPISODE_SCHEMA),
            config=source_config,
        )
        _write_parquet(
            meta / "events.parquet",
            pa.Table.from_pylist(event_rows, schema=EVENT_SCHEMA),
            config=source_config,
        )
        _write_file_fsynced(
            meta / "collection_summary.json",
            _json_bytes(collection_summary.to_mapping()),
        )
        _write_file_fsynced(
            building / "schema.json",
            _json_bytes(build_schema_document()),
        )
        provenance = build_provenance_document(tuple(provenance_metadata))
        provenance.update(
            {
                "formal_request": request.to_mapping(),
                "source_config": source_config.to_mapping(),
            }
        )
        _write_file_fsynced(
            building / "provenance.json",
            _json_bytes(provenance),
        )
        _write_file_fsynced(
            building / "README.md",
            b"# Formal structured-expert source corpus\n",
        )
        artifacts = {}
        for path in sorted(item for item in building.rglob("*") if item.is_file()):
            relative = path.relative_to(building).as_posix()
            artifacts[relative] = {
                "sha256": _hash_file(path),
                "bytes": path.stat().st_size,
            }
        manifest = {
            "schema_version": source_config.schema_version,
            "format_id": f"structured_expert_source_corpus_v{source_config.schema_version}",
            "complete": True,
            "corpus_id": request.config.corpus_id,
            "request_sha256": request.request_sha256,
            "source_config_sha256": source_config.sha256,
            "admitted_master_task_indices": list(complete_blocks),
            "master_task_count": len(complete_blocks),
            "level_task_instance_count": len(ordered_tasks),
            "episode_count": episode_count,
            "episodes_by_level": {
                str(level): episodes_by_level[level] for level in sorted(episodes_by_level)
            },
            "frame_count": frame_count,
            "shard_count": len(published_shards),
            "artifacts": artifacts,
        }
        _write_file_fsynced(building / "manifest.json", _json_bytes(manifest))
        for directory in sorted(
            (item for item in building.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            _fsync_directory(directory)
        _fsync_directory(building)
        _rename_noreplace(building, target)
        _fsync_directory(target.parent)
        return target / "manifest.json"
    except BaseException:
        _cleanup_owned_staging(
            building,
            expected_device=building_stat.st_dev,
            expected_inode=building_stat.st_ino,
            parent=target.parent,
            expected_parent_device=parent_stat.st_dev,
            expected_parent_inode=parent_stat.st_ino,
        )
        raise
