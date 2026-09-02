"""Success-only atomic publication for the formal structured-expert source corpus."""

from __future__ import annotations

import json
import os
import uuid
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pyarrow.parquet as pq

from latency_meta_mdp.expert_realization.artifacts import (
    _cleanup_owned_staging,
    _fsync_directory,
    _hash_file,
    _rename_noreplace,
    _write_file_fsynced,
)
from latency_meta_mdp.expert_realization.contracts import (
    FailureClass,
    FormalRequestUniverse,
    build_formal_realization_requests,
)
from latency_meta_mdp.expert_realization.recording_contracts import json_thaw
from latency_meta_mdp.expert_realization.source_corpus.config import (
    MasterTaskSplitPlan,
    SourceCorpusConfig,
)
from latency_meta_mdp.expert_realization.source_corpus.contracts import (
    FormalSourceSynchronizedEpisode,
)
from latency_meta_mdp.expert_realization.source_corpus.metadata import (
    SourceEpisodeMetadataEntry,
    SourceTaskMetadataEntry,
    build_episode_table,
    build_event_table,
    build_provenance_document,
    build_schema_document,
    build_task_instance_table,
)
from latency_meta_mdp.expert_realization.source_corpus.parquet import (
    SourceParquetShardWriter,
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

    def to_mapping(self) -> dict[str, Any]:
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


def _validate_admitted_inventory(
    *,
    request: FormalRequestUniverse,
    source_config: SourceCorpusConfig,
    split_plan: MasterTaskSplitPlan,
    task_entries: tuple[SourceTaskMetadataEntry, ...],
    admitted_episodes: tuple[AdmittedSourceEpisode, ...],
) -> tuple[int, ...]:
    if not isinstance(request, FormalRequestUniverse):
        raise TypeError("request must be FormalRequestUniverse")
    if not isinstance(source_config, SourceCorpusConfig):
        raise TypeError("source_config must be SourceCorpusConfig")
    if not isinstance(split_plan, MasterTaskSplitPlan):
        raise TypeError("split_plan must be MasterTaskSplitPlan")
    if split_plan.corpus_id != request.config.corpus_id:
        raise ValueError("split plan corpus does not match formal request")
    all_tasks = request.primary_tasks + request.reserve_tasks
    split_plan.require_exact_indices(tuple(task.logical_task_index for task in all_tasks))
    if type(task_entries) is not tuple or any(
        not isinstance(entry, SourceTaskMetadataEntry) for entry in task_entries
    ):
        raise TypeError("task_entries must be a tuple of SourceTaskMetadataEntry")
    if type(admitted_episodes) is not tuple or any(
        not isinstance(entry, AdmittedSourceEpisode) for entry in admitted_episodes
    ):
        raise TypeError("admitted_episodes must be a tuple of AdmittedSourceEpisode")
    tasks_by_key = {
        (row.logical_master_task_index, row.task_instance_id.level): row
        for row in task_entries
    }
    if len(tasks_by_key) != len(task_entries):
        raise ValueError("source task metadata contains duplicate task identities")
    episodes_by_task: dict[tuple[int, int], list[AdmittedSourceEpisode]] = defaultdict(list)
    for admitted in admitted_episodes:
        metadata = admitted.episode.metadata
        logical = metadata.logical_master_task_index
        level = metadata.task_instance_id.level
        if metadata.corpus_id != request.config.corpus_id:
            raise ValueError("source episode corpus does not match formal request")
        if metadata.formal_corpus_config_sha256 != request.corpus_config_sha256:
            raise ValueError("source episode formal config does not match request")
        if metadata.source_corpus_config_sha256 != source_config.sha256:
            raise ValueError("source episode storage config does not match source config")
        if metadata.master_task_split_plan_sha256 != split_plan.sha256:
            raise ValueError("source episode split identity does not match split plan")
        expected_requests = build_formal_realization_requests(
            request,
            metadata.task_instance_id,
        )
        key = metadata.expert_realization_id.expert_realization_key
        if key.realization_index >= len(expected_requests):
            raise ValueError("source realization is outside the formal request")
        expected_request = expected_requests[key.realization_index]
        if key != expected_request.to_expert_realization_key() or (
            metadata.strategy_family is not expected_request.assigned_family
        ):
            raise ValueError("source realization identity does not match formal request")
        task = tasks_by_key.get((logical, level))
        if task is None:
            raise ValueError("source episode has no admitted task metadata")
        if metadata.task_instance_id != task.task_instance_id:
            raise ValueError("source episode does not match exact task metadata")
        if split_plan.split_for(logical) != task.split:
            raise ValueError("source task metadata does not match split plan")
        episodes_by_task[(logical, level)].append(admitted)
    complete_blocks = []
    candidate_indices = sorted({logical for logical, _level in tasks_by_key})
    request_tasks = {row.logical_task_index: row for row in all_tasks}
    for logical in candidate_indices:
        if logical not in request_tasks:
            raise ValueError("admitted task is outside formal request")
        complete = True
        for level in request.config.levels:
            task = tasks_by_key.get((logical, level))
            episodes = episodes_by_task.get((logical, level), [])
            if task is None or (
                task.admitted_realization_count != request.config.realizations_per_task
            ):
                complete = False
                break
            if task.corpus_id != request.config.corpus_id or (
                task.split != split_plan.split_for(logical)
            ):
                raise ValueError("source task metadata does not match split plan")
            if task.task_instance_id.task_instance_seed != request_tasks[logical].master_task_seed:
                raise ValueError("source task seed does not match formal request")
            slots = sorted(
                item.episode.metadata.expert_realization_id.expert_realization_key.realization_index
                for item in episodes
            )
            if slots != list(range(request.config.realizations_per_task)):
                complete = False
                break
        if complete:
            complete_blocks.append(logical)
    if len(complete_blocks) != request.config.task_instance_count:
        raise ValueError("publication requires complete admitted master-task blocks")
    admitted_keys = {
        (logical, level) for logical in complete_blocks for level in request.config.levels
    }
    if set(tasks_by_key) != admitted_keys or set(episodes_by_task) != admitted_keys:
        raise ValueError("publication contains rows outside complete admitted master-task blocks")
    return tuple(complete_blocks)


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
    split_plan: MasterTaskSplitPlan,
    task_entries: tuple[SourceTaskMetadataEntry, ...],
    admitted_episodes: tuple[AdmittedSourceEpisode, ...],
    collection_summary: CollectionSummary,
) -> Path:
    if not isinstance(collection_summary, CollectionSummary):
        raise TypeError("collection_summary must be CollectionSummary")
    complete_blocks = _validate_admitted_inventory(
        request=request,
        source_config=source_config,
        split_plan=split_plan,
        task_entries=task_entries,
        admitted_episodes=admitted_episodes,
    )
    if collection_summary.admitted_realizations != len(admitted_episodes):
        raise ValueError("collection summary admitted count does not match source episodes")
    total_requested = (
        len(request.primary_tasks + request.reserve_tasks)
        * len(request.config.levels)
        * request.config.realizations_per_task
    )
    if collection_summary.requested_realizations != total_requested:
        raise ValueError("collection summary requested count does not match formal universe")
    ordered_tasks = tuple(
        sorted(
            task_entries,
            key=lambda item: (
                item.logical_master_task_index,
                item.task_instance_id.level,
            ),
        )
    )
    ordered_admitted = tuple(
        sorted(
            admitted_episodes,
            key=lambda item: (
                item.episode.metadata.task_instance_id.level,
                item.episode.metadata.logical_master_task_index,
                item.episode.metadata.expert_realization_id.expert_realization_key.realization_index,
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
        indexed_entries = []
        published_shards: list[tuple[str, Any]] = []
        by_level: dict[int, list[AdmittedSourceEpisode]] = defaultdict(list)
        for admitted in ordered_admitted:
            by_level[admitted.episode.metadata.task_instance_id.level].append(admitted)
        for level in sorted(by_level):
            ordered = sorted(
                by_level[level],
                key=lambda item: (
                    item.episode.metadata.logical_master_task_index,
                    item.episode.metadata.expert_realization_id.expert_realization_key.realization_index,
                ),
            )
            shard_index = 0
            writer: SourceParquetShardWriter | None = None
            shard_relative = ""
            for position, admitted in enumerate(ordered):
                if writer is None:
                    shard_relative = f"data/level-{level}/shard-{shard_index:05d}.parquet"
                    writer = SourceParquetShardWriter(
                        target=building / shard_relative,
                        level=level,
                        config=source_config,
                    )
                location = writer.add_episode(admitted.episode)
                split = split_plan.split_for(
                    admitted.episode.metadata.logical_master_task_index
                )
                indexed_entries.append(
                    SourceEpisodeMetadataEntry(
                        episode=admitted.episode,
                        logical_master_task_index=(
                            admitted.episode.metadata.logical_master_task_index
                        ),
                        split=split,
                        strategy_parameters=admitted.strategy_parameters,
                        selected_planner_fingerprint=(
                            admitted.selected_planner_fingerprint
                        ),
                        qualification=admitted.qualification,
                        location=location,
                        data_shard=shard_relative,
                    )
                )
                is_last = position == len(ordered) - 1
                if writer.current_byte_count >= source_config.target_shard_bytes or is_last:
                    published_shards.append((shard_relative, writer.close()))
                    writer = None
                    shard_index += 1
        meta = building / "meta"
        _write_parquet(
            meta / "task_instances.parquet",
            build_task_instance_table(ordered_tasks),
            config=source_config,
        )
        _write_parquet(
            meta / "episodes.parquet",
            build_episode_table(tuple(indexed_entries)),
            config=source_config,
        )
        _write_parquet(
            meta / "events.parquet",
            build_event_table(tuple(item.episode for item in ordered_admitted)),
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
        provenance = build_provenance_document(
            tuple(item.episode.metadata for item in ordered_admitted)
        )
        provenance.update(
            {
                "formal_request": request.to_mapping(),
                "source_config": source_config.to_mapping(),
                "split_plan": split_plan.to_mapping(),
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
        episodes_by_level = Counter(
            item.episode.metadata.task_instance_id.level for item in ordered_admitted
        )
        manifest = {
            "schema_version": 1,
            "format_id": "structured_expert_source_corpus_v1",
            "complete": True,
            "corpus_id": request.config.corpus_id,
            "request_sha256": request.request_sha256,
            "source_config_sha256": source_config.sha256,
            "split_plan_sha256": split_plan.sha256,
            "admitted_master_task_indices": list(complete_blocks),
            "master_task_count": len(complete_blocks),
            "level_task_instance_count": len(ordered_tasks),
            "episode_count": len(ordered_admitted),
            "episodes_by_level": {
                str(level): episodes_by_level[level] for level in sorted(episodes_by_level)
            },
            "frame_count": sum(
                len(item.episode.boundaries) for item in ordered_admitted
            ),
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
