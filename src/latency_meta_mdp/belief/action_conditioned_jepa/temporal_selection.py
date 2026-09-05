"""Grouped folds and one shared launch-index artifact for temporal selection."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from latency_meta_mdp.belief.action_conditioned_jepa.temporal_view import SharedJepaSampleIndex
from latency_meta_mdp.expert_realization.artifacts import (
    _fsync_directory,
    _hash_file,
    _rename_noreplace,
    _write_file_fsynced,
)

_FORMAT_ID = "action_conditioned_jepa_temporal_selection_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class GroupedMasterFold:
    fold_index: int
    fit_master_indices: tuple[int, ...]
    development_master_indices: tuple[int, ...]


@dataclass(frozen=True)
class TemporalSelectionFold:
    fold_index: int
    fit_master_indices: tuple[int, ...]
    development_master_indices: tuple[int, ...]
    fit_episode_ids: tuple[str, ...]
    development_episode_ids: tuple[str, ...]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "fold_index": self.fold_index,
            "fit_master_indices": list(self.fit_master_indices),
            "development_master_indices": list(self.development_master_indices),
            "fit_episode_ids": list(self.fit_episode_ids),
            "development_episode_ids": list(self.development_episode_ids),
        }


@dataclass(frozen=True)
class LoadedTemporalSelection:
    manifest: dict[str, Any]
    indices: tuple[SharedJepaSampleIndex, ...]
    folds: tuple[TemporalSelectionFold, ...]


def _ordered_unique_ints(values: tuple[int, ...], *, name: str) -> tuple[int, ...]:
    if (
        type(values) is not tuple
        or not values
        or any(type(value) is not int or value < 0 for value in values)
        or values != tuple(sorted(set(values)))
    ):
        raise ValueError(f"{name} must be a non-empty sorted unique tuple")
    return values


def _ordered_unique_ids(values: tuple[str, ...], *, name: str) -> tuple[str, ...]:
    if (
        type(values) is not tuple
        or not values
        or any(type(value) is not str or not value for value in values)
        or values != tuple(sorted(set(values)))
    ):
        raise ValueError(f"{name} must be a non-empty sorted unique tuple")
    return values


def build_grouped_master_folds(
    *,
    master_indices: tuple[int, ...],
    fold_count: int,
    fold_seed: int,
) -> tuple[GroupedMasterFold, ...]:
    masters = _ordered_unique_ints(master_indices, name="master_indices")
    if type(fold_count) is not int or fold_count < 2 or len(masters) % fold_count:
        raise ValueError("fold_count must evenly divide the master-task inventory")
    if type(fold_seed) is not int or fold_seed < 0:
        raise ValueError("fold_seed must be a nonnegative integer")
    ranked = tuple(
        sorted(
            masters,
            key=lambda master: hashlib.sha256(f"{fold_seed}:{master}".encode()).hexdigest(),
        )
    )
    development = tuple(
        tuple(sorted(ranked[offset::fold_count])) for offset in range(fold_count)
    )
    return tuple(
        GroupedMasterFold(
            fold_index=fold_index,
            fit_master_indices=tuple(sorted(set(masters) - set(development_masters))),
            development_master_indices=development_masters,
        )
        for fold_index, development_masters in enumerate(development)
    )


def _selection_folds(
    *,
    master_indices: tuple[int, ...],
    episode_master_indices: dict[str, int],
    fold_count: int,
    fold_seed: int,
) -> tuple[TemporalSelectionFold, ...]:
    grouped = build_grouped_master_folds(
        master_indices=master_indices,
        fold_count=fold_count,
        fold_seed=fold_seed,
    )

    def episode_ids(selected: tuple[int, ...]) -> tuple[str, ...]:
        admitted = set(selected)
        return tuple(
            sorted(
                episode_id
                for episode_id, master in episode_master_indices.items()
                if master in admitted
            )
        )

    return tuple(
        TemporalSelectionFold(
            fold_index=fold.fold_index,
            fit_master_indices=fold.fit_master_indices,
            development_master_indices=fold.development_master_indices,
            fit_episode_ids=episode_ids(fold.fit_master_indices),
            development_episode_ids=episode_ids(fold.development_master_indices),
        )
        for fold in grouped
    )


def _index_mapping(index: SharedJepaSampleIndex) -> dict[str, Any]:
    return {
        "level": index.level,
        "split": index.split,
        "episode_id": index.episode_id,
        "source_tick": index.source_tick,
        "boundary_disposition": index.boundary_disposition,
    }


def write_temporal_selection_artifact(
    *,
    output_dir: Path,
    selection_id: str,
    indices: tuple[SharedJepaSampleIndex, ...],
    episode_master_indices: dict[str, int],
    train_pool_master_indices: tuple[int, ...],
    candidate_config_ids: tuple[str, ...],
    fold_count: int,
    fold_seed: int,
    source_manifest_sha256: str,
    cache_manifest_sha256: str,
    split_manifest_sha256: str,
) -> Path:
    if type(selection_id) is not str or not selection_id or selection_id.strip() != selection_id:
        raise ValueError("selection_id must be a normalized non-empty string")
    if (
        type(indices) is not tuple
        or not indices
        or any(not isinstance(value, SharedJepaSampleIndex) for value in indices)
        or indices != tuple(sorted(set(indices)))
        or any(value.level != 3 or value.split != "train" for value in indices)
    ):
        raise ValueError("indices must be a sorted unique L3 train-pool tuple")
    masters = _ordered_unique_ints(
        train_pool_master_indices,
        name="train_pool_master_indices",
    )
    candidates = _ordered_unique_ids(
        tuple(sorted(candidate_config_ids)),
        name="candidate_config_ids",
    )
    episode_ids = {index.episode_id for index in indices}
    if (
        type(episode_master_indices) is not dict
        or set(episode_master_indices) != episode_ids
        or any(
            type(value) is not int or value not in masters
            for value in episode_master_indices.values()
        )
    ):
        raise ValueError("episode_master_indices must cover the shared index exactly")
    provenance = {}
    for name, value in (
        ("source_manifest_sha256", source_manifest_sha256),
        ("cache_manifest_sha256", cache_manifest_sha256),
        ("split_manifest_sha256", split_manifest_sha256),
    ):
        if type(value) is not str or _SHA256.fullmatch(value) is None:
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        provenance[name] = value
    folds = _selection_folds(
        master_indices=masters,
        episode_master_indices=episode_master_indices,
        fold_count=fold_count,
        fold_seed=fold_seed,
    )
    index_payload = "".join(
        json.dumps(_index_mapping(index), sort_keys=True, separators=(",", ":")) + "\n"
        for index in indices
    ).encode()
    target = Path(output_dir).absolute()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f".{target.name}.building-{os.getpid()}-{uuid.uuid4().hex}"
    building.mkdir()
    try:
        index_path = building / "indices.jsonl"
        _write_file_fsynced(index_path, index_payload)
        manifest = {
            "schema_version": 1,
            "format_id": _FORMAT_ID,
            "selection_id": selection_id,
            "level": 3,
            **provenance,
            "candidate_config_ids": list(candidates),
            "fold_count": fold_count,
            "fold_seed": fold_seed,
            "train_pool_master_indices": list(masters),
            "shared_context_count": len(indices),
            "recorded_complete_count": sum(
                value.boundary_disposition == "recorded_complete" for value in indices
            ),
            "certified_absorbing_extension_count": sum(
                value.boundary_disposition == "certified_absorbing_extension"
                for value in indices
            ),
            "folds": [fold.to_mapping() for fold in folds],
            "artifacts": {
                "indices.jsonl": {
                    "bytes": index_path.stat().st_size,
                    "sha256": _hash_file(index_path),
                }
            },
        }
        _write_file_fsynced(
            building / "manifest.json",
            (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
        )
        _fsync_directory(building)
        _rename_noreplace(building, target)
        _fsync_directory(target.parent)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return target / "manifest.json"


def _load_fold(value: Any) -> TemporalSelectionFold:
    expected = {
        "fold_index",
        "fit_master_indices",
        "development_master_indices",
        "fit_episode_ids",
        "development_episode_ids",
    }
    if type(value) is not dict or set(value) != expected:
        raise ValueError("temporal selection fold fields are invalid")
    return TemporalSelectionFold(
        fold_index=value["fold_index"],
        fit_master_indices=tuple(value["fit_master_indices"]),
        development_master_indices=tuple(value["development_master_indices"]),
        fit_episode_ids=tuple(value["fit_episode_ids"]),
        development_episode_ids=tuple(value["development_episode_ids"]),
    )


def load_temporal_selection_artifact(output_dir: Path) -> LoadedTemporalSelection:
    root = Path(output_dir).resolve()
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, FileNotFoundError) as error:
        raise ValueError("temporal selection manifest is invalid") from error
    expected = {
        "schema_version",
        "format_id",
        "selection_id",
        "level",
        "source_manifest_sha256",
        "cache_manifest_sha256",
        "split_manifest_sha256",
        "candidate_config_ids",
        "fold_count",
        "fold_seed",
        "train_pool_master_indices",
        "shared_context_count",
        "recorded_complete_count",
        "certified_absorbing_extension_count",
        "folds",
        "artifacts",
    }
    if (
        type(manifest) is not dict
        or set(manifest) != expected
        or manifest["schema_version"] != 1
        or manifest["format_id"] != _FORMAT_ID
        or manifest["level"] != 3
    ):
        raise ValueError("temporal selection manifest fields are invalid")
    metadata = manifest["artifacts"]
    index_path = root / "indices.jsonl"
    if (
        type(metadata) is not dict
        or set(metadata) != {"indices.jsonl"}
        or type(metadata["indices.jsonl"]) is not dict
        or set(metadata["indices.jsonl"]) != {"bytes", "sha256"}
        or not index_path.is_file()
        or index_path.stat().st_size != metadata["indices.jsonl"]["bytes"]
        or _hash_file(index_path) != metadata["indices.jsonl"]["sha256"]
    ):
        raise ValueError("temporal selection artifact verification failed")
    indices = []
    try:
        for line in index_path.read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            if type(value) is not dict or set(value) != {
                "level",
                "split",
                "episode_id",
                "source_tick",
                "boundary_disposition",
            }:
                raise ValueError("temporal selection index fields are invalid")
            indices.append(SharedJepaSampleIndex(**value))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("temporal selection index JSON is invalid") from error
    ordered_indices = tuple(indices)
    if (
        ordered_indices != tuple(sorted(set(ordered_indices)))
        or manifest["shared_context_count"] != len(ordered_indices)
        or manifest["recorded_complete_count"]
        != sum(value.boundary_disposition == "recorded_complete" for value in ordered_indices)
        or manifest["certified_absorbing_extension_count"]
        != sum(
            value.boundary_disposition == "certified_absorbing_extension"
            for value in ordered_indices
        )
    ):
        raise ValueError("temporal selection index inventory is invalid")
    folds = tuple(_load_fold(value) for value in manifest["folds"])
    masters = tuple(manifest["train_pool_master_indices"])
    expected_master_folds = build_grouped_master_folds(
        master_indices=masters,
        fold_count=manifest["fold_count"],
        fold_seed=manifest["fold_seed"],
    )
    if len(folds) != len(expected_master_folds) or any(
        (fold.fold_index, fold.fit_master_indices, fold.development_master_indices)
        != (expected.fold_index, expected.fit_master_indices, expected.development_master_indices)
        for fold, expected in zip(folds, expected_master_folds, strict=True)
    ):
        raise ValueError("temporal selection fold inventory is invalid")
    all_episode_ids = {value.episode_id for value in ordered_indices}
    if any(
        set(fold.fit_episode_ids).intersection(fold.development_episode_ids)
        or set(fold.fit_episode_ids).union(fold.development_episode_ids) != all_episode_ids
        for fold in folds
    ):
        raise ValueError("temporal selection fold episode inventory is invalid")
    return LoadedTemporalSelection(manifest=manifest, indices=ordered_indices, folds=folds)
