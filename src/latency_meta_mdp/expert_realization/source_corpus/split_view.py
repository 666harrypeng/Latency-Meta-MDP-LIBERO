"""Derived master-task splits bound to one immutable source corpus."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from latency_meta_mdp.expert_realization.artifacts import (
    _fsync_directory,
    _rename_noreplace,
    _write_file_fsynced,
)
from latency_meta_mdp.expert_realization.source_corpus.loader import VerifiedSourceCorpus

_FORMAT_ID = "structured_expert_source_split_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _strict(value: Any, expected: set[str], *, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be a mapping")
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown or missing:
        raise ValueError(f"{name} fields mismatch; unknown={unknown}, missing={missing}")
    return value


def _ordered_nonnegative(values: Any, *, name: str) -> tuple[int, ...]:
    if type(values) is not list or any(type(value) is not int or value < 0 for value in values):
        raise TypeError(f"{name} must be a JSON list of non-negative integers")
    result = tuple(values)
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{name} must be sorted and unique")
    return result


def _ordered_ids(values: Any, *, name: str) -> tuple[str, ...]:
    if type(values) is not list or any(type(value) is not str or not value for value in values):
        raise TypeError(f"{name} must be a JSON list of episode IDs")
    result = tuple(values)
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{name} must be sorted and unique")
    return result


@dataclass(frozen=True)
class SourceSplitManifest:
    split_id: str
    source_corpus_id: str
    source_manifest_sha256: str
    split_seed: int
    train_master_task_indices: tuple[int, ...]
    validation_master_task_indices: tuple[int, ...]
    train_episode_ids: tuple[str, ...]
    validation_episode_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("split_id", "source_corpus_id"):
            value = getattr(self, name)
            if type(value) is not str or not value or value.strip() != value:
                raise ValueError(f"{name} must be a non-empty normalized string")
        if _SHA256.fullmatch(self.source_manifest_sha256) is None:
            raise ValueError("source_manifest_sha256 must be a SHA-256 digest")
        if type(self.split_seed) is not int or self.split_seed < 0:
            raise ValueError("split_seed must be a non-negative integer")
        for name in ("train_master_task_indices", "validation_master_task_indices"):
            values = getattr(self, name)
            if type(values) is not tuple or not values or values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be a non-empty sorted unique tuple")
        if set(self.train_master_task_indices) & set(self.validation_master_task_indices):
            raise ValueError("train and validation master tasks must be disjoint")
        for name in ("train_episode_ids", "validation_episode_ids"):
            values = getattr(self, name)
            if type(values) is not tuple or not values or values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be a non-empty sorted unique tuple")
        if set(self.train_episode_ids) & set(self.validation_episode_ids):
            raise ValueError("train and validation episodes must be disjoint")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "format_id": _FORMAT_ID,
            "split_id": self.split_id,
            "source_corpus_id": self.source_corpus_id,
            "source_manifest_sha256": self.source_manifest_sha256,
            "split_seed": self.split_seed,
            "train_master_task_indices": list(self.train_master_task_indices),
            "validation_master_task_indices": list(self.validation_master_task_indices),
            "train_episode_ids": list(self.train_episode_ids),
            "validation_episode_ids": list(self.validation_episode_ids),
        }

    @classmethod
    def from_mapping(cls, mapping: Any) -> SourceSplitManifest:
        raw = _strict(
            mapping,
            {
                "schema_version",
                "format_id",
                "split_id",
                "source_corpus_id",
                "source_manifest_sha256",
                "split_seed",
                "train_master_task_indices",
                "validation_master_task_indices",
                "train_episode_ids",
                "validation_episode_ids",
            },
            name=cls.__name__,
        )
        if raw["schema_version"] != 1 or raw["format_id"] != _FORMAT_ID:
            raise ValueError("unsupported source split format")
        return cls(
            split_id=raw["split_id"],
            source_corpus_id=raw["source_corpus_id"],
            source_manifest_sha256=raw["source_manifest_sha256"],
            split_seed=raw["split_seed"],
            train_master_task_indices=_ordered_nonnegative(
                raw["train_master_task_indices"], name="train_master_task_indices"
            ),
            validation_master_task_indices=_ordered_nonnegative(
                raw["validation_master_task_indices"],
                name="validation_master_task_indices",
            ),
            train_episode_ids=_ordered_ids(raw["train_episode_ids"], name="train_episode_ids"),
            validation_episode_ids=_ordered_ids(
                raw["validation_episode_ids"], name="validation_episode_ids"
            ),
        )


def _source_manifest_sha256(corpus: VerifiedSourceCorpus) -> str:
    return hashlib.sha256((corpus.root / "manifest.json").read_bytes()).hexdigest()


def _episode_inventory(corpus: VerifiedSourceCorpus) -> dict[int, tuple[dict[str, Any], ...]]:
    if not isinstance(corpus, VerifiedSourceCorpus):
        raise TypeError("corpus must be a VerifiedSourceCorpus")
    rows = tuple(dict(value) for value in corpus._episodes.values())
    if len(rows) != corpus.manifest.episode_count:
        raise ValueError("source does not contain complete master-task blocks")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["logical_master_task_index"]].append(row)
    expected_keys = {(level, realization) for level in (1, 2, 3) for realization in range(4)}
    if set(grouped) != set(corpus.manifest.admitted_master_task_indices):
        raise ValueError("source does not contain complete master-task blocks")
    for values in grouped.values():
        keys = {(row["level"], row["realization_index"]) for row in values}
        if len(values) != 12 or keys != expected_keys:
            raise ValueError("source does not contain complete master-task blocks")
    return {
        master: tuple(
            sorted(
                values,
                key=lambda row: (row["level"], row["realization_index"]),
            )
        )
        for master, values in grouped.items()
    }


def build_source_split(
    corpus: VerifiedSourceCorpus,
    *,
    split_id: str,
    validation_master_count: int,
    split_seed: int,
) -> SourceSplitManifest:
    inventory = _episode_inventory(corpus)
    if type(validation_master_count) is not int or not 0 < validation_master_count < len(inventory):
        raise ValueError("validation_master_count must leave non-empty train and validation sets")
    if type(split_seed) is not int or split_seed < 0:
        raise ValueError("split_seed must be a non-negative integer")
    ranked = sorted(
        inventory,
        key=lambda master: hashlib.sha256(f"{split_seed}:{master}".encode()).hexdigest(),
    )
    validation = tuple(sorted(ranked[:validation_master_count]))
    train = tuple(sorted(set(inventory) - set(validation)))

    def episode_ids(masters: tuple[int, ...]) -> tuple[str, ...]:
        return tuple(
            sorted(row["episode_id"] for master in masters for row in inventory[master])
        )

    return SourceSplitManifest(
        split_id=split_id,
        source_corpus_id=corpus.manifest.corpus_id,
        source_manifest_sha256=_source_manifest_sha256(corpus),
        split_seed=split_seed,
        train_master_task_indices=train,
        validation_master_task_indices=validation,
        train_episode_ids=episode_ids(train),
        validation_episode_ids=episode_ids(validation),
    )


def write_source_split(path: Path, manifest: SourceSplitManifest) -> Path:
    if not isinstance(manifest, SourceSplitManifest):
        raise TypeError("manifest must be a SourceSplitManifest")
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(manifest.to_mapping(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    staging = path.parent / f".{path.name}.building-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        _write_file_fsynced(staging, payload)
        _rename_noreplace(staging, path)
        _fsync_directory(path.parent)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    return path


def load_verified_source_split(
    path: Path, corpus: VerifiedSourceCorpus
) -> SourceSplitManifest:
    try:
        mapping = json.loads(Path(path).read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("source split must contain valid JSON") from error
    manifest = SourceSplitManifest.from_mapping(mapping)
    if (
        manifest.source_corpus_id != corpus.manifest.corpus_id
        or manifest.source_manifest_sha256 != _source_manifest_sha256(corpus)
    ):
        raise ValueError("source split does not match source manifest")
    expected = build_source_split(
        corpus,
        split_id=manifest.split_id,
        validation_master_count=len(manifest.validation_master_task_indices),
        split_seed=manifest.split_seed,
    )
    if manifest != expected:
        raise ValueError("source split episode inventory is invalid")
    return manifest
