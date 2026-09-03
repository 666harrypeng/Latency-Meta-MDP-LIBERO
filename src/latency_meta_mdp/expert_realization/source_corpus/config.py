"""Strict storage and derived-split contracts for the formal source corpus."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

_MIB = 1024 * 1024


def _strict_mapping(value: Any, expected: set[str], *, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be a YAML mapping")
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown:
        raise ValueError(f"unknown {name} fields: {unknown}")
    if missing:
        raise ValueError(f"missing {name} fields: {missing}")
    return value


def _normalized_text(value: Any, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty normalized string")
    return value


def _integer(value: Any, *, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    return value


def _load_yaml(path: Path, expected: set[str], *, name: str) -> dict[str, Any]:
    path = Path(path)
    return _strict_mapping(
        yaml.safe_load(path.read_text(encoding="utf-8")),
        expected,
        name=name,
    )


@dataclass(frozen=True)
class SourceCorpusConfig:
    schema_version: int
    format_id: str
    image_encoding: str
    png_compress_level: int
    parquet_compression: str
    parquet_compression_level: int
    target_shard_bytes: int
    episode_row_group: bool

    def __post_init__(self) -> None:
        for name in (
            "schema_version",
            "png_compress_level",
            "parquet_compression_level",
            "target_shard_bytes",
        ):
            _integer(getattr(self, name), name=name)
        if type(self.episode_row_group) is not bool:
            raise TypeError("episode_row_group must be a boolean")
        for name in (
            "format_id",
            "image_encoding",
            "parquet_compression",
        ):
            _normalized_text(getattr(self, name), name=name)
        if self.schema_version not in (2, 3):
            raise ValueError("schema_version must equal 2 or 3")
        expected_format = f"structured_expert_source_parquet_v{self.schema_version}"
        if self.format_id != expected_format:
            raise ValueError(f"format_id must equal {expected_format}")
        if self.image_encoding != "lossless_png":
            raise ValueError("image_encoding must equal lossless_png")
        if not 0 <= self.png_compress_level <= 9:
            raise ValueError("png_compress_level must be in [0, 9]")
        if self.parquet_compression != "zstd":
            raise ValueError("parquet_compression must equal zstd")
        if not 1 <= self.parquet_compression_level <= 22:
            raise ValueError("parquet_compression_level must be in [1, 22]")
        if not 64 * _MIB <= self.target_shard_bytes <= 1024 * _MIB:
            raise ValueError("target_shard_bytes must be in [64 MiB, 1 GiB]")
        if self.episode_row_group is not True:
            raise ValueError("episode_row_group must be true")

    def to_mapping(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.to_mapping(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def from_mapping(cls, mapping: Any) -> SourceCorpusConfig:
        raw = _strict_mapping(
            mapping,
            {item.name for item in fields(cls)},
            name="source corpus",
        )
        return cls(**raw)


@dataclass(frozen=True)
class MasterTaskSplitPlan:
    schema_version: int
    split_id: str
    corpus_id: str
    train_master_task_indices: tuple[int, ...]
    validation_master_task_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        _integer(self.schema_version, name="schema_version")
        _normalized_text(self.split_id, name="split_id")
        _normalized_text(self.corpus_id, name="corpus_id")
        if self.schema_version != 1:
            raise ValueError("schema_version must equal 1")
        for name in ("train_master_task_indices", "validation_master_task_indices"):
            values = getattr(self, name)
            if type(values) is not tuple or not values:
                raise ValueError(f"{name} must be a non-empty tuple")
            if any(type(value) is not int or value < 0 for value in values):
                raise ValueError(f"{name} values must be non-negative integers")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} values must be unique")
            if tuple(sorted(values)) != values:
                raise ValueError(f"{name} values must be sorted")
        if set(self.train_master_task_indices) & set(self.validation_master_task_indices):
            raise ValueError("train and validation master-task indices must be disjoint")

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.to_mapping(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def split_for(self, logical_task_index: int) -> str:
        _integer(logical_task_index, name="logical_task_index")
        if logical_task_index in self.train_master_task_indices:
            return "train"
        if logical_task_index in self.validation_master_task_indices:
            return "validation"
        raise KeyError(f"master task {logical_task_index} is absent from the split plan")

    def require_exact_indices(self, logical_task_indices: tuple[int, ...]) -> None:
        if type(logical_task_indices) is not tuple or any(
            type(value) is not int or value < 0 for value in logical_task_indices
        ):
            raise TypeError("logical_task_indices must be a tuple of non-negative integers")
        declared = self.train_master_task_indices + self.validation_master_task_indices
        if len(set(logical_task_indices)) != len(logical_task_indices) or set(declared) != set(
            logical_task_indices
        ):
            raise ValueError("split plan must cover the exact requested master-task universe")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "split_id": self.split_id,
            "corpus_id": self.corpus_id,
            "train_master_task_indices": list(self.train_master_task_indices),
            "validation_master_task_indices": list(self.validation_master_task_indices),
        }

    @classmethod
    def from_mapping(cls, mapping: Any) -> MasterTaskSplitPlan:
        raw = _strict_mapping(
            mapping,
            {item.name for item in fields(cls)},
            name="master task split",
        ).copy()
        for name in ("train_master_task_indices", "validation_master_task_indices"):
            if type(raw[name]) is not list:
                raise TypeError(f"{name} must be a YAML list")
            raw[name] = tuple(raw[name])
        return cls(**raw)


@dataclass(frozen=True)
class SourceExecutionConfig:
    schema_version: int
    execution_id: str
    planner_candidates_per_draw: int
    maximum_realization_draws_per_level: int
    infrastructure_retry_limit: int
    determinism_canaries_per_level: int
    maximum_formal_ticks: int
    admission_unit: str

    def __post_init__(self) -> None:
        for name in (
            "schema_version",
            "planner_candidates_per_draw",
            "maximum_realization_draws_per_level",
            "infrastructure_retry_limit",
            "determinism_canaries_per_level",
            "maximum_formal_ticks",
        ):
            _integer(getattr(self, name), name=name)
        for name in ("execution_id", "admission_unit"):
            _normalized_text(getattr(self, name), name=name)
        if self.schema_version != 2:
            raise ValueError("schema_version must equal 2")
        if self.execution_id != "panda-ball-formal-source-success-quota-v1":
            raise ValueError("execution_id is not the reviewed formal source execution")
        if self.planner_candidates_per_draw != 1:
            raise ValueError("planner_candidates_per_draw must equal 1")
        if self.maximum_realization_draws_per_level != 16:
            raise ValueError("maximum_realization_draws_per_level must equal 16")
        if self.infrastructure_retry_limit != 2:
            raise ValueError("infrastructure_retry_limit must equal 2")
        if self.determinism_canaries_per_level != 1:
            raise ValueError("determinism_canaries_per_level must equal 1")
        if self.maximum_formal_ticks != 220:
            raise ValueError("maximum_formal_ticks must equal 220")
        if self.admission_unit != "paired_master_block":
            raise ValueError("admission_unit must equal paired_master_block")

    def to_mapping(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.to_mapping(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def from_mapping(cls, mapping: Any) -> SourceExecutionConfig:
        raw = _strict_mapping(
            mapping,
            {item.name for item in fields(cls)},
            name="source execution",
        )
        return cls(**raw)


def load_source_corpus_config(path: Path) -> SourceCorpusConfig:
    expected = {item.name for item in fields(SourceCorpusConfig)}
    return SourceCorpusConfig.from_mapping(_load_yaml(path, expected, name="source corpus"))


def load_master_task_split_plan(path: Path) -> MasterTaskSplitPlan:
    expected = {item.name for item in fields(MasterTaskSplitPlan)}
    return MasterTaskSplitPlan.from_mapping(
        _load_yaml(path, expected, name="master task split")
    )


def load_source_execution_config(path: Path) -> SourceExecutionConfig:
    expected = {item.name for item in fields(SourceExecutionConfig)}
    return SourceExecutionConfig.from_mapping(
        _load_yaml(path, expected, name="source execution")
    )
