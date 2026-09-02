"""Strict storage and master-task split contracts for the formal source corpus."""

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
    split_unit: str
    source_stats_split: str
    source_stats_quantiles: tuple[float, float]

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
            "split_unit",
            "source_stats_split",
        ):
            _normalized_text(getattr(self, name), name=name)
        if type(self.source_stats_quantiles) is not tuple or any(
            type(value) is not float for value in self.source_stats_quantiles
        ):
            raise TypeError("source_stats_quantiles must be a tuple of floats")
        if self.schema_version != 1:
            raise ValueError("schema_version must equal 1")
        if self.format_id != "structured_expert_source_parquet_v1":
            raise ValueError("format_id must equal structured_expert_source_parquet_v1")
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
        if self.split_unit != "master_task_index":
            raise ValueError("split_unit must equal master_task_index")
        if self.source_stats_split != "train":
            raise ValueError("source_stats_split must equal train")
        if self.source_stats_quantiles != (0.01, 0.99):
            raise ValueError("source_stats_quantiles must equal the canonical quantiles")

    def to_mapping(self) -> dict[str, Any]:
        result = {item.name: getattr(self, item.name) for item in fields(self)}
        result["source_stats_quantiles"] = list(self.source_stats_quantiles)
        return result

    @classmethod
    def from_mapping(cls, mapping: Any) -> SourceCorpusConfig:
        raw = _strict_mapping(
            mapping,
            {item.name for item in fields(cls)},
            name="source corpus",
        ).copy()
        quantiles = raw["source_stats_quantiles"]
        if type(quantiles) is not list:
            raise TypeError("source_stats_quantiles must be a YAML list")
        raw["source_stats_quantiles"] = tuple(quantiles)
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


def load_source_corpus_config(path: Path) -> SourceCorpusConfig:
    expected = {item.name for item in fields(SourceCorpusConfig)}
    return SourceCorpusConfig.from_mapping(_load_yaml(path, expected, name="source corpus"))


def load_master_task_split_plan(path: Path) -> MasterTaskSplitPlan:
    expected = {item.name for item in fields(MasterTaskSplitPlan)}
    return MasterTaskSplitPlan.from_mapping(
        _load_yaml(path, expected, name="master task split")
    )
