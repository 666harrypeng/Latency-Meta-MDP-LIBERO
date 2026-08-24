"""Explicit episode-level seed split plans for model training and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class EpisodeSeedRange:
    name: str
    start: int
    count: int

    def __post_init__(self) -> None:
        if (
            self.name not in {"train", "validation", "holdout"}
            or isinstance(self.start, bool)
            or not isinstance(self.start, int)
            or self.start < 0
            or isinstance(self.count, bool)
            or not isinstance(self.count, int)
            or self.count <= 0
        ):
            raise ValueError("episode split range is invalid")

    @property
    def stop(self) -> int:
        return self.start + self.count


@dataclass(frozen=True)
class EpisodeSplitPlan:
    schema_version: int
    split_id: str
    source_seed_start: int
    source_seed_count: int
    ranges: tuple[EpisodeSeedRange, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or not self.split_id
            or isinstance(self.source_seed_start, bool)
            or not isinstance(self.source_seed_start, int)
            or self.source_seed_start < 0
            or isinstance(self.source_seed_count, bool)
            or not isinstance(self.source_seed_count, int)
            or self.source_seed_count <= 0
            or not self.ranges
        ):
            raise ValueError("episode split plan header is invalid")
        ordered = tuple(sorted(self.ranges, key=lambda row: row.start))
        if ordered != self.ranges or len({row.name for row in ordered}) != len(ordered):
            raise ValueError("episode split ranges must be ordered with unique names")
        cursor = self.source_seed_start
        for row in ordered:
            if row.start != cursor:
                raise ValueError("episode split ranges contain a gap or overlap")
            cursor = row.stop
        if cursor != self.source_seed_start + self.source_seed_count:
            raise ValueError("episode split ranges do not cover the source bank")

    @property
    def split_names(self) -> tuple[str, ...]:
        return tuple(row.name for row in self.ranges)

    @property
    def counts(self) -> dict[str, int]:
        return {row.name: row.count for row in self.ranges}

    def split_for_seed(self, seed: int) -> str:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("episode split seed must be an integer")
        for row in self.ranges:
            if row.start <= seed < row.stop:
                return row.name
        raise ValueError("episode split seed lies outside the source bank")


def load_episode_split_plan(path: Path) -> EpisodeSplitPlan:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "split_id",
        "source_seed_start",
        "source_seed_count",
        "splits",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("episode split config fields are invalid")
    splits = raw.pop("splits")
    if not isinstance(splits, dict) or not splits:
        raise ValueError("episode split config requires split mappings")
    ranges = []
    for name, row in splits.items():
        if not isinstance(row, dict) or set(row) != {"start", "count"}:
            raise ValueError("episode split row fields are invalid")
        ranges.append(EpisodeSeedRange(name=name, **row))
    return EpisodeSplitPlan(**raw, ranges=tuple(ranges))
