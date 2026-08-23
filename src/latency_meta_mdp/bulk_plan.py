"""Validated seed and collection contract for the Panda-ball bulk corpus."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from latency_meta_mdp.recording import RecordProfile


@dataclass(frozen=True)
class SeedBank:
    start: int
    count: int
    collect_expert: bool

    def __post_init__(self) -> None:
        if (
            isinstance(self.start, bool)
            or not isinstance(self.start, int)
            or self.start < 0
            or isinstance(self.count, bool)
            or not isinstance(self.count, int)
            or self.count <= 0
        ):
            raise ValueError("seed-bank start and count are invalid")
        if type(self.collect_expert) is not bool:
            raise TypeError("collect_expert must be a boolean")

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(range(self.start, self.start + self.count))


@dataclass(frozen=True)
class BulkCollectionPlan:
    schema_version: int
    collection_id: str
    levels: tuple[int, ...]
    record_profile: RecordProfile
    camera_width: int
    camera_height: int
    same_master_seeds_across_levels: bool
    excluded_pilot_seeds: tuple[int, ...]
    first_tranche_count: int
    minimum_first_attempt_success_rate: float
    review_video_count_per_level: int
    review_video_fps: int
    train: SeedBank
    development: SeedBank
    test: SeedBank

    def __post_init__(self) -> None:
        object.__setattr__(self, "levels", tuple(self.levels))
        object.__setattr__(self, "excluded_pilot_seeds", tuple(self.excluded_pilot_seeds))
        if self.schema_version != 1 or self.collection_id != "panda_ball_bulk_v1":
            raise ValueError("unsupported bulk collection schema or identifier")
        if self.levels != (1, 2, 3):
            raise ValueError("bulk collection must define L1, L2, and L3")
        if self.record_profile is not RecordProfile.BELIEF:
            raise ValueError("bulk expert trajectories must use the belief record profile")
        if not self.same_master_seeds_across_levels:
            raise ValueError("bulk levels must share master seed identities")
        for name in (
            "camera_width",
            "camera_height",
            "first_tranche_count",
            "review_video_count_per_level",
            "review_video_fps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.camera_width % 2 or self.camera_height % 2:
            raise ValueError("bulk camera dimensions must be even")
        if self.first_tranche_count > self.train.count:
            raise ValueError("first tranche cannot exceed the train seed bank")
        if not 0.0 < self.minimum_first_attempt_success_rate <= 1.0:
            raise ValueError("minimum success rate must be in (0, 1]")
        if not self.train.collect_expert:
            raise ValueError("the train bank must collect expert trajectories")
        if self.development.collect_expert or self.test.collect_expert:
            raise ValueError("development and test banks are rollout-only")
        banks = [set(bank.seeds) for bank in (self.train, self.development, self.test)]
        if any(left & right for index, left in enumerate(banks) for right in banks[index + 1 :]):
            raise ValueError("seed banks must be disjoint")
        excluded = set(self.excluded_pilot_seeds)
        if len(excluded) != len(self.excluded_pilot_seeds) or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in self.excluded_pilot_seeds
        ):
            raise ValueError("excluded pilot seeds are invalid")
        if excluded & set.union(*banks):
            raise ValueError("excluded pilot seeds overlap a formal seed bank")


def load_bulk_collection_plan(path: Path) -> BulkCollectionPlan:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "collection_id",
        "levels",
        "record_profile",
        "camera_width",
        "camera_height",
        "same_master_seeds_across_levels",
        "excluded_pilot_seeds",
        "first_tranche_count",
        "minimum_first_attempt_success_rate",
        "review_video_count_per_level",
        "review_video_fps",
        "seed_banks",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("bulk collection config fields are invalid")
    banks = raw.pop("seed_banks")
    if not isinstance(banks, dict) or set(banks) != {"train", "development", "test"}:
        raise ValueError("bulk seed-bank definitions are invalid")
    if any(not isinstance(row, dict) for row in banks.values()):
        raise ValueError("bulk seed-bank rows must be mappings")
    levels = tuple(raw.pop("levels"))
    excluded_pilot_seeds = tuple(raw.pop("excluded_pilot_seeds"))
    record_profile = RecordProfile(raw.pop("record_profile"))
    return BulkCollectionPlan(
        **raw,
        levels=levels,
        excluded_pilot_seeds=excluded_pilot_seeds,
        record_profile=record_profile,
        train=SeedBank(**banks["train"]),
        development=SeedBank(**banks["development"]),
        test=SeedBank(**banks["test"]),
    )
