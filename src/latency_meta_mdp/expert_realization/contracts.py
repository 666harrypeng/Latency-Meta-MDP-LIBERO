"""Deterministic semantic identities and requests for structured-expert pilot work."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import InitVar, dataclass
from dataclasses import field as dataclass_field
from enum import Enum
from types import MappingProxyType
from typing import Any

from latency_meta_mdp.expert_realization.config import (
    SUBSEED_TAGS,
    FormalCorpusConfig,
    PilotConfig,
    StructuredExpertConfig,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _require_sha256(value: str, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _seed(value: dict[str, Any]) -> int:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _strict_identity_mapping(value: Any, fields: set[str], *, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be a mapping")
    unknown = sorted(set(value).difference(fields))
    missing = sorted(fields.difference(value))
    if unknown:
        raise ValueError(f"{name} contains unknown fields: {unknown}")
    if missing:
        raise ValueError(f"{name} is missing fields: {missing}")
    return value


class StrategyFamily(str, Enum):
    CANONICAL_DIRECT = "canonical_direct"
    EARLY_HIGH_ARC = "early_high_arc"
    LATERAL_ARC = "lateral_arc"
    TIME_SHIFTED_SMOOTH = "time_shifted_smooth"


class FailureClass(str, Enum):
    PLANNER_FAILURE = "planner_failure"
    TASK_FAILURE = "task_failure"
    SAFETY_FAILURE = "safety_failure"
    DIVERSITY_REJECTION = "diversity_rejection"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"


@dataclass(frozen=True)
class MasterTaskRequest:
    corpus_id: str
    logical_task_index: int
    master_task_seed: int
    reserve: bool

    def __post_init__(self) -> None:
        if type(self.corpus_id) is not str or not self.corpus_id:
            raise ValueError("corpus_id must be a non-empty string")
        if type(self.logical_task_index) is not int or self.logical_task_index < 0:
            raise ValueError("logical_task_index must be a non-negative integer")
        if type(self.reserve) is not bool:
            raise TypeError("reserve must be boolean")
        expected = _seed(
            {
                "corpus_id": self.corpus_id,
                "logical_task_index": self.logical_task_index,
            }
        )
        if type(self.master_task_seed) is not int or self.master_task_seed != expected:
            raise ValueError("master_task_seed does not match corpus/task identity")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "corpus_id": self.corpus_id,
            "logical_task_index": self.logical_task_index,
            "master_task_seed": self.master_task_seed,
            "reserve": self.reserve,
        }

    @classmethod
    def from_mapping(cls, mapping: Any) -> MasterTaskRequest:
        return cls(
            **_strict_identity_mapping(
                mapping,
                {"corpus_id", "logical_task_index", "master_task_seed", "reserve"},
                name=cls.__name__,
            )
        )


@dataclass(frozen=True)
class FamilySlotAssignment:
    realization_slot: int
    family: StrategyFamily

    def __post_init__(self) -> None:
        if type(self.realization_slot) is not int or self.realization_slot < 0:
            raise ValueError("realization_slot must be a non-negative integer")
        if not isinstance(self.family, StrategyFamily):
            raise TypeError("family must be a StrategyFamily")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "realization_slot": self.realization_slot,
            "family": self.family.value,
        }

    @classmethod
    def from_mapping(cls, mapping: Any) -> FamilySlotAssignment:
        raw = _strict_identity_mapping(
            mapping,
            {"realization_slot", "family"},
            name=cls.__name__,
        )
        return cls(
            realization_slot=raw["realization_slot"],
            family=StrategyFamily(raw["family"]),
        )


@dataclass(frozen=True)
class TaskInstanceId:
    level: int
    task_instance_seed: int
    motion_profile_sha256: str
    initial_state_sha256: str

    def __post_init__(self) -> None:
        if type(self.level) is not int or self.level not in (1, 2, 3):
            raise ValueError("task instance level must be one of L1, L2, or L3")
        if type(self.task_instance_seed) is not int or self.task_instance_seed < 0:
            raise ValueError("task_instance_seed must be a non-negative integer")
        _require_sha256(self.motion_profile_sha256, name="motion_profile_sha256")
        _require_sha256(self.initial_state_sha256, name="initial_state_sha256")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "task_instance_seed": self.task_instance_seed,
            "motion_profile_sha256": self.motion_profile_sha256,
            "initial_state_sha256": self.initial_state_sha256,
        }

    def canonical_json(self) -> str:
        return _canonical_json(self.to_mapping())

    @classmethod
    def from_mapping(cls, mapping: Any) -> TaskInstanceId:
        raw = _strict_identity_mapping(
            mapping,
            {
                "level",
                "task_instance_seed",
                "motion_profile_sha256",
                "initial_state_sha256",
            },
            name="TaskInstanceId",
        )
        return cls(**raw)


def _mapping_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def sample_uniform_family_for_slot(
    *,
    families: tuple[str, ...],
    master_task_seed: int,
    structured_expert_config_sha256: str,
    realization_slot: int,
) -> StrategyFamily:
    if families != tuple(family.value for family in StrategyFamily):
        raise ValueError("families must contain the canonical strategy-family order")
    if type(master_task_seed) is not int or not 0 <= master_task_seed < 2**64:
        raise ValueError("master_task_seed must be uint64")
    _require_sha256(
        structured_expert_config_sha256,
        name="structured_expert_config_sha256",
    )
    if type(realization_slot) is not int or realization_slot < 0:
        raise ValueError("realization_slot must be a non-negative integer")
    typed_families = tuple(StrategyFamily(value) for value in families)
    return typed_families[
        _seed(
            {
                "master_task_seed": master_task_seed,
                "structured_expert_config_sha256": structured_expert_config_sha256,
                "kind": "iid_uniform_family_slot",
                "realization_slot": realization_slot,
            }
        )
        % len(typed_families)
    ]


def _family_assignments(
    config: FormalCorpusConfig,
    task: MasterTaskRequest,
    *,
    structured_expert_config_sha256: str,
) -> tuple[FamilySlotAssignment, ...]:
    return tuple(
        FamilySlotAssignment(
            realization_slot=slot,
            family=sample_uniform_family_for_slot(
                families=config.families,
                master_task_seed=task.master_task_seed,
                structured_expert_config_sha256=structured_expert_config_sha256,
                realization_slot=slot,
            ),
        )
        for slot in range(config.realizations_per_task)
    )


@dataclass(frozen=True)
class FormalRequestUniverse:
    schema_version: int
    format_id: str
    config: FormalCorpusConfig
    corpus_config_sha256: str
    structured_expert_config_sha256: str
    primary_tasks: tuple[MasterTaskRequest, ...]
    reserve_tasks: tuple[MasterTaskRequest, ...]
    family_assignments: Mapping[int, tuple[FamilySlotAssignment, ...]]
    request_sha256: str = dataclass_field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.format_id != "structured_expert_formal_request_v1":
            raise ValueError("unsupported formal request format")
        if not isinstance(self.config, FormalCorpusConfig):
            raise TypeError("config must be a FormalCorpusConfig")
        _require_sha256(self.corpus_config_sha256, name="corpus_config_sha256")
        _require_sha256(
            self.structured_expert_config_sha256,
            name="structured_expert_config_sha256",
        )
        for name in ("primary_tasks", "reserve_tasks"):
            rows = getattr(self, name)
            if type(rows) is not tuple or any(
                not isinstance(row, MasterTaskRequest) for row in rows
            ):
                raise TypeError(f"{name} must contain MasterTaskRequest values")
        expected_primary = self.config.primary_task_indices
        expected_reserve = self.config.reserve_task_indices
        if tuple(row.logical_task_index for row in self.primary_tasks) != expected_primary or any(
            row.reserve or row.corpus_id != self.config.corpus_id for row in self.primary_tasks
        ):
            raise ValueError("primary task inventory does not match formal config")
        if tuple(row.logical_task_index for row in self.reserve_tasks) != expected_reserve or any(
            not row.reserve or row.corpus_id != self.config.corpus_id for row in self.reserve_tasks
        ):
            raise ValueError("reserve task inventory does not match formal config")
        all_tasks = self.primary_tasks + self.reserve_tasks
        expected_indices = {row.logical_task_index for row in all_tasks}
        if not isinstance(self.family_assignments, Mapping) or set(
            self.family_assignments
        ) != expected_indices:
            raise ValueError("family assignment inventory does not match task universe")
        frozen_assignments: dict[int, tuple[FamilySlotAssignment, ...]] = {}
        expected_families = {StrategyFamily(value) for value in self.config.families}
        for task_index in sorted(expected_indices):
            rows = self.family_assignments[task_index]
            if type(rows) is not tuple or any(
                not isinstance(row, FamilySlotAssignment) for row in rows
            ):
                raise TypeError("family assignments must be typed tuples")
            if [row.realization_slot for row in rows] != list(
                range(self.config.realizations_per_task)
            ):
                raise ValueError("family assignment slots are incomplete or out of order")
            if any(row.family not in expected_families for row in rows):
                raise ValueError("family assignment contains an unknown strategy family")
            frozen_assignments[task_index] = rows
        object.__setattr__(
            self,
            "family_assignments",
            MappingProxyType(frozen_assignments),
        )
        object.__setattr__(self, "request_sha256", _mapping_sha256(self._payload_mapping()))

    def _payload_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "format_id": self.format_id,
            "config": self.config.to_mapping(),
            "corpus_config_sha256": self.corpus_config_sha256,
            "structured_expert_config_sha256": self.structured_expert_config_sha256,
            "primary_tasks": [row.to_mapping() for row in self.primary_tasks],
            "reserve_tasks": [row.to_mapping() for row in self.reserve_tasks],
            "family_assignments": [
                {
                    "logical_task_index": task_index,
                    "slots": [row.to_mapping() for row in self.family_assignments[task_index]],
                }
                for task_index in sorted(self.family_assignments)
            ],
        }

    def to_mapping(self) -> dict[str, Any]:
        return {**self._payload_mapping(), "request_sha256": self.request_sha256}

    @classmethod
    def from_mapping(cls, mapping: Any) -> FormalRequestUniverse:
        raw = _strict_identity_mapping(
            mapping,
            {
                "schema_version",
                "format_id",
                "config",
                "corpus_config_sha256",
                "structured_expert_config_sha256",
                "primary_tasks",
                "reserve_tasks",
                "family_assignments",
                "request_sha256",
            },
            name=cls.__name__,
        ).copy()
        stored_sha = raw.pop("request_sha256")
        if type(raw["primary_tasks"]) is not list or type(raw["reserve_tasks"]) is not list:
            raise TypeError("formal task inventories must be JSON lists")
        if type(raw["family_assignments"]) is not list:
            raise TypeError("family_assignments must be a JSON list")
        config = FormalCorpusConfig.from_mapping(raw["config"])
        assignments: dict[int, tuple[FamilySlotAssignment, ...]] = {}
        for item in raw["family_assignments"]:
            assignment = _strict_identity_mapping(
                item,
                {"logical_task_index", "slots"},
                name="family assignment row",
            )
            task_index = assignment["logical_task_index"]
            if type(task_index) is not int or task_index < 0 or task_index in assignments:
                raise ValueError("family assignment task index is invalid or duplicated")
            if type(assignment["slots"]) is not list:
                raise TypeError("family assignment slots must be a JSON list")
            assignments[task_index] = tuple(
                FamilySlotAssignment.from_mapping(row) for row in assignment["slots"]
            )
        universe = cls(
            schema_version=raw["schema_version"],
            format_id=raw["format_id"],
            config=config,
            corpus_config_sha256=raw["corpus_config_sha256"],
            structured_expert_config_sha256=raw["structured_expert_config_sha256"],
            primary_tasks=tuple(
                MasterTaskRequest.from_mapping(row) for row in raw["primary_tasks"]
            ),
            reserve_tasks=tuple(
                MasterTaskRequest.from_mapping(row) for row in raw["reserve_tasks"]
            ),
            family_assignments=assignments,
        )
        if stored_sha != universe.request_sha256:
            raise ValueError("request_sha256 does not bind the formal request payload")
        return universe


def build_formal_request_universe(
    config: FormalCorpusConfig,
    *,
    corpus_config_sha256: str,
    structured_expert_config_sha256: str,
) -> FormalRequestUniverse:
    if not isinstance(config, FormalCorpusConfig):
        raise TypeError("config must be a FormalCorpusConfig")
    _require_sha256(corpus_config_sha256, name="corpus_config_sha256")
    _require_sha256(
        structured_expert_config_sha256,
        name="structured_expert_config_sha256",
    )

    def task(index: int, *, reserve: bool) -> MasterTaskRequest:
        return MasterTaskRequest(
            corpus_id=config.corpus_id,
            logical_task_index=index,
            master_task_seed=_seed(
                {
                    "corpus_id": config.corpus_id,
                    "logical_task_index": index,
                }
            ),
            reserve=reserve,
        )

    primary = tuple(task(index, reserve=False) for index in config.primary_task_indices)
    reserve = tuple(task(index, reserve=True) for index in config.reserve_task_indices)
    assignments = {
        row.logical_task_index: _family_assignments(
            config,
            row,
            structured_expert_config_sha256=structured_expert_config_sha256,
        )
        for row in primary + reserve
    }
    return FormalRequestUniverse(
        schema_version=1,
        format_id="structured_expert_formal_request_v1",
        config=config,
        corpus_config_sha256=corpus_config_sha256,
        structured_expert_config_sha256=structured_expert_config_sha256,
        primary_tasks=primary,
        reserve_tasks=reserve,
        family_assignments=assignments,
    )


@dataclass(frozen=True)
class FormalRealizationRequest:
    task_instance_id: TaskInstanceId
    realization_slot: int
    assigned_family: StrategyFamily
    realization_namespace_sha256: str
    realization_seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.task_instance_id, TaskInstanceId):
            raise TypeError("task_instance_id must be a TaskInstanceId")
        if type(self.realization_slot) is not int or self.realization_slot < 0:
            raise ValueError("realization_slot must be a non-negative integer")
        if not isinstance(self.assigned_family, StrategyFamily):
            raise TypeError("assigned_family must be a StrategyFamily")
        _require_sha256(
            self.realization_namespace_sha256,
            name="realization_namespace_sha256",
        )
        expected = derive_realization_seed(
            self.task_instance_id,
            self.realization_slot,
            self.realization_namespace_sha256,
        )
        if type(self.realization_seed) is not int or self.realization_seed != expected:
            raise ValueError("realization_seed does not match request identity")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "task_instance_id": self.task_instance_id.to_mapping(),
            "realization_slot": self.realization_slot,
            "assigned_family": self.assigned_family.value,
            "realization_namespace_sha256": self.realization_namespace_sha256,
            "realization_seed": self.realization_seed,
        }

    def to_expert_realization_key(self) -> ExpertRealizationKey:
        key = ExpertRealizationKey(
            self.task_instance_id,
            self.realization_slot,
            self.realization_namespace_sha256,
        )
        if key.realization_seed != self.realization_seed:
            raise ValueError("formal request and expert realization key seeds do not match")
        return key

    @classmethod
    def from_mapping(cls, mapping: Any) -> FormalRealizationRequest:
        raw = _strict_identity_mapping(
            mapping,
            {
                "task_instance_id",
                "realization_slot",
                "assigned_family",
                "realization_namespace_sha256",
                "realization_seed",
            },
            name=cls.__name__,
        )
        return cls(
            task_instance_id=TaskInstanceId.from_mapping(raw["task_instance_id"]),
            realization_slot=raw["realization_slot"],
            assigned_family=StrategyFamily(raw["assigned_family"]),
            realization_namespace_sha256=raw["realization_namespace_sha256"],
            realization_seed=raw["realization_seed"],
        )


@dataclass(frozen=True)
class RealizationUniverseIdentity:
    request_sha256: str
    task_instance_id: TaskInstanceId
    realization_slots: tuple[int, ...]
    universe_sha256: str = dataclass_field(init=False)

    def __post_init__(self) -> None:
        _require_sha256(self.request_sha256, name="request_sha256")
        if not isinstance(self.task_instance_id, TaskInstanceId):
            raise TypeError("task_instance_id must be a TaskInstanceId")
        if (
            type(self.realization_slots) is not tuple
            or not self.realization_slots
            or any(type(slot) is not int for slot in self.realization_slots)
            or self.realization_slots != tuple(range(len(self.realization_slots)))
        ):
            raise ValueError("realization slots must be one canonical contiguous tuple from zero")
        object.__setattr__(
            self,
            "universe_sha256",
            _mapping_sha256(
                {
                    "request_sha256": self.request_sha256,
                    "task_instance_id": self.task_instance_id.to_mapping(),
                    "realization_slots": list(self.realization_slots),
                }
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "request_sha256": self.request_sha256,
            "task_instance_id": self.task_instance_id.to_mapping(),
            "realization_slots": list(self.realization_slots),
            "universe_sha256": self.universe_sha256,
        }

    @classmethod
    def from_mapping(cls, mapping: Any) -> RealizationUniverseIdentity:
        raw = _strict_identity_mapping(
            mapping,
            {
                "request_sha256",
                "task_instance_id",
                "realization_slots",
                "universe_sha256",
            },
            name=cls.__name__,
        )
        if type(raw["realization_slots"]) is not list:
            raise TypeError("realization_slots must be a JSON list")
        stored_sha = raw["universe_sha256"]
        identity = cls(
            request_sha256=raw["request_sha256"],
            task_instance_id=TaskInstanceId.from_mapping(raw["task_instance_id"]),
            realization_slots=tuple(raw["realization_slots"]),
        )
        if stored_sha != identity.universe_sha256:
            raise ValueError("universe_sha256 does not bind request/task/slot payload")
        return identity


def build_realization_universe_identity(
    *,
    request_sha256: str,
    task_instance_id: TaskInstanceId,
    realization_slots: tuple[int, ...],
) -> RealizationUniverseIdentity:
    return RealizationUniverseIdentity(
        request_sha256=request_sha256,
        task_instance_id=task_instance_id,
        realization_slots=realization_slots,
    )


def build_formal_realization_requests(
    universe: FormalRequestUniverse,
    task_instance_id: TaskInstanceId,
) -> tuple[FormalRealizationRequest, ...]:
    if not isinstance(universe, FormalRequestUniverse):
        raise TypeError("universe must be a FormalRequestUniverse")
    if not isinstance(task_instance_id, TaskInstanceId):
        raise TypeError("task_instance_id must be a TaskInstanceId")
    if task_instance_id.level not in universe.config.levels:
        raise ValueError("task level is outside the formal request")
    matching = [
        row
        for row in universe.primary_tasks + universe.reserve_tasks
        if row.master_task_seed == task_instance_id.task_instance_seed
    ]
    if len(matching) != 1:
        raise ValueError("task instance seed does not identify one formal master task")
    master = matching[0]
    namespace = _mapping_sha256(
        {
            "request_sha256": universe.request_sha256,
            "logical_task_index": master.logical_task_index,
        }
    )
    return tuple(
        FormalRealizationRequest(
            task_instance_id=task_instance_id,
            realization_slot=assignment.realization_slot,
            assigned_family=assignment.family,
            realization_namespace_sha256=namespace,
            realization_seed=derive_realization_seed(
                task_instance_id,
                assignment.realization_slot,
                namespace,
            ),
        )
        for assignment in universe.family_assignments[master.logical_task_index]
    )


def validate_nonoverlapping_extension(
    existing_intervals: tuple[tuple[int, int], ...],
    requested_interval: tuple[int, int],
) -> None:
    def validate(interval: tuple[int, int], *, name: str) -> tuple[int, int]:
        if (
            type(interval) is not tuple
            or len(interval) != 2
            or any(type(value) is not int for value in interval)
            or interval[0] < 0
            or interval[1] <= interval[0]
        ):
            raise ValueError(f"{name} must be a non-empty non-negative half-open interval")
        return interval

    if type(existing_intervals) is not tuple:
        raise TypeError("existing_intervals must be a tuple")
    requested_start, requested_stop = validate(requested_interval, name="requested_interval")
    validated_existing = tuple(
        validate(interval, name="existing interval") for interval in existing_intervals
    )
    for index, (start, stop) in enumerate(validated_existing):
        if any(
            max(start, other_start) < min(stop, other_stop)
            for other_start, other_stop in validated_existing[index + 1 :]
        ):
            raise ValueError("existing corpus intervals overlap each other")
        if max(start, requested_start) < min(stop, requested_stop):
            raise ValueError("requested corpus interval overlaps existing inventory")


def derive_realization_seed(
    task_instance_id: TaskInstanceId,
    realization_index: int,
    realization_namespace_sha256: str,
) -> int:
    if (
        isinstance(realization_index, bool)
        or not isinstance(realization_index, int)
        or realization_index < 0
    ):
        raise ValueError("realization_index must be a non-negative integer")
    _require_sha256(realization_namespace_sha256, name="realization_namespace_sha256")
    return _seed(
        {
            "task_instance_id": task_instance_id.to_mapping(),
            "realization_index": realization_index,
            "realization_namespace_sha256": realization_namespace_sha256,
        }
    )


def derive_subseed(realization_seed: int, tag: str) -> int:
    if (
        isinstance(realization_seed, bool)
        or not isinstance(realization_seed, int)
        or not 0 <= realization_seed < 2**64
    ):
        raise ValueError("realization_seed must be a uint64")
    if tag not in SUBSEED_TAGS:
        raise ValueError(f"unknown structured-expert subseed tag: {tag}")
    return _seed({"realization_seed": realization_seed, "tag": tag})


@dataclass(frozen=True)
class ExpertRealizationKey:
    task_instance_id: TaskInstanceId
    realization_index: int
    realization_namespace_sha256: str
    realization_seed: int = dataclass_field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.task_instance_id, TaskInstanceId):
            raise TypeError("task_instance_id must be a TaskInstanceId")
        if type(self.realization_index) is not int or self.realization_index < 0:
            raise ValueError("realization_index must be a non-negative integer")
        _require_sha256(
            self.realization_namespace_sha256,
            name="realization_namespace_sha256",
        )
        object.__setattr__(
            self,
            "realization_seed",
            derive_realization_seed(
                self.task_instance_id,
                self.realization_index,
                self.realization_namespace_sha256,
            ),
        )

    def subseeds(self) -> dict[str, int]:
        return {tag: derive_subseed(self.realization_seed, tag) for tag in SUBSEED_TAGS}

    def to_mapping(self) -> dict[str, Any]:
        return {
            "task_instance_id": self.task_instance_id.to_mapping(),
            "realization_index": self.realization_index,
            "realization_namespace_sha256": self.realization_namespace_sha256,
            "realization_seed": self.realization_seed,
        }

    @classmethod
    def from_mapping(
        cls,
        mapping: Any,
        *,
        structured_expert_config_sha256: str | None = None,
    ) -> ExpertRealizationKey:
        raw = _strict_identity_mapping(
            mapping,
            {
                "task_instance_id",
                "realization_index",
                "realization_namespace_sha256",
                "realization_seed",
            },
            name="ExpertRealizationKey",
        )
        if (
            structured_expert_config_sha256 is not None
            and raw["realization_namespace_sha256"] != structured_expert_config_sha256
        ):
            raise ValueError("serialized realization namespace does not match caller context")
        stored_seed = raw["realization_seed"]
        if type(stored_seed) is not int or not 0 <= stored_seed < 2**64:
            raise ValueError("realization_seed must be a uint64")
        key = cls(
            TaskInstanceId.from_mapping(raw["task_instance_id"]),
            raw["realization_index"],
            raw["realization_namespace_sha256"],
        )
        if stored_seed != key.realization_seed:
            raise ValueError("realization_seed does not match the canonical identity derivation")
        return key


@dataclass(frozen=True)
class ExpertRealizationId:
    expert_realization_key: ExpertRealizationKey
    task_instance_plan_set_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.expert_realization_key, ExpertRealizationKey):
            raise TypeError("expert_realization_key must be an ExpertRealizationKey")
        _require_sha256(self.task_instance_plan_set_sha256, name="task_instance_plan_set_sha256")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "expert_realization_key": self.expert_realization_key.to_mapping(),
            "task_instance_plan_set_sha256": self.task_instance_plan_set_sha256,
        }

    @classmethod
    def from_mapping(
        cls,
        mapping: Any,
        *,
        structured_expert_config_sha256: str | None = None,
    ) -> ExpertRealizationId:
        raw = _strict_identity_mapping(
            mapping,
            {"expert_realization_key", "task_instance_plan_set_sha256"},
            name="ExpertRealizationId",
        )
        return cls(
            expert_realization_key=ExpertRealizationKey.from_mapping(
                raw["expert_realization_key"],
                structured_expert_config_sha256=structured_expert_config_sha256,
            ),
            task_instance_plan_set_sha256=raw["task_instance_plan_set_sha256"],
        )


@dataclass(frozen=True)
class AttemptId:
    expert_realization_id: ExpertRealizationId
    attempt_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.expert_realization_id, ExpertRealizationId):
            raise TypeError("expert_realization_id must be an ExpertRealizationId")
        if (
            isinstance(self.attempt_index, bool)
            or not isinstance(self.attempt_index, int)
            or self.attempt_index < 0
        ):
            raise ValueError("attempt_index must be a non-negative integer")

    def semantic_identity(self) -> ExpertRealizationId:
        return self.expert_realization_id

    def to_mapping(self) -> dict[str, Any]:
        return {
            "expert_realization_id": self.expert_realization_id.to_mapping(),
            "attempt_index": self.attempt_index,
        }

    @classmethod
    def from_mapping(
        cls,
        mapping: Any,
        *,
        structured_expert_config_sha256: str | None = None,
    ) -> AttemptId:
        raw = _strict_identity_mapping(
            mapping,
            {"expert_realization_id", "attempt_index"},
            name="AttemptId",
        )
        return cls(
            expert_realization_id=ExpertRealizationId.from_mapping(
                raw["expert_realization_id"],
                structured_expert_config_sha256=structured_expert_config_sha256,
            ),
            attempt_index=raw["attempt_index"],
        )


def _draw_unit_interval(seed: int, *, field: str) -> float:
    return _seed({"field": field, "seed": seed}) / 2**64


def _draw_uniform(seed: int, *, field: str, bounds: tuple[float, float]) -> float:
    return bounds[0] + (bounds[1] - bounds[0]) * _draw_unit_interval(seed, field=field)


def _draw_int(seed: int, *, field: str, bounds: tuple[int, int]) -> int:
    return bounds[0] + _seed({"field": field, "seed": seed}) % (bounds[1] - bounds[0] + 1)


@dataclass(frozen=True)
class StrategyParameters:
    family: StrategyFamily
    close_target_tick: int
    prediction_lead_seconds: float
    funnel_entry_height_m: float
    high_arc_extra_height_m: float | None
    lateral_offset_m: float | None
    lateral_direction_sign: int | None
    soft_guide_radius_m: float
    tracking_error_clip_m: float
    grasp_eef_height_offset_m: float
    funnel_descent_ticks: int
    funnel_entry_deadline_slack_ticks: int
    close_window_half_width_ticks: int
    handoff_window_ticks: int
    close_dwell_ticks: int
    bilateral_contact_acquisition_ticks: int
    close_centering_tolerance_m: float
    close_distance_tolerance_m: float
    close_relative_speed_tolerance_mps: float
    lift_vertical_displacement_m: float
    fixed_orientation: bool
    rotation_action_variation: bool
    iid_per_tick_action_noise: bool
    config: InitVar[StructuredExpertConfig]

    def __post_init__(self, config: StructuredExpertConfig) -> None:
        if not isinstance(config, StructuredExpertConfig):
            raise TypeError("config must be a StructuredExpertConfig")
        if not isinstance(self.family, StrategyFamily):
            raise TypeError("family must be a StrategyFamily")
        if type(self.close_target_tick) is not int:
            raise ValueError("close_target_tick must be an integer")
        if (
            not config.close_target_tick_ranges[self.family.value][0]
            <= self.close_target_tick
            <= config.close_target_tick_ranges[self.family.value][1]
        ):
            raise ValueError("close_target_tick is outside the family-specific range")
        values = {
            "prediction_lead_seconds": (
                self.prediction_lead_seconds,
                config.prediction_lead_seconds,
            ),
        }
        for name, (value, bounds) in values.items():
            if (
                type(value) is not float
                or not math.isfinite(value)
                or not bounds[0] <= value <= bounds[1]
            ):
                raise ValueError(f"{name} is outside the structured expert bounds")
        if self.family is StrategyFamily.EARLY_HIGH_ARC:
            if (
                type(self.high_arc_extra_height_m) is not float
                or not config.high_arc_extra_height_m[0]
                <= self.high_arc_extra_height_m
                <= config.high_arc_extra_height_m[1]
            ):
                raise ValueError("early_high_arc requires a bounded extra height")
        elif self.high_arc_extra_height_m is not None:
            raise ValueError("only early_high_arc may have an extra height")
        if self.family is StrategyFamily.LATERAL_ARC and self.lateral_offset_m is None:
            raise ValueError("lateral_arc requires a lateral offset")
        if self.family is not StrategyFamily.LATERAL_ARC and self.lateral_offset_m is not None:
            raise ValueError("only lateral_arc may have a lateral offset")
        if self.lateral_offset_m is not None and (
            type(self.lateral_offset_m) is not float
            or not math.isfinite(self.lateral_offset_m)
            or not config.lateral_offset_m[0] <= self.lateral_offset_m <= config.lateral_offset_m[1]
        ):
            raise ValueError("lateral_offset_m is outside the structured expert bounds")
        if self.family is StrategyFamily.LATERAL_ARC:
            if self.lateral_direction_sign not in (-1, 1):
                raise ValueError("lateral_arc requires a signed lateral direction")
        elif self.lateral_direction_sign is not None:
            raise ValueError("only lateral_arc may have a lateral direction sign")
        canonical_values = {
            "funnel_entry_height_m": config.funnel_entry_height_m,
            "soft_guide_radius_m": config.soft_guide_radius_m,
            "tracking_error_clip_m": config.tracking_error_clip_m,
            "grasp_eef_height_offset_m": config.grasp_eef_height_offset_m,
            "funnel_descent_ticks": config.funnel_descent_ticks,
            "funnel_entry_deadline_slack_ticks": config.funnel_entry_deadline_slack_ticks,
            "close_window_half_width_ticks": config.close_window_half_width_ticks,
            "handoff_window_ticks": config.handoff_window_ticks,
            "close_dwell_ticks": config.close_dwell_ticks,
            "bilateral_contact_acquisition_ticks": config.bilateral_contact_acquisition_ticks,
            "close_centering_tolerance_m": config.close_centering_tolerance_m,
            "close_distance_tolerance_m": config.close_distance_tolerance_m,
            "close_relative_speed_tolerance_mps": config.close_relative_speed_tolerance_mps,
            "lift_vertical_displacement_m": config.lift_vertical_displacement_m,
        }
        for name, expected in canonical_values.items():
            if getattr(self, name) != expected:
                raise ValueError(f"{name} must equal the canonical grasp-funnel config")
        if (
            type(self.fixed_orientation) is not bool
            or type(self.rotation_action_variation) is not bool
            or type(self.iid_per_tick_action_noise) is not bool
            or self.fixed_orientation is not config.fixed_orientation
            or self.rotation_action_variation is not config.rotation_action_variation
            or self.iid_per_tick_action_noise is not config.iid_per_tick_action_noise
        ):
            raise ValueError("strategy action/noise flags do not match the structured expert")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "family": self.family.value,
            "close_target_tick": self.close_target_tick,
            "prediction_lead_seconds": self.prediction_lead_seconds,
            "funnel_entry_height_m": self.funnel_entry_height_m,
            "high_arc_extra_height_m": self.high_arc_extra_height_m,
            "lateral_offset_m": self.lateral_offset_m,
            "lateral_direction_sign": self.lateral_direction_sign,
            "soft_guide_radius_m": self.soft_guide_radius_m,
            "tracking_error_clip_m": self.tracking_error_clip_m,
            "grasp_eef_height_offset_m": self.grasp_eef_height_offset_m,
            "funnel_descent_ticks": self.funnel_descent_ticks,
            "funnel_entry_deadline_slack_ticks": self.funnel_entry_deadline_slack_ticks,
            "close_window_half_width_ticks": self.close_window_half_width_ticks,
            "handoff_window_ticks": self.handoff_window_ticks,
            "close_dwell_ticks": self.close_dwell_ticks,
            "bilateral_contact_acquisition_ticks": self.bilateral_contact_acquisition_ticks,
            "close_centering_tolerance_m": self.close_centering_tolerance_m,
            "close_distance_tolerance_m": self.close_distance_tolerance_m,
            "close_relative_speed_tolerance_mps": self.close_relative_speed_tolerance_mps,
            "lift_vertical_displacement_m": self.lift_vertical_displacement_m,
            "fixed_orientation": self.fixed_orientation,
            "rotation_action_variation": self.rotation_action_variation,
            "iid_per_tick_action_noise": self.iid_per_tick_action_noise,
        }


def sample_strategy_parameters(
    config: StructuredExpertConfig, *, family: str | StrategyFamily, realization_seed: int
) -> StrategyParameters:
    if not isinstance(config, StructuredExpertConfig):
        raise TypeError("config must be a StructuredExpertConfig")
    family_value = StrategyFamily(family)
    strategy_seed = derive_subseed(realization_seed, "strategy")
    trajectory_intent_seed = derive_subseed(realization_seed, "trajectory_intent")
    timing_seed = derive_subseed(realization_seed, "timing")
    lateral_offset = (
        _draw_uniform(strategy_seed, field="lateral_offset_m", bounds=config.lateral_offset_m)
        if family_value is StrategyFamily.LATERAL_ARC
        else None
    )
    lateral_direction_sign = (
        -1
        if _seed({"field": "lateral_direction_sign", "seed": trajectory_intent_seed}) % 2 == 0
        else 1
    )
    high_arc_extra_height = (
        _draw_uniform(
            trajectory_intent_seed,
            field="high_arc_extra_height_m",
            bounds=config.high_arc_extra_height_m,
        )
        if family_value is StrategyFamily.EARLY_HIGH_ARC
        else None
    )
    return StrategyParameters(
        family=family_value,
        close_target_tick=_draw_int(
            timing_seed,
            field="close_target_tick",
            bounds=config.close_target_tick_ranges[family_value.value],
        ),
        prediction_lead_seconds=_draw_uniform(
            strategy_seed,
            field="prediction_lead_seconds",
            bounds=config.prediction_lead_seconds,
        ),
        funnel_entry_height_m=config.funnel_entry_height_m,
        high_arc_extra_height_m=high_arc_extra_height,
        lateral_offset_m=lateral_offset,
        lateral_direction_sign=(
            lateral_direction_sign if family_value is StrategyFamily.LATERAL_ARC else None
        ),
        soft_guide_radius_m=config.soft_guide_radius_m,
        tracking_error_clip_m=config.tracking_error_clip_m,
        grasp_eef_height_offset_m=config.grasp_eef_height_offset_m,
        funnel_descent_ticks=config.funnel_descent_ticks,
        funnel_entry_deadline_slack_ticks=config.funnel_entry_deadline_slack_ticks,
        close_window_half_width_ticks=config.close_window_half_width_ticks,
        handoff_window_ticks=config.handoff_window_ticks,
        close_dwell_ticks=config.close_dwell_ticks,
        bilateral_contact_acquisition_ticks=config.bilateral_contact_acquisition_ticks,
        close_centering_tolerance_m=config.close_centering_tolerance_m,
        close_distance_tolerance_m=config.close_distance_tolerance_m,
        close_relative_speed_tolerance_mps=config.close_relative_speed_tolerance_mps,
        lift_vertical_displacement_m=config.lift_vertical_displacement_m,
        fixed_orientation=config.fixed_orientation,
        rotation_action_variation=config.rotation_action_variation,
        iid_per_tick_action_noise=config.iid_per_tick_action_noise,
        config=config,
    )


@dataclass(frozen=True)
class PilotRequest:
    pilot_config: PilotConfig
    structured_expert_config_sha256: str
    curobo_planner_config_sha256: str
    pilot_gate_config_sha256: str
    task_instances: tuple[TaskInstanceId, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.pilot_config, PilotConfig):
            raise TypeError("pilot_config must be a PilotConfig")
        for field in (
            "structured_expert_config_sha256",
            "curobo_planner_config_sha256",
            "pilot_gate_config_sha256",
        ):
            _require_sha256(getattr(self, field), name=field)
        if type(self.task_instances) is not tuple:
            raise TypeError("task_instances must be a tuple in parent canonical order")
        expected = tuple(
            (level, seed) for level in self.pilot_config.levels for seed in self.pilot_config.seeds
        )
        if any(not isinstance(item, TaskInstanceId) for item in self.task_instances):
            raise TypeError("task_instances must contain TaskInstanceId values")
        actual = tuple((item.level, item.task_instance_seed) for item in self.task_instances)
        if actual != expected:
            raise ValueError("task_instances must bind parent levels and seeds in canonical order")

    @property
    def pilot_id(self) -> str:
        return self.pilot_config.pilot_id

    @property
    def bounded_review_only(self) -> bool:
        return self.pilot_config.bounded_review_only

    def training_eligibility(self) -> dict[str, bool]:
        return self.pilot_config.training_eligibility()

    def realization_keys(self) -> tuple[ExpertRealizationKey, ...]:
        keys: list[ExpertRealizationKey] = []
        for task_instance in self.task_instances:
            for family_index, _family in enumerate(self.pilot_config.families):
                for continuous_sample_index in range(self.pilot_config.samples_per_family):
                    realization_index = (
                        family_index * self.pilot_config.samples_per_family
                        + continuous_sample_index
                    )
                    keys.append(
                        ExpertRealizationKey(
                            task_instance_id=task_instance,
                            realization_index=realization_index,
                            realization_namespace_sha256=self.structured_expert_config_sha256,
                        )
                    )
        return tuple(keys)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "pilot_config": self.pilot_config.to_mapping(),
            "structured_expert_config_sha256": self.structured_expert_config_sha256,
            "curobo_planner_config_sha256": self.curobo_planner_config_sha256,
            "pilot_gate_config_sha256": self.pilot_gate_config_sha256,
            "task_instances": [item.to_mapping() for item in self.task_instances],
        }

    def canonical_json(self) -> str:
        return _canonical_json(self.to_mapping())


@dataclass(frozen=True)
class PilotStageRequest:
    parent: PilotRequest
    stage_id: str
    task_instance_seed_start: int
    task_instance_seed_stop: int
    levels: tuple[int, ...]
    output_root: str
    resource_report_sha256: str
    planner_worker_count: int
    collection_worker_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.parent, PilotRequest):
            raise TypeError("parent must be a PilotRequest")
        if not isinstance(self.stage_id, str) or not self.stage_id.strip():
            raise ValueError("stage_id must be a non-empty string")
        if not isinstance(self.output_root, str) or not self.output_root.strip():
            raise ValueError("output_root must be a non-empty string")
        if (
            type(self.task_instance_seed_start) is not int
            or type(self.task_instance_seed_stop) is not int
            or self.task_instance_seed_start >= self.task_instance_seed_stop
            or self.task_instance_seed_start < self.parent.pilot_config.task_instance_seed_start
            or self.task_instance_seed_stop > self.parent.pilot_config.task_instance_seed_stop
        ):
            raise ValueError("stage seed interval must be a non-empty parent subset")
        if type(self.levels) is not tuple or any(type(level) is not int for level in self.levels):
            raise TypeError("stage levels must be an integer tuple")
        levels = self.levels
        parent_levels = self.parent.pilot_config.levels
        if not levels or any(level not in parent_levels for level in levels):
            raise ValueError("stage levels must be a non-empty parent subset")
        expected_order = tuple(level for level in parent_levels if level in levels)
        if levels != expected_order or len(set(levels)) != len(levels):
            raise ValueError("stage levels must preserve parent order")
        _require_sha256(self.resource_report_sha256, name="resource_report_sha256")
        for field in ("planner_worker_count", "collection_worker_count"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")

    @property
    def parent_pilot_id(self) -> str:
        return self.parent.pilot_id

    def training_eligibility(self) -> dict[str, bool]:
        return self.parent.training_eligibility()

    def to_mapping(self) -> dict[str, Any]:
        return {
            "parent_pilot_id": self.parent_pilot_id,
            "parent_request": self.parent.to_mapping(),
            "stage_id": self.stage_id,
            "task_instance_seed_start": self.task_instance_seed_start,
            "task_instance_seed_stop": self.task_instance_seed_stop,
            "levels": list(self.levels),
            "output_root": self.output_root,
            "resource_report_sha256": self.resource_report_sha256,
            "planner_worker_count": self.planner_worker_count,
            "collection_worker_count": self.collection_worker_count,
            "training_eligibility": self.training_eligibility(),
        }
