"""Deterministic semantic identities and requests for structured-expert pilot work."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import InitVar, dataclass
from dataclasses import field as dataclass_field
from enum import Enum
from typing import Any

from latency_meta_mdp.expert_realization.config import (
    SUBSEED_TAGS,
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


def derive_realization_seed(
    task_instance_id: TaskInstanceId,
    realization_index: int,
    structured_expert_config_sha256: str,
) -> int:
    if (
        isinstance(realization_index, bool)
        or not isinstance(realization_index, int)
        or not 0 <= realization_index <= 7
    ):
        raise ValueError("realization_index must be an integer in the canonical range 0..7")
    _require_sha256(structured_expert_config_sha256, name="structured_expert_config_sha256")
    return _seed(
        {
            "task_instance_id": task_instance_id.to_mapping(),
            "realization_index": realization_index,
            "structured_expert_config_sha256": structured_expert_config_sha256,
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
    structured_expert_config_sha256: InitVar[str]
    realization_seed: int = dataclass_field(init=False)

    def __post_init__(self, structured_expert_config_sha256: str) -> None:
        if not isinstance(self.task_instance_id, TaskInstanceId):
            raise TypeError("task_instance_id must be a TaskInstanceId")
        if type(self.realization_index) is not int or not 0 <= self.realization_index <= 7:
            raise ValueError("realization_index must be an integer in the canonical range 0..7")
        _require_sha256(structured_expert_config_sha256, name="structured_expert_config_sha256")
        object.__setattr__(
            self,
            "realization_seed",
            derive_realization_seed(
                self.task_instance_id,
                self.realization_index,
                structured_expert_config_sha256,
            ),
        )

    def subseeds(self) -> dict[str, int]:
        return {tag: derive_subseed(self.realization_seed, tag) for tag in SUBSEED_TAGS}

    def to_mapping(self) -> dict[str, Any]:
        return {
            "task_instance_id": self.task_instance_id.to_mapping(),
            "realization_index": self.realization_index,
            "realization_seed": self.realization_seed,
        }

    @classmethod
    def from_mapping(
        cls, mapping: Any, *, structured_expert_config_sha256: str
    ) -> ExpertRealizationKey:
        raw = _strict_identity_mapping(
            mapping,
            {"task_instance_id", "realization_index", "realization_seed"},
            name="ExpertRealizationKey",
        )
        stored_seed = raw["realization_seed"]
        if type(stored_seed) is not int or not 0 <= stored_seed < 2**64:
            raise ValueError("realization_seed must be a uint64")
        key = cls(
            TaskInstanceId.from_mapping(raw["task_instance_id"]),
            raw["realization_index"],
            structured_expert_config_sha256,
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
        cls, mapping: Any, *, structured_expert_config_sha256: str
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
    def from_mapping(cls, mapping: Any, *, structured_expert_config_sha256: str) -> AttemptId:
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
    interception_tick: int
    interception_lead_seconds: float
    pregrasp_height_m: float
    lateral_offset_m: float | None
    lateral_direction_sign: int | None
    tracking_error_clip_m: float
    close_dwell_ticks: int
    lift_lateral_offset_m: float
    lift_lateral_direction_sign: int
    lift_vertical_offset_m: float
    fixed_orientation: bool
    rotation_action_variation: bool
    iid_per_tick_action_noise: bool
    config: InitVar[StructuredExpertConfig]

    def __post_init__(self, config: StructuredExpertConfig) -> None:
        if not isinstance(config, StructuredExpertConfig):
            raise TypeError("config must be a StructuredExpertConfig")
        if not isinstance(self.family, StrategyFamily):
            raise TypeError("family must be a StrategyFamily")
        if type(self.interception_tick) is not int:
            raise ValueError("interception_tick must be an integer")
        if (
            not config.interception_tick_ranges[self.family.value][0]
            <= self.interception_tick
            <= config.interception_tick_ranges[self.family.value][1]
        ):
            raise ValueError("interception_tick is outside the family-specific range")
        values = {
            "interception_lead_seconds": (
                self.interception_lead_seconds,
                config.interception_lead_seconds,
            ),
            "pregrasp_height_m": (self.pregrasp_height_m, config.pregrasp_height_m),
            "tracking_error_clip_m": (self.tracking_error_clip_m, config.tracking_error_clip_m),
            "lift_lateral_offset_m": (self.lift_lateral_offset_m, config.lift_lateral_offset_m),
            "lift_vertical_offset_m": (self.lift_vertical_offset_m, config.lift_vertical_offset_m),
        }
        for name, (value, bounds) in values.items():
            if (
                type(value) is not float
                or not math.isfinite(value)
                or not bounds[0] <= value <= bounds[1]
            ):
                raise ValueError(f"{name} is outside the structured expert bounds")
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
        if (
            type(self.close_dwell_ticks) is not int
            or self.close_dwell_ticks not in config.close_dwell_ticks
        ):
            raise ValueError("close_dwell_ticks is outside the structured expert choices")
        if type(self.lift_lateral_direction_sign) is not int or (
            self.lift_lateral_direction_sign not in (-1, 1)
        ):
            raise ValueError("lift_lateral_direction_sign must be -1 or +1")
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
            "interception_tick": self.interception_tick,
            "interception_lead_seconds": self.interception_lead_seconds,
            "pregrasp_height_m": self.pregrasp_height_m,
            "lateral_offset_m": self.lateral_offset_m,
            "lateral_direction_sign": self.lateral_direction_sign,
            "tracking_error_clip_m": self.tracking_error_clip_m,
            "close_dwell_ticks": self.close_dwell_ticks,
            "lift_lateral_offset_m": self.lift_lateral_offset_m,
            "lift_lateral_direction_sign": self.lift_lateral_direction_sign,
            "lift_vertical_offset_m": self.lift_vertical_offset_m,
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
    keypose_seed = derive_subseed(realization_seed, "keypose")
    timing_seed = derive_subseed(realization_seed, "timing")
    lateral_offset = (
        _draw_uniform(strategy_seed, field="lateral_offset_m", bounds=config.lateral_offset_m)
        if family_value is StrategyFamily.LATERAL_ARC
        else None
    )
    lateral_direction_sign = (
        -1
        if _seed({"field": "lateral_direction_sign", "seed": keypose_seed}) % 2 == 0
        else 1
    )
    lift_lateral_direction_sign = (
        -1
        if _seed({"field": "lift_lateral_direction_sign", "seed": keypose_seed}) % 2 == 0
        else 1
    )
    return StrategyParameters(
        family=family_value,
        interception_tick=_draw_int(
            strategy_seed,
            field="interception_tick",
            bounds=config.interception_tick_ranges[family_value.value],
        ),
        interception_lead_seconds=_draw_uniform(
            strategy_seed,
            field="interception_lead_seconds",
            bounds=config.interception_lead_seconds,
        ),
        pregrasp_height_m=_draw_uniform(
            strategy_seed, field="pregrasp_height_m", bounds=config.pregrasp_height_m
        ),
        lateral_offset_m=lateral_offset,
        lateral_direction_sign=(
            lateral_direction_sign if family_value is StrategyFamily.LATERAL_ARC else None
        ),
        tracking_error_clip_m=_draw_uniform(
            strategy_seed,
            field="tracking_error_clip_m",
            bounds=config.tracking_error_clip_m,
        ),
        close_dwell_ticks=config.close_dwell_ticks[
            _seed({"field": "close_dwell_ticks", "seed": timing_seed})
            % len(config.close_dwell_ticks)
        ],
        lift_lateral_offset_m=_draw_uniform(
            strategy_seed,
            field="lift_lateral_offset_m",
            bounds=config.lift_lateral_offset_m,
        ),
        lift_lateral_direction_sign=lift_lateral_direction_sign,
        lift_vertical_offset_m=_draw_uniform(
            strategy_seed,
            field="lift_vertical_offset_m",
            bounds=config.lift_vertical_offset_m,
        ),
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
                            structured_expert_config_sha256=self.structured_expert_config_sha256,
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
