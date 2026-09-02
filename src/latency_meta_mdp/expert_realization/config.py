"""Strict, versioned configuration for the structured Panda-ball expert pilot."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CANONICAL_FAMILIES = (
    "canonical_direct",
    "early_high_arc",
    "lateral_arc",
    "time_shifted_smooth",
)
SUBSEED_TAGS = ("strategy", "keypose", "planner", "timing")


def _strict_mapping(raw: Any, fields: set[str], *, name: str) -> dict[str, Any]:
    if type(raw) is not dict:
        raise ValueError(f"{name} config must be a YAML mapping")
    unknown = sorted(set(raw).difference(fields))
    missing = sorted(fields.difference(raw))
    if unknown:
        raise ValueError(f"unknown {name} config fields: {unknown}")
    if missing:
        raise ValueError(f"missing {name} config fields: {missing}")
    return raw


def _load_mapping(path: Path, fields: set[str], *, name: str) -> dict[str, Any]:
    return _strict_mapping(yaml.safe_load(path.read_text(encoding="utf-8")), fields, name=name)


def _require_int(value: Any, *, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    return value


def _require_float(value: Any, *, name: str) -> float:
    if type(value) is not float:
        raise TypeError(f"{name} must be a float")
    return value


def _require_str(value: Any, *, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    return value


def _require_bool(value: Any, *, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")
    return value


def _tuple_of(value: Any, *, name: str, check: Callable[[Any], Any]) -> tuple[Any, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    for item in value:
        check(item)
    return value


def _yaml_tuple(value: Any, *, name: str, check: Callable[[Any], Any]) -> tuple[Any, ...]:
    if type(value) is not list:
        raise TypeError(f"{name} must be a YAML list")
    for item in value:
        check(item)
    return tuple(value)


def _pair(value: Any, *, name: str) -> tuple[float, float]:
    result = _yaml_tuple(value, name=name, check=lambda item: _require_float(item, name=name))
    if len(result) != 2 or result[0] > result[1]:
        raise ValueError(f"{name} must be an ordered pair")
    return result  # type: ignore[return-value]


def _float_pair(value: Any, *, name: str) -> tuple[float, float]:
    result = _tuple_of(value, name=name, check=lambda item: _require_float(item, name=name))
    if len(result) != 2 or result[0] > result[1]:
        raise ValueError(f"{name} must be an ordered pair")
    return result  # type: ignore[return-value]


@dataclass(frozen=True)
class StructuredExpertConfig:
    schema_version: int
    expert_id: str
    action_contract_id: str
    decision_source_tick: int
    shared_prefix_policy: str
    families: tuple[str, ...]
    samples_per_family: int
    interception_lead_seconds: tuple[float, float]
    pregrasp_height_m: tuple[float, float]
    lateral_offset_m: tuple[float, float]
    tracking_error_clip_m: tuple[float, float]
    close_dwell_ticks: tuple[int, ...]
    lift_lateral_offset_m: tuple[float, float]
    lift_vertical_offset_m: tuple[float, float]
    interception_tick_ranges: dict[str, tuple[int, int]]
    fixed_orientation: bool
    rotation_action_variation: bool
    iid_per_tick_action_noise: bool
    subseed_tags: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_int(self.schema_version, name="schema_version")
        for field in ("expert_id", "action_contract_id", "shared_prefix_policy"):
            _require_str(getattr(self, field), name=field)
        _require_int(self.decision_source_tick, name="decision_source_tick")
        _tuple_of(
            self.families, name="families", check=lambda item: _require_str(item, name="family")
        )
        _require_int(self.samples_per_family, name="samples_per_family")
        for field in (
            "interception_lead_seconds",
            "pregrasp_height_m",
            "lateral_offset_m",
            "tracking_error_clip_m",
            "lift_lateral_offset_m",
            "lift_vertical_offset_m",
        ):
            _float_pair(getattr(self, field), name=field)
        _tuple_of(
            self.close_dwell_ticks,
            name="close_dwell_ticks",
            check=lambda item: _require_int(item, name="close_dwell_ticks item"),
        )
        if type(self.interception_tick_ranges) is not dict:
            raise TypeError("interception_tick_ranges must be a mapping")
        for family, bounds in self.interception_tick_ranges.items():
            _require_str(family, name="interception_tick_ranges key")
            values = _tuple_of(
                bounds,
                name="interception tick range",
                check=lambda item: _require_int(item, name="interception tick range item"),
            )
            if len(values) != 2:
                raise ValueError("interception tick ranges must be pairs")
        for field in (
            "fixed_orientation",
            "rotation_action_variation",
            "iid_per_tick_action_noise",
        ):
            _require_bool(getattr(self, field), name=field)
        _tuple_of(
            self.subseed_tags,
            name="subseed_tags",
            check=lambda item: _require_str(item, name="subseed tag"),
        )
        if self.schema_version != 1:
            raise ValueError("structured expert schema_version must be 1")
        if (
            self.expert_id != "panda_ball_structured_v1"
            or self.action_contract_id != "panda_osc_pose_delta_v1"
            or self.decision_source_tick != 5
            or self.shared_prefix_policy != "settle_open_hold_v1"
        ):
            raise ValueError("unsupported structured expert semantic identity")
        if self.families != CANONICAL_FAMILIES or self.samples_per_family != 2:
            raise ValueError("structured expert family layout is invalid")
        expected_ranges = {
            "canonical_direct": (75, 105),
            "early_high_arc": (60, 85),
            "lateral_arc": (75, 110),
            "time_shifted_smooth": (100, 125),
        }
        if self.interception_tick_ranges != expected_ranges:
            raise ValueError("structured expert interception ranges are invalid")
        if (
            self.interception_lead_seconds != (0.14, 0.26)
            or self.pregrasp_height_m != (0.08, 0.13)
            or self.lateral_offset_m != (0.015, 0.04)
            or self.tracking_error_clip_m != (0.022, 0.038)
            or self.close_dwell_ticks != (0, 1, 2, 3, 4)
            or self.lift_lateral_offset_m != (0.0, 0.02)
            or self.lift_vertical_offset_m != (0.14, 0.19)
        ):
            raise ValueError("structured expert proposal bounds are invalid")
        if (
            self.fixed_orientation is not True
            or self.rotation_action_variation is not False
            or self.iid_per_tick_action_noise is not False
            or self.subseed_tags != SUBSEED_TAGS
        ):
            raise ValueError("structured expert action or seed contract is invalid")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "expert_id": self.expert_id,
            "action_contract_id": self.action_contract_id,
            "decision_source_tick": self.decision_source_tick,
            "shared_prefix_policy": self.shared_prefix_policy,
            "families": list(self.families),
            "samples_per_family": self.samples_per_family,
            "interception_lead_seconds": list(self.interception_lead_seconds),
            "pregrasp_height_m": list(self.pregrasp_height_m),
            "lateral_offset_m": list(self.lateral_offset_m),
            "tracking_error_clip_m": list(self.tracking_error_clip_m),
            "close_dwell_ticks": list(self.close_dwell_ticks),
            "lift_lateral_offset_m": list(self.lift_lateral_offset_m),
            "lift_vertical_offset_m": list(self.lift_vertical_offset_m),
            "interception_tick_ranges": {
                family: list(bounds) for family, bounds in self.interception_tick_ranges.items()
            },
            "fixed_orientation": self.fixed_orientation,
            "rotation_action_variation": self.rotation_action_variation,
            "iid_per_tick_action_noise": self.iid_per_tick_action_noise,
            "subseed_tags": list(self.subseed_tags),
        }


def load_structured_expert_config(path: Path) -> StructuredExpertConfig:
    raw = _load_mapping(
        path, set(StructuredExpertConfig.__dataclass_fields__), name="structured expert"
    )
    ranges = raw["interception_tick_ranges"]
    if type(ranges) is not dict or set(ranges) != set(CANONICAL_FAMILIES):
        raise ValueError("interception_tick_ranges must cover the canonical families")
    parsed_ranges: dict[str, tuple[int, int]] = {}
    for family in CANONICAL_FAMILIES:
        parsed = _yaml_tuple(
            ranges[family],
            name=f"interception tick range for {family}",
            check=lambda item: _require_int(item, name="interception tick range item"),
        )
        if len(parsed) != 2:
            raise ValueError(f"interception tick range for {family} is invalid")
        parsed_ranges[family] = parsed  # type: ignore[assignment]
    return StructuredExpertConfig(
        schema_version=raw["schema_version"],
        expert_id=raw["expert_id"],
        action_contract_id=raw["action_contract_id"],
        decision_source_tick=raw["decision_source_tick"],
        shared_prefix_policy=raw["shared_prefix_policy"],
        families=_yaml_tuple(
            raw["families"], name="families", check=lambda item: _require_str(item, name="family")
        ),
        samples_per_family=raw["samples_per_family"],
        interception_lead_seconds=_pair(
            raw["interception_lead_seconds"], name="interception_lead_seconds"
        ),
        pregrasp_height_m=_pair(raw["pregrasp_height_m"], name="pregrasp_height_m"),
        lateral_offset_m=_pair(raw["lateral_offset_m"], name="lateral_offset_m"),
        tracking_error_clip_m=_pair(raw["tracking_error_clip_m"], name="tracking_error_clip_m"),
        close_dwell_ticks=_yaml_tuple(
            raw["close_dwell_ticks"],
            name="close_dwell_ticks",
            check=lambda item: _require_int(item, name="close_dwell_ticks item"),
        ),
        lift_lateral_offset_m=_pair(raw["lift_lateral_offset_m"], name="lift_lateral_offset_m"),
        lift_vertical_offset_m=_pair(raw["lift_vertical_offset_m"], name="lift_vertical_offset_m"),
        interception_tick_ranges=parsed_ranges,
        fixed_orientation=raw["fixed_orientation"],
        rotation_action_variation=raw["rotation_action_variation"],
        iid_per_tick_action_noise=raw["iid_per_tick_action_noise"],
        subseed_tags=_yaml_tuple(
            raw["subseed_tags"],
            name="subseed_tags",
            check=lambda item: _require_str(item, name="subseed tag"),
        ),
    )


@dataclass(frozen=True)
class CuroboPlannerConfig:
    schema_version: int
    runtime_config: str
    robot: str
    planner_candidate_count: int
    planner_invocation_timeout_seconds: float
    fk_translation_tolerance_m: float
    fk_rotation_tolerance_degrees: float
    planner_worker_import_boundary: str

    def __post_init__(self) -> None:
        _require_int(self.schema_version, name="schema_version")
        for field in ("runtime_config", "robot", "planner_worker_import_boundary"):
            _require_str(getattr(self, field), name=field)
        _require_int(self.planner_candidate_count, name="planner_candidate_count")
        for field in (
            "planner_invocation_timeout_seconds",
            "fk_translation_tolerance_m",
            "fk_rotation_tolerance_degrees",
        ):
            _require_float(getattr(self, field), name=field)
        if (
            self.schema_version != 1
            or self.runtime_config != "configs/expert_realization/curobo_runtime.yaml"
            or self.robot != "franka.yml"
            or self.planner_candidate_count != 8
            or self.planner_invocation_timeout_seconds != 5.0
            or self.fk_translation_tolerance_m != 0.001
            or self.fk_rotation_tolerance_degrees != 0.5
            or self.planner_worker_import_boundary != "subprocess_json_npz"
        ):
            raise ValueError("unsupported CuRobo Panda planner contract")


def load_curobo_planner_config(path: Path) -> CuroboPlannerConfig:
    return CuroboPlannerConfig(
        **_load_mapping(path, set(CuroboPlannerConfig.__dataclass_fields__), name="CuRobo planner")
    )


@dataclass(frozen=True)
class PilotConfig:
    schema_version: int
    pilot_id: str
    task_instance_seed_start: int
    task_instance_seed_stop: int
    levels: tuple[int, ...]
    families: tuple[str, ...]
    samples_per_family: int
    maximum_infrastructure_attempts_per_realization: int
    record_profile: str
    camera_width: int
    camera_height: int
    review_video_fps: int
    bounded_review_only: bool
    formal_training_authorized: bool
    formal_dino_cache_authorized: bool
    jepa_training_authorized: bool
    policy_training_authorized: bool
    meta_policy_training_authorized: bool

    def __post_init__(self) -> None:
        for field in (
            "schema_version",
            "task_instance_seed_start",
            "task_instance_seed_stop",
            "samples_per_family",
            "maximum_infrastructure_attempts_per_realization",
            "camera_width",
            "camera_height",
            "review_video_fps",
        ):
            _require_int(getattr(self, field), name=field)
        for field in ("pilot_id", "record_profile"):
            _require_str(getattr(self, field), name=field)
        _tuple_of(self.levels, name="levels", check=lambda item: _require_int(item, name="level"))
        _tuple_of(
            self.families, name="families", check=lambda item: _require_str(item, name="family")
        )
        for field in (
            "bounded_review_only",
            "formal_training_authorized",
            "formal_dino_cache_authorized",
            "jepa_training_authorized",
            "policy_training_authorized",
            "meta_policy_training_authorized",
        ):
            _require_bool(getattr(self, field), name=field)
        if (
            self.schema_version != 1
            or self.pilot_id != "panda-ball-structured-paired-4000-4019-v1"
            or self.task_instance_seed_start != 4000
            or self.task_instance_seed_stop != 4020
            or self.levels != (1, 2, 3)
            or self.families != CANONICAL_FAMILIES
            or self.samples_per_family != 2
            or self.maximum_infrastructure_attempts_per_realization != 2
            or self.record_profile != "pilot_debug"
            or self.camera_width != 256
            or self.camera_height != 256
            or self.review_video_fps != 50
            or self.bounded_review_only is not True
        ):
            raise ValueError("unsupported structured expert pilot contract")
        if any(
            getattr(self, field) is not False
            for field in (
                "formal_training_authorized",
                "formal_dino_cache_authorized",
                "jepa_training_authorized",
                "policy_training_authorized",
                "meta_policy_training_authorized",
            )
        ):
            raise ValueError("the pilot cannot authorize training")

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(range(self.task_instance_seed_start, self.task_instance_seed_stop))

    def training_eligibility(self) -> dict[str, bool]:
        return {
            "formal_training_authorized": self.formal_training_authorized,
            "formal_dino_cache_authorized": self.formal_dino_cache_authorized,
            "jepa_training_authorized": self.jepa_training_authorized,
            "policy_training_authorized": self.policy_training_authorized,
            "meta_policy_training_authorized": self.meta_policy_training_authorized,
        }

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "pilot_id": self.pilot_id,
            "task_instance_seed_start": self.task_instance_seed_start,
            "task_instance_seed_stop": self.task_instance_seed_stop,
            "levels": list(self.levels),
            "families": list(self.families),
            "samples_per_family": self.samples_per_family,
            "maximum_infrastructure_attempts_per_realization": (
                self.maximum_infrastructure_attempts_per_realization
            ),
            "record_profile": self.record_profile,
            "camera_width": self.camera_width,
            "camera_height": self.camera_height,
            "review_video_fps": self.review_video_fps,
            "bounded_review_only": self.bounded_review_only,
            **self.training_eligibility(),
        }


def load_pilot_config(path: Path) -> PilotConfig:
    raw = _load_mapping(path, set(PilotConfig.__dataclass_fields__), name="pilot")
    return PilotConfig(
        **{
            **raw,
            "levels": _yaml_tuple(
                raw["levels"], name="levels", check=lambda item: _require_int(item, name="level")
            ),
            "families": _yaml_tuple(
                raw["families"],
                name="families",
                check=lambda item: _require_str(item, name="family"),
            ),
        }
    )


@dataclass(frozen=True)
class PilotGateConfig:
    schema_version: int
    pregrasp_tracking_translation_m: float
    pregrasp_tracking_rotation_degrees: float
    non_contact_environment_clearance_m: float
    osc_action_bounds: tuple[float, float]
    pre_handoff_saturation_fraction: float
    joint_position_margin_rad: float
    joint_velocity_fraction_of_model_limit: float
    eef_speed_mps: float
    eef_acceleration_mps2: float
    eef_jerk_mps3: float
    reference_to_achieved_eef_error_outside_contact_m: float
    unintended_pregrasp_ball_contacts: int
    intentional_contact_penetration_m: float
    peak_pad_ball_impulse_per_physics_contact_event_ns: float
    near_duplicate_d20_xyz_action_rms: float
    near_duplicate_eef_frechet_m: float

    def __post_init__(self) -> None:
        _require_int(self.schema_version, name="schema_version")
        _float_pair(self.osc_action_bounds, name="osc_action_bounds")
        _require_int(
            self.unintended_pregrasp_ball_contacts, name="unintended_pregrasp_ball_contacts"
        )
        for field in (
            "pregrasp_tracking_translation_m",
            "pregrasp_tracking_rotation_degrees",
            "non_contact_environment_clearance_m",
            "pre_handoff_saturation_fraction",
            "joint_position_margin_rad",
            "joint_velocity_fraction_of_model_limit",
            "eef_speed_mps",
            "eef_acceleration_mps2",
            "eef_jerk_mps3",
            "reference_to_achieved_eef_error_outside_contact_m",
            "intentional_contact_penetration_m",
            "peak_pad_ball_impulse_per_physics_contact_event_ns",
            "near_duplicate_d20_xyz_action_rms",
            "near_duplicate_eef_frechet_m",
        ):
            _require_float(getattr(self, field), name=field)
        expected = {
            "schema_version": 1,
            "pregrasp_tracking_translation_m": 0.01,
            "pregrasp_tracking_rotation_degrees": 2.0,
            "non_contact_environment_clearance_m": 0.002,
            "osc_action_bounds": (-1.0, 1.0),
            "pre_handoff_saturation_fraction": 0.05,
            "joint_position_margin_rad": 1.0e-4,
            "joint_velocity_fraction_of_model_limit": 0.8,
            "eef_speed_mps": 0.75,
            "eef_acceleration_mps2": 5.0,
            "eef_jerk_mps3": 100.0,
            "reference_to_achieved_eef_error_outside_contact_m": 0.015,
            "unintended_pregrasp_ball_contacts": 0,
            "intentional_contact_penetration_m": 0.002,
            "peak_pad_ball_impulse_per_physics_contact_event_ns": 1.0,
            "near_duplicate_d20_xyz_action_rms": 0.02,
            "near_duplicate_eef_frechet_m": 0.005,
        }
        if any(getattr(self, name) != value for name, value in expected.items()):
            raise ValueError("unsupported structured expert pilot gate contract")


def load_pilot_gate_config(path: Path) -> PilotGateConfig:
    raw = _load_mapping(path, set(PilotGateConfig.__dataclass_fields__), name="pilot gate")
    return PilotGateConfig(
        **{
            **raw,
            "osc_action_bounds": _pair(raw["osc_action_bounds"], name="osc_action_bounds"),
        }
    )
