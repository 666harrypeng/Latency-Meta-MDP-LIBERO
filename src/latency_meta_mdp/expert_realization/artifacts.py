"""Strict hierarchical manifests and durable structured-expert publication."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import stat
import uuid
from dataclasses import dataclass, fields
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from latency_meta_mdp.expert_realization.contracts import (
    AttemptId,
    ExpertRealizationKey,
    FailureClass,
    TaskInstanceId,
)
from latency_meta_mdp.expert_realization.recording_contracts import ImplementationIdentity

TASK_INSTANCE_FORMAT = "structured_expert_task_instance_v1"
PLAN_SET_FORMAT = "structured_expert_plan_set_v1"
ATTEMPT_FORMAT = "structured_expert_attempt_v1"
PILOT_FORMAT = "structured_expert_pilot_v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LEGACY_IDS = frozenset(
    {
        "synchronized_episode_npz_v3",
        "panda_ball_bulk_first_tranche_v1",
        "panda_ball_bulk_range_v1",
        "expert_pilot_run_v1",
        "panda_ball_formal_corpus_v1",
        "vision_feature_cache_v1",
    }
)
_TRAINING_FLAGS = frozenset(
    {
        "formal_training_authorized",
        "formal_dino_cache_authorized",
        "jepa_training_authorized",
        "policy_training_authorized",
        "meta_policy_training_authorized",
    }
)
_TASK_REF = re.compile(r"^task_instances/L([123])/seed-([0-9]+)/task_instance/manifest[.]json$")
_PLAN_REF = re.compile(r"^task_instances/L([123])/seed-([0-9]+)/plan_set/manifest[.]json$")
_ATTEMPT_REF = re.compile(
    r"^task_instances/L([123])/seed-([0-9]+)/attempts/"
    r"realization-((?:[0-9]{3}|[1-9][0-9]{3,}))/attempt-([0-9]{3})/manifest[.]json$"
)


def _strict(value: Any, expected: set[str], *, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be a mapping")
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown or missing:
        raise ValueError(f"{name} fields mismatch; unknown={unknown}, missing={missing}")
    return value


def _sha(value: Any, *, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _text(value: Any, *, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_inventory_key(path: str, pattern: re.Pattern[str], *, name: str) -> tuple[int, ...]:
    match = pattern.fullmatch(path)
    if match is None:
        raise ValueError(f"{name} path does not use the locked hierarchical layout")
    return tuple(int(value) for value in match.groups())


def _task_base(task_instance_id: TaskInstanceId) -> str:
    return f"task_instances/L{task_instance_id.level}/seed-{task_instance_id.task_instance_seed}"


def _task_manifest_path(task_instance_id: TaskInstanceId) -> str:
    return f"{_task_base(task_instance_id)}/task_instance/manifest.json"


def _plan_manifest_path(task_instance_id: TaskInstanceId) -> str:
    return f"{_task_base(task_instance_id)}/plan_set/manifest.json"


def _require_target_suffix(target: Path, expected: str) -> None:
    parts = PurePosixPath(target.absolute().as_posix()).parts
    expected_parts = PurePosixPath(expected).parts
    if tuple(parts[-len(expected_parts) :]) != expected_parts:
        raise ValueError(f"publication target must end with the canonical path {expected}")


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _load_json_bytes(payload: bytes, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} must contain valid JSON") from error
    if type(value) is not dict:
        raise ValueError(f"{name} must contain a JSON mapping")
    return value


def _safe_relative(path: Any, *, name: str = "artifact path") -> str:
    if type(path) is not str or not path or "\\" in path:
        raise ValueError(f"{name} must be a normalized relative POSIX path")
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != path or "." in pure.parts:
        raise ValueError(f"{name} must be a normalized relative POSIX path")
    return path


def _resolve(root: Path, relative: str) -> Path:
    safe = _safe_relative(relative)
    root_absolute = _assert_no_symlink_ancestry(root, allow_missing=False)
    candidate_lexical = root_absolute / safe
    _assert_no_symlink_ancestry(candidate_lexical, allow_missing=False)
    candidate = candidate_lexical.resolve(strict=True)
    physical_root = root_absolute.resolve(strict=True)
    try:
        candidate.relative_to(physical_root)
    except ValueError as error:
        raise ValueError("artifact path escapes artifact root") from error
    _reject_hidden_components(candidate_lexical)
    _reject_hidden_components(candidate)
    return candidate


def _reject_legacy_payload(path: str, payload: bytes) -> None:
    if any(identifier.encode() in payload for identifier in _LEGACY_IDS):
        raise ValueError(f"legacy format is forbidden in {path}")
    if path.endswith(".json"):
        value = _load_json_bytes(payload, name=path)
        if value.get("format_id") in _LEGACY_IDS:
            raise ValueError(f"legacy format is forbidden in {path}")
    elif path.endswith(".jsonl"):
        for line in payload.splitlines():
            if not line.strip():
                continue
            value = _load_json_bytes(line, name=path)
            if value.get("format_id") in _LEGACY_IDS:
                raise ValueError(f"legacy format is forbidden in {path}")


@dataclass(frozen=True)
class ArtifactRef:
    path: str
    sha256: str

    def __post_init__(self) -> None:
        _safe_relative(self.path)
        _sha(self.sha256, name="artifact sha256")

    def to_mapping(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}

    @classmethod
    def from_mapping(cls, mapping: Any) -> ArtifactRef:
        return cls(**_strict(mapping, {"path", "sha256"}, name=cls.__name__))


@dataclass(frozen=True)
class VerifiedTaskInstanceBundle:
    manifest: TaskInstanceManifest
    manifest_bytes: bytes
    manifest_sha256: str
    files: MappingProxyType


@dataclass(frozen=True)
class VerifiedFrozenPlanBundle:
    manifest: FrozenPlanSetManifest
    manifest_bytes: bytes
    manifest_sha256: str
    files: MappingProxyType
    task: VerifiedTaskInstanceBundle


@dataclass(frozen=True)
class VerifiedAttemptBundle:
    manifest: AttemptManifest
    manifest_bytes: bytes
    status_bytes: bytes
    qualification_bytes: bytes | None
    episode: Any | None
    plan: VerifiedFrozenPlanBundle


@dataclass(frozen=True)
class TaskInstanceManifest:
    schema_version: int
    format_id: str
    task_instance_id: TaskInstanceId
    task_id: str
    instruction: str
    decision_source_tick: int
    shared_prefix_policy: str
    task_config_sha256: str
    motion_config_sha256: str
    runtime_config_sha256: str
    controller_config_sha256: str
    implementation: ImplementationIdentity
    artifacts: dict[str, str]

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.format_id != TASK_INSTANCE_FORMAT
        ):
            raise ValueError("unsupported task-instance manifest format")
        if not isinstance(self.task_instance_id, TaskInstanceId):
            raise TypeError("task_instance_id must be a TaskInstanceId")
        if self.task_id != "dynamic_grasp_lift":
            raise ValueError("unsupported task_id")
        _text(self.instruction, name="instruction")
        if type(self.decision_source_tick) is not int or self.decision_source_tick < 0:
            raise ValueError("decision_source_tick must be non-negative")
        _text(self.shared_prefix_policy, name="shared_prefix_policy")
        for name in (
            "task_config_sha256",
            "motion_config_sha256",
            "runtime_config_sha256",
            "controller_config_sha256",
        ):
            _sha(getattr(self, name), name=name)
        if not isinstance(self.implementation, ImplementationIdentity):
            raise TypeError("implementation must be an ImplementationIdentity")
        expected = {"task_instance.json", "motion_profile.json", "initial_state.npz"}
        if type(self.artifacts) is not dict or set(self.artifacts) != expected:
            raise ValueError(
                "task instance artifacts must contain exactly the three locked children"
            )
        for name, digest in self.artifacts.items():
            _safe_relative(name)
            _sha(digest, name=f"artifact {name}")
        if self.task_instance_id.motion_profile_sha256 != self.artifacts["motion_profile.json"]:
            raise ValueError("motion profile identity hash mismatch")
        if self.task_instance_id.initial_state_sha256 != self.artifacts["initial_state.npz"]:
            raise ValueError("initial state identity hash mismatch")
        object.__setattr__(self, "artifacts", MappingProxyType(dict(self.artifacts)))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "format_id": self.format_id,
            "task_instance_id": self.task_instance_id.to_mapping(),
            "task_id": self.task_id,
            "instruction": self.instruction,
            "decision_source_tick": self.decision_source_tick,
            "shared_prefix_policy": self.shared_prefix_policy,
            "task_config_sha256": self.task_config_sha256,
            "motion_config_sha256": self.motion_config_sha256,
            "runtime_config_sha256": self.runtime_config_sha256,
            "controller_config_sha256": self.controller_config_sha256,
            "implementation": self.implementation.to_mapping(),
            "artifacts": dict(self.artifacts),
        }

    @classmethod
    def from_mapping(cls, mapping: Any) -> TaskInstanceManifest:
        raw = _strict(mapping, {item.name for item in fields(cls)}, name=cls.__name__).copy()
        raw["task_instance_id"] = TaskInstanceId.from_mapping(raw["task_instance_id"])
        raw["implementation"] = ImplementationIdentity.from_mapping(raw["implementation"])
        return cls(**raw)


@dataclass(frozen=True)
class FrozenPlanRow:
    expert_realization_key: ExpertRealizationKey
    strategy: ArtifactRef
    planner_candidates: ArtifactRef
    selection_status: str
    selected_candidate_fingerprint: str | None
    selected_reference: ArtifactRef | None

    def __post_init__(self) -> None:
        if not isinstance(self.expert_realization_key, ExpertRealizationKey):
            raise TypeError("expert_realization_key must be an ExpertRealizationKey")
        if not isinstance(self.strategy, ArtifactRef) or not isinstance(
            self.planner_candidates, ArtifactRef
        ):
            raise TypeError("plan row artifacts must be ArtifactRef values")
        if self.selection_status not in {"selected", "planner_failure"}:
            raise ValueError("selection_status must be selected or planner_failure")
        index = self.expert_realization_key.realization_index
        expected_strategy = f"strategies/realization-{index:03d}.json"
        expected_candidates = f"planner_candidates/realization-{index:03d}.jsonl"
        if (
            self.strategy.path != expected_strategy
            or self.planner_candidates.path != expected_candidates
        ):
            raise ValueError("plan row artifact path does not match its realization index")
        if self.selection_status == "selected":
            _text(self.selected_candidate_fingerprint, name="selected_candidate_fingerprint")
            if not isinstance(self.selected_reference, ArtifactRef):
                raise ValueError("selected rows require a selected reference")
            if self.selected_reference.path != f"selected_references/realization-{index:03d}.npz":
                raise ValueError("selected reference path does not match its realization index")
        elif self.selected_candidate_fingerprint is not None or self.selected_reference is not None:
            raise ValueError("planner failures cannot have selected candidate data")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "expert_realization_key": self.expert_realization_key.to_mapping(),
            "strategy": self.strategy.to_mapping(),
            "planner_candidates": self.planner_candidates.to_mapping(),
            "selection_status": self.selection_status,
            "selected_candidate_fingerprint": self.selected_candidate_fingerprint,
            "selected_reference": self.selected_reference.to_mapping()
            if self.selected_reference
            else None,
        }

    @classmethod
    def from_mapping(cls, mapping: Any) -> FrozenPlanRow:
        raw = _strict(mapping, {item.name for item in fields(cls)}, name=cls.__name__).copy()
        raw["expert_realization_key"] = ExpertRealizationKey.from_mapping(
            raw["expert_realization_key"]
        )
        raw["strategy"] = ArtifactRef.from_mapping(raw["strategy"])
        raw["planner_candidates"] = ArtifactRef.from_mapping(raw["planner_candidates"])
        if raw["selected_reference"] is not None:
            raw["selected_reference"] = ArtifactRef.from_mapping(raw["selected_reference"])
        return cls(**raw)


@dataclass(frozen=True)
class FrozenPlanSetManifest:
    schema_version: int
    format_id: str
    task_instance_id: TaskInstanceId
    task_instance_manifest: ArtifactRef
    structured_expert_config_sha256: str
    curobo_planner_config_sha256: str
    realization_universe_sha256: str
    realization_slots: tuple[int, ...]
    implementation: ImplementationIdentity
    realizations: tuple[FrozenPlanRow, ...]

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.format_id != PLAN_SET_FORMAT
        ):
            raise ValueError("unsupported frozen plan-set manifest format")
        if not isinstance(self.task_instance_id, TaskInstanceId) or not isinstance(
            self.task_instance_manifest, ArtifactRef
        ):
            raise TypeError("frozen plan-set task fields have wrong types")
        if self.task_instance_manifest.path != _task_manifest_path(self.task_instance_id):
            raise ValueError("task-instance manifest path does not use the locked layout")
        _sha(self.structured_expert_config_sha256, name="structured_expert_config_sha256")
        _sha(self.curobo_planner_config_sha256, name="curobo_planner_config_sha256")
        _sha(self.realization_universe_sha256, name="realization_universe_sha256")
        if (
            type(self.realization_slots) is not tuple
            or not self.realization_slots
            or self.realization_slots != tuple(range(len(self.realization_slots)))
        ):
            raise ValueError("realization universe slots must be contiguous from zero")
        if not isinstance(self.implementation, ImplementationIdentity):
            raise TypeError("implementation must be an ImplementationIdentity")
        if type(self.realizations) is not tuple or not self.realizations:
            raise ValueError("realizations must be a non-empty tuple")
        if any(not isinstance(row, FrozenPlanRow) for row in self.realizations):
            raise TypeError("realizations must contain FrozenPlanRow values")
        if any(
            row.expert_realization_key.task_instance_id != self.task_instance_id
            for row in self.realizations
        ):
            raise ValueError("plan rows must share the manifest task identity")
        indexes = [row.expert_realization_key.realization_index for row in self.realizations]
        if tuple(indexes) != self.realization_slots:
            raise ValueError("plan rows do not match the declared realization universe")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "format_id": self.format_id,
            "task_instance_id": self.task_instance_id.to_mapping(),
            "task_instance_manifest": self.task_instance_manifest.to_mapping(),
            "structured_expert_config_sha256": self.structured_expert_config_sha256,
            "curobo_planner_config_sha256": self.curobo_planner_config_sha256,
            "realization_universe_sha256": self.realization_universe_sha256,
            "realization_slots": list(self.realization_slots),
            "implementation": self.implementation.to_mapping(),
            "realizations": [row.to_mapping() for row in self.realizations],
        }

    @classmethod
    def from_mapping(cls, mapping: Any) -> FrozenPlanSetManifest:
        raw = _strict(mapping, {item.name for item in fields(cls)}, name=cls.__name__).copy()
        _sha(raw["structured_expert_config_sha256"], name="structured_expert_config_sha256")
        raw["task_instance_id"] = TaskInstanceId.from_mapping(raw["task_instance_id"])
        raw["task_instance_manifest"] = ArtifactRef.from_mapping(raw["task_instance_manifest"])
        raw["implementation"] = ImplementationIdentity.from_mapping(raw["implementation"])
        if type(raw["realizations"]) is not list:
            raise TypeError("realizations must be a JSON list")
        if type(raw["realization_slots"]) is not list:
            raise TypeError("realization_slots must be a JSON list")
        raw["realization_slots"] = tuple(raw["realization_slots"])
        raw["realizations"] = tuple(
            FrozenPlanRow.from_mapping(item) for item in raw["realizations"]
        )
        return cls(**raw)


class AttemptStatusClass(str, Enum):
    SUCCESS = "success"
    PLANNER_FAILURE = FailureClass.PLANNER_FAILURE.value
    TASK_FAILURE = FailureClass.TASK_FAILURE.value
    SAFETY_FAILURE = FailureClass.SAFETY_FAILURE.value
    DIVERSITY_REJECTION = FailureClass.DIVERSITY_REJECTION.value
    INFRASTRUCTURE_FAILURE = FailureClass.INFRASTRUCTURE_FAILURE.value


@dataclass(frozen=True)
class AttemptManifest:
    schema_version: int
    format_id: str
    attempt_id: AttemptId
    task_instance_manifest: ArtifactRef
    frozen_plan_set_manifest: ArtifactRef
    status_class: AttemptStatusClass | str
    status: ArtifactRef
    episode_manifest: ArtifactRef | None
    qualification: ArtifactRef | None
    implementation: ImplementationIdentity
    pilot_config_sha256: str
    pilot_gate_config_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.format_id != ATTEMPT_FORMAT
        ):
            raise ValueError("unsupported attempt manifest format")
        if not isinstance(self.attempt_id, AttemptId):
            raise TypeError("attempt_id must be an AttemptId")
        for name in ("task_instance_manifest", "frozen_plan_set_manifest", "status"):
            if not isinstance(getattr(self, name), ArtifactRef):
                raise TypeError(f"{name} must be an ArtifactRef")
        task = self.attempt_id.expert_realization_id.expert_realization_key.task_instance_id
        if self.task_instance_manifest.path != _task_manifest_path(task):
            raise ValueError("task-instance manifest path does not use the locked layout")
        if self.frozen_plan_set_manifest.path != _plan_manifest_path(task):
            raise ValueError("frozen plan-set manifest path does not use the locked layout")
        if self.status.path != "status.json":
            raise ValueError("status artifact path must be status.json")
        if self.episode_manifest is not None and self.episode_manifest.path != (
            "synchronized_episode/manifest.json"
        ):
            raise ValueError("episode manifest path does not use the locked layout")
        if self.qualification is not None and self.qualification.path != "qualification.json":
            raise ValueError("qualification artifact path must be qualification.json")
        if not isinstance(self.implementation, ImplementationIdentity):
            raise TypeError("implementation must be an execution-stage ImplementationIdentity")
        _sha(self.pilot_config_sha256, name="pilot_config_sha256")
        _sha(self.pilot_gate_config_sha256, name="pilot_gate_config_sha256")
        try:
            status_class = AttemptStatusClass(self.status_class)
        except ValueError as error:
            raise ValueError("unknown attempt status class") from error
        object.__setattr__(self, "status_class", status_class)
        if (
            self.attempt_id.expert_realization_id.task_instance_plan_set_sha256
            != self.frozen_plan_set_manifest.sha256
        ):
            raise ValueError("attempt realization plan hash does not match frozen plan reference")
        paired = self.episode_manifest is not None and self.qualification is not None
        if (self.episode_manifest is None) != (self.qualification is None):
            raise ValueError("episode and qualification must be jointly present or absent")
        if status_class is AttemptStatusClass.PLANNER_FAILURE and paired:
            raise ValueError("planner failure cannot contain an episode")
        if (
            status_class
            in {
                AttemptStatusClass.SUCCESS,
                AttemptStatusClass.TASK_FAILURE,
                AttemptStatusClass.SAFETY_FAILURE,
                AttemptStatusClass.DIVERSITY_REJECTION,
            }
            and not paired
        ):
            raise ValueError(f"{status_class.value} requires episode and qualification")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "format_id": self.format_id,
            "attempt_id": self.attempt_id.to_mapping(),
            "task_instance_manifest": self.task_instance_manifest.to_mapping(),
            "frozen_plan_set_manifest": self.frozen_plan_set_manifest.to_mapping(),
            "status_class": self.status_class.value,
            "status": self.status.to_mapping(),
            "episode_manifest": self.episode_manifest.to_mapping()
            if self.episode_manifest
            else None,
            "qualification": self.qualification.to_mapping() if self.qualification else None,
            "implementation": self.implementation.to_mapping(),
            "pilot_config_sha256": self.pilot_config_sha256,
            "pilot_gate_config_sha256": self.pilot_gate_config_sha256,
        }

    @classmethod
    def from_mapping(cls, mapping: Any, *, structured_expert_config_sha256: str) -> AttemptManifest:
        raw = _strict(mapping, {item.name for item in fields(cls)}, name=cls.__name__).copy()
        raw["attempt_id"] = AttemptId.from_mapping(
            raw["attempt_id"], structured_expert_config_sha256=structured_expert_config_sha256
        )
        for name in ("task_instance_manifest", "frozen_plan_set_manifest", "status"):
            raw[name] = ArtifactRef.from_mapping(raw[name])
        for name in ("episode_manifest", "qualification"):
            if raw[name] is not None:
                raw[name] = ArtifactRef.from_mapping(raw[name])
        raw["implementation"] = ImplementationIdentity.from_mapping(raw["implementation"])
        return cls(**raw)


@dataclass(frozen=True)
class StructuredPilotManifest:
    schema_version: int
    format_id: str
    pilot_id: str
    pilot_config_sha256: str
    pilot_gate_config_sha256: str
    structured_expert_config_sha256: str
    curobo_planner_config_sha256: str
    implementation: ImplementationIdentity
    task_instances: tuple[ArtifactRef, ...]
    frozen_plan_sets: tuple[ArtifactRef, ...]
    attempts: tuple[ArtifactRef, ...]
    bounded_review_only: bool
    training_eligibility: dict[str, bool]

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.format_id != PILOT_FORMAT
        ):
            raise ValueError("unsupported structured pilot manifest format")
        _text(self.pilot_id, name="pilot_id")
        for name in (
            "pilot_config_sha256",
            "pilot_gate_config_sha256",
            "structured_expert_config_sha256",
            "curobo_planner_config_sha256",
        ):
            _sha(getattr(self, name), name=name)
        if not isinstance(self.implementation, ImplementationIdentity):
            raise TypeError("implementation must be an ImplementationIdentity")
        for name in ("task_instances", "frozen_plan_sets", "attempts"):
            value = getattr(self, name)
            if type(value) is not tuple or any(not isinstance(item, ArtifactRef) for item in value):
                raise TypeError(f"{name} must be a tuple of ArtifactRef values")
        for name, pattern in (
            ("task_instances", _TASK_REF),
            ("frozen_plan_sets", _PLAN_REF),
            ("attempts", _ATTEMPT_REF),
        ):
            keys = [
                _ordered_inventory_key(item.path, pattern, name=name)
                for item in getattr(self, name)
            ]
            if keys != sorted(keys) or len(keys) != len(set(keys)):
                raise ValueError(f"{name} must preserve canonical parent order without duplicates")
        if self.bounded_review_only is not True:
            raise ValueError("structured pilot inventory must be bounded review only")
        if (
            type(self.training_eligibility) is not dict
            or set(self.training_eligibility) != _TRAINING_FLAGS
            or any(value is not False for value in self.training_eligibility.values())
        ):
            raise ValueError("all five training eligibility flags must be false")
        object.__setattr__(
            self, "training_eligibility", MappingProxyType(dict(self.training_eligibility))
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "format_id": self.format_id,
            "pilot_id": self.pilot_id,
            "pilot_config_sha256": self.pilot_config_sha256,
            "pilot_gate_config_sha256": self.pilot_gate_config_sha256,
            "structured_expert_config_sha256": self.structured_expert_config_sha256,
            "curobo_planner_config_sha256": self.curobo_planner_config_sha256,
            "implementation": self.implementation.to_mapping(),
            "task_instances": [item.to_mapping() for item in self.task_instances],
            "frozen_plan_sets": [item.to_mapping() for item in self.frozen_plan_sets],
            "attempts": [item.to_mapping() for item in self.attempts],
            "bounded_review_only": self.bounded_review_only,
            "training_eligibility": dict(self.training_eligibility),
        }

    @classmethod
    def from_mapping(cls, mapping: Any) -> StructuredPilotManifest:
        raw = _strict(mapping, {item.name for item in fields(cls)}, name=cls.__name__).copy()
        raw["implementation"] = ImplementationIdentity.from_mapping(raw["implementation"])
        for name in ("task_instances", "frozen_plan_sets", "attempts"):
            if type(raw[name]) is not list:
                raise TypeError(f"{name} must be a JSON list")
            raw[name] = tuple(ArtifactRef.from_mapping(item) for item in raw[name])
        return cls(**raw)


def _write_file_fsynced(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace publication requires Linux renameat2")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1)
    if result == 0:
        return
    code = ctypes.get_errno()
    if code == errno.EEXIST:
        raise FileExistsError(code, os.strerror(code), target)
    if code in {errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP}:
        raise RuntimeError("filesystem does not support atomic no-replace directory publication")
    raise OSError(code, os.strerror(code), target)


def _publish_tree(
    target: Path, files: dict[str, bytes], *, expected_hashes: dict[str, str]
) -> Path:
    target = Path(target)
    _reject_hidden_ancestry(target, allow_missing=True)
    if target.exists():
        raise FileExistsError(f"publication target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _reject_hidden_ancestry(target.parent, allow_missing=False)
    parent_stat = os.lstat(target.parent)
    if (
        not stat.S_ISDIR(parent_stat.st_mode)
        or stat.S_ISLNK(parent_stat.st_mode)
        or parent_stat.st_uid != os.geteuid()
        or parent_stat.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    ):
        raise ValueError(
            "publication requires an owner-only writable, current-UID regular parent directory"
        )
    staging = target.parent / f".{target.name}.building-{os.getpid()}-{uuid.uuid4().hex}"
    staging.mkdir()
    staging_stat = os.lstat(staging)
    try:
        if "manifest.json" not in files or "manifest.json" in expected_hashes:
            raise ValueError("publication requires one un-hashed manifest written last")
        child_files = {name: payload for name, payload in files.items() if name != "manifest.json"}
        if set(child_files) != set(expected_hashes):
            raise ValueError("every published child must have one expected raw-byte hash")
        for relative, payload in child_files.items():
            _safe_relative(relative)
            if type(payload) is not bytes:
                raise TypeError("artifact payloads must be bytes")
            _write_file_fsynced(staging / relative, payload)
        for relative, expected in expected_hashes.items():
            if _hash_file(staging / relative) != expected:
                raise ValueError(f"staged artifact hash mismatch: {relative}")
        _write_file_fsynced(staging / "manifest.json", files["manifest.json"])
        directories = {staging}
        for relative in files:
            parent = (staging / relative).parent
            while parent != staging:
                directories.add(parent)
                parent = parent.parent
        for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
            _fsync_directory(directory)
        _rename_noreplace(staging, target)
        _fsync_directory(target.parent)
    except BaseException:
        _cleanup_owned_staging(
            staging,
            expected_device=staging_stat.st_dev,
            expected_inode=staging_stat.st_ino,
            parent=target.parent,
            expected_parent_device=parent_stat.st_dev,
            expected_parent_inode=parent_stat.st_ino,
        )
        raise
    return target / "manifest.json"


def _cleanup_owned_staging(
    staging: Path,
    *,
    expected_device: int,
    expected_inode: int,
    parent: Path,
    expected_parent_device: int,
    expected_parent_inode: int,
) -> None:
    try:
        current_parent = os.lstat(parent)
        current = os.lstat(staging)
    except FileNotFoundError:
        return
    if (
        not stat.S_ISDIR(current_parent.st_mode)
        or stat.S_ISLNK(current_parent.st_mode)
        or (current_parent.st_dev, current_parent.st_ino)
        != (expected_parent_device, expected_parent_inode)
        or not stat.S_ISDIR(current.st_mode)
        or stat.S_ISLNK(current.st_mode)
        or (current.st_dev, current.st_ino) != (expected_device, expected_inode)
    ):
        return
    shutil.rmtree(staging)


def _verify_raw(path: Path, expected_sha: str, *, name: str) -> bytes:
    payload = _read_regular_file_nofollow(path, name=name)
    if _hash_bytes(payload) != expected_sha:
        raise ValueError(f"{name} hash mismatch")
    _reject_legacy_payload(path.as_posix(), payload)
    return payload


def _read_regular_file_nofollow(path: Path, *, name: str) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise ValueError(f"missing or unsafe {name}: {path}") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"{name} must be a regular non-symlink file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _check_final_root(root: Path) -> Path:
    root = Path(root)
    _reject_hidden_ancestry(root, allow_missing=False)
    if not root.is_dir():
        raise ValueError("artifact root must be a final directory")
    return root


def _reject_hidden_components(path: Path) -> None:
    if any(".building-" in component for component in path.parts):
        raise ValueError("hidden building directories are not final artifacts")


def _assert_no_symlink_ancestry(path: Path, *, allow_missing: bool) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            value = os.lstat(current)
        except FileNotFoundError as error:
            if allow_missing:
                return absolute
            raise ValueError(f"artifact path component is missing: {current}") from error
        if stat.S_ISLNK(value.st_mode):
            raise ValueError(f"artifact paths may not contain symlink components: {current}")
    return absolute


def _reject_hidden_ancestry(path: Path, *, allow_missing: bool) -> None:
    lexical = _assert_no_symlink_ancestry(path, allow_missing=allow_missing)
    _reject_hidden_components(lexical)
    physical = lexical.resolve(strict=not allow_missing)
    _reject_hidden_components(physical)


def load_verified_task_instance(root: Path) -> VerifiedTaskInstanceBundle:
    root = _check_final_root(root)
    payload = _read_regular_file_nofollow(root / "manifest.json", name="task manifest")
    manifest = TaskInstanceManifest.from_mapping(
        _load_json_bytes(payload, name="task-instance manifest")
    )
    _require_target_suffix(root, f"{_task_base(manifest.task_instance_id)}/task_instance")
    files: dict[str, bytes] = {}
    for relative, digest in manifest.artifacts.items():
        files[relative] = _verify_raw(_resolve(root, relative), digest, name=relative)
    return VerifiedTaskInstanceBundle(
        manifest=manifest,
        manifest_bytes=payload,
        manifest_sha256=_hash_bytes(payload),
        files=MappingProxyType(files),
    )


def _load_task_instance(root: Path) -> tuple[TaskInstanceManifest, str]:
    bundle = load_verified_task_instance(root)
    return bundle.manifest, bundle.manifest_sha256


def publish_task_instance(
    manifest: TaskInstanceManifest,
    target: Path,
    *,
    artifacts: dict[str, bytes],
) -> Path:
    if not isinstance(manifest, TaskInstanceManifest):
        raise TypeError("manifest must be a TaskInstanceManifest")
    _require_target_suffix(Path(target), f"{_task_base(manifest.task_instance_id)}/task_instance")
    if set(artifacts) != set(manifest.artifacts):
        raise ValueError("task-instance payload names do not match manifest")
    for name, payload in artifacts.items():
        _reject_legacy_payload(name, payload)
        if _hash_bytes(payload) != manifest.artifacts[name]:
            raise ValueError(f"task-instance payload hash mismatch: {name}")
    return _publish_tree(
        Path(target),
        {**artifacts, "manifest.json": _json_bytes(manifest.to_mapping())},
        expected_hashes=dict(manifest.artifacts),
    )


def publish_frozen_plan_set(
    manifest: FrozenPlanSetManifest,
    target: Path,
    *,
    artifacts: dict[str, bytes],
    artifact_root: Path,
) -> Path:
    if not isinstance(manifest, FrozenPlanSetManifest):
        raise TypeError("manifest must be a FrozenPlanSetManifest")
    expected_target = Path(artifact_root) / _task_base(manifest.task_instance_id) / "plan_set"
    if Path(target).absolute() != expected_target.absolute():
        raise ValueError("publication target does not match the canonical plan-set path")
    task_path = _resolve(Path(artifact_root), manifest.task_instance_manifest.path)
    task_manifest, task_sha = _load_task_instance(task_path.parent)
    if task_sha != manifest.task_instance_manifest.sha256:
        raise ValueError("task-instance ancestor hash mismatch")
    if task_manifest.task_instance_id != manifest.task_instance_id:
        raise ValueError("plan-set task identity does not match task-instance ancestor")
    expected: dict[str, str] = {}
    for row in manifest.realizations:
        expected[row.strategy.path] = row.strategy.sha256
        expected[row.planner_candidates.path] = row.planner_candidates.sha256
        if row.selected_reference:
            expected[row.selected_reference.path] = row.selected_reference.sha256
    if set(artifacts) != set(expected):
        raise ValueError("plan-set payload names do not match manifest rows")
    for name, payload in artifacts.items():
        _reject_legacy_payload(name, payload)
        if _hash_bytes(payload) != expected[name]:
            raise ValueError(f"plan-set payload hash mismatch: {name}")
    return _publish_tree(
        Path(target),
        {**artifacts, "manifest.json": _json_bytes(manifest.to_mapping())},
        expected_hashes=expected,
    )


def load_verified_frozen_plan_bundle(
    root: Path, *, artifact_root: Path
) -> VerifiedFrozenPlanBundle:
    root = _check_final_root(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("frozen plan-set manifest is missing")
    raw = _read_regular_file_nofollow(manifest_path, name="frozen plan-set manifest")
    manifest = FrozenPlanSetManifest.from_mapping(
        _load_json_bytes(raw, name="frozen plan-set manifest")
    )
    expected_root = Path(artifact_root) / _task_base(manifest.task_instance_id) / "plan_set"
    if root.absolute() != expected_root.absolute():
        raise ValueError("frozen plan root does not match the canonical hierarchy")
    task_path = _resolve(Path(artifact_root), manifest.task_instance_manifest.path)
    task = load_verified_task_instance(task_path.parent)
    if task.manifest_sha256 != manifest.task_instance_manifest.sha256:
        raise ValueError("task-instance ancestor hash mismatch")
    if task.manifest.task_instance_id != manifest.task_instance_id:
        raise ValueError("plan-set task identity does not match task-instance ancestor")
    files: dict[str, bytes] = {}
    for row in manifest.realizations:
        for reference in (row.strategy, row.planner_candidates, row.selected_reference):
            if reference is not None:
                files[reference.path] = _verify_raw(
                    _resolve(root, reference.path), reference.sha256, name=reference.path
                )
    return VerifiedFrozenPlanBundle(
        manifest=manifest,
        manifest_bytes=raw,
        manifest_sha256=_hash_bytes(raw),
        files=MappingProxyType(files),
        task=task,
    )


def load_verified_frozen_plan(
    root: Path, *, artifact_root: Path
) -> tuple[FrozenPlanSetManifest, str]:
    bundle = load_verified_frozen_plan_bundle(root, artifact_root=artifact_root)
    return bundle.manifest, bundle.manifest_sha256


def _validate_attempt_ancestors(
    manifest: AttemptManifest,
    *,
    artifact_root: Path,
    plan_bundle: VerifiedFrozenPlanBundle | None = None,
) -> tuple[VerifiedFrozenPlanBundle, FrozenPlanRow]:
    plan_path = _resolve(artifact_root, manifest.frozen_plan_set_manifest.path)
    bundle = plan_bundle or load_verified_frozen_plan_bundle(
        plan_path.parent, artifact_root=artifact_root
    )
    plan = bundle.manifest
    if bundle.manifest_sha256 != manifest.frozen_plan_set_manifest.sha256:
        raise ValueError("frozen plan-set ancestor hash mismatch")
    if manifest.task_instance_manifest != plan.task_instance_manifest:
        raise ValueError("attempt task-instance reference does not match frozen plan set")
    if bundle.task.manifest_sha256 != manifest.task_instance_manifest.sha256:
        raise ValueError("task-instance ancestor hash mismatch")
    key = manifest.attempt_id.expert_realization_id.expert_realization_key
    if key.task_instance_id != plan.task_instance_id:
        raise ValueError("attempt task identity does not match ancestors")
    matching = [row for row in plan.realizations if row.expert_realization_key == key]
    if len(matching) != 1:
        raise ValueError("attempt realization key is absent from frozen plan set")
    if manifest.status_class is AttemptStatusClass.PLANNER_FAILURE:
        if matching[0].selection_status != "planner_failure":
            raise ValueError("planner-failure attempt does not match plan row")
    elif matching[0].selection_status != "selected":
        raise ValueError("rollout attempt requires a selected plan row")
    return bundle, matching[0]


def _validate_episode_provenance(
    *,
    manifest: AttemptManifest,
    metadata: Any,
    plan: FrozenPlanSetManifest,
    task: TaskInstanceManifest,
    row: FrozenPlanRow,
) -> None:
    selected_reference_sha = row.selected_reference.sha256 if row.selected_reference else None
    expected = {
        "task_instance_id": task.task_instance_id,
        "expert_realization_id": manifest.attempt_id.expert_realization_id,
        "attempt_id": manifest.attempt_id,
        "task_id": task.task_id,
        "instruction": task.instruction,
        "task_instance_manifest_sha256": manifest.task_instance_manifest.sha256,
        "frozen_plan_set_manifest_sha256": manifest.frozen_plan_set_manifest.sha256,
        "realization_universe_sha256": plan.realization_universe_sha256,
        "task_config_sha256": task.task_config_sha256,
        "motion_config_sha256": task.motion_config_sha256,
        "runtime_config_sha256": task.runtime_config_sha256,
        "controller_config_sha256": task.controller_config_sha256,
        "structured_expert_config_sha256": plan.structured_expert_config_sha256,
        "curobo_planner_config_sha256": plan.curobo_planner_config_sha256,
        "strategy_sha256": row.strategy.sha256,
        "planner_candidates_sha256": row.planner_candidates.sha256,
        "selected_reference_sha256": selected_reference_sha,
        "implementation": manifest.implementation,
        "pilot_config_sha256": manifest.pilot_config_sha256,
        "pilot_gate_config_sha256": manifest.pilot_gate_config_sha256,
    }
    mismatches = [name for name, value in expected.items() if getattr(metadata, name) != value]
    if mismatches:
        raise ValueError(f"episode metadata provenance mismatch: {mismatches}")


def publish_attempt(
    manifest: AttemptManifest,
    target: Path,
    *,
    status: bytes,
    episode: Path | None,
    qualification: bytes | None,
    artifact_root: Path,
) -> Path:
    if not isinstance(manifest, AttemptManifest):
        raise TypeError("manifest must be an AttemptManifest")
    key = manifest.attempt_id.expert_realization_id.expert_realization_key
    expected_target = (
        Path(artifact_root)
        / _task_base(key.task_instance_id)
        / "attempts"
        / f"realization-{key.realization_index:03d}"
        / f"attempt-{manifest.attempt_id.attempt_index:03d}"
    )
    if Path(target).absolute() != expected_target.absolute():
        raise ValueError("publication target does not match the canonical attempt path")
    plan_bundle, row = _validate_attempt_ancestors(manifest, artifact_root=Path(artifact_root))
    plan = plan_bundle.manifest
    task = plan_bundle.task.manifest
    if _hash_bytes(status) != manifest.status.sha256:
        raise ValueError("status payload hash mismatch")
    _reject_legacy_payload("status.json", status)
    status_value = _load_json_bytes(status, name="status.json")
    if status_value.get("status_class") != manifest.status_class.value:
        raise ValueError("status payload class does not match attempt manifest")
    files = {manifest.status.path: status}
    expected_hashes = {manifest.status.path: manifest.status.sha256}
    if manifest.episode_manifest is not None:
        if episode is None or qualification is None:
            raise ValueError("episode and qualification sources are required")
        from latency_meta_mdp.expert_realization.recording_artifacts import (
            VerifiedStructuredEpisodeSnapshot,
            load_verified_structured_episode_snapshot,
        )

        snapshot = (
            episode
            if isinstance(episode, VerifiedStructuredEpisodeSnapshot)
            else load_verified_structured_episode_snapshot(episode)
        )
        loaded = snapshot.episode
        _validate_episode_provenance(
            manifest=manifest,
            metadata=loaded.metadata,
            plan=plan,
            task=task,
            row=row,
        )
        if snapshot.manifest_sha256 != manifest.episode_manifest.sha256:
            raise ValueError("episode manifest hash mismatch")
        if _hash_bytes(qualification) != manifest.qualification.sha256:
            raise ValueError("qualification hash mismatch")
        _reject_legacy_payload("qualification.json", qualification)
        for relative, payload in snapshot.files.items():
            destination = f"synchronized_episode/{relative}"
            files[destination] = payload
            expected_hashes[destination] = _hash_bytes(payload)
        files[manifest.qualification.path] = qualification
        expected_hashes[manifest.qualification.path] = manifest.qualification.sha256
    elif episode is not None or qualification is not None:
        raise ValueError("attempt without episode references cannot receive episode payloads")
    files["manifest.json"] = _json_bytes(manifest.to_mapping())
    return _publish_tree(Path(target), files, expected_hashes=expected_hashes)


def load_verified_attempt_bundle(root: Path, *, artifact_root: Path) -> VerifiedAttemptBundle:
    root = _check_final_root(root)
    manifest_path = root / "manifest.json"
    manifest_bytes = _read_regular_file_nofollow(manifest_path, name="attempt manifest")
    raw_mapping = _load_json_bytes(manifest_bytes, name="attempt manifest")
    config_sha = None
    plan_bundle = None
    plan_ref = raw_mapping.get("frozen_plan_set_manifest")
    if type(plan_ref) is dict and type(plan_ref.get("path")) is str:
        plan_path = _resolve(Path(artifact_root), plan_ref["path"])
        plan_bundle = load_verified_frozen_plan_bundle(
            plan_path.parent, artifact_root=Path(artifact_root)
        )
        config_sha = plan_bundle.manifest.structured_expert_config_sha256
    _sha(config_sha, name="structured_expert_config_sha256")
    manifest = AttemptManifest.from_mapping(raw_mapping, structured_expert_config_sha256=config_sha)
    key = manifest.attempt_id.expert_realization_id.expert_realization_key
    expected_root = (
        Path(artifact_root)
        / _task_base(key.task_instance_id)
        / "attempts"
        / f"realization-{key.realization_index:03d}"
        / f"attempt-{manifest.attempt_id.attempt_index:03d}"
    )
    if root.absolute() != expected_root.absolute():
        raise ValueError("attempt root does not match the canonical hierarchy")
    plan_bundle, row = _validate_attempt_ancestors(
        manifest,
        artifact_root=Path(artifact_root),
        plan_bundle=plan_bundle,
    )
    plan = plan_bundle.manifest
    task = plan_bundle.task.manifest
    status_payload = _verify_raw(
        _resolve(root, manifest.status.path), manifest.status.sha256, name="status"
    )
    status_value = _load_json_bytes(status_payload, name="status")
    if status_value.get("status_class") != manifest.status_class.value:
        raise ValueError("status class mismatch")
    qualification_bytes = None
    episode_snapshot = None
    if manifest.episode_manifest:
        episode_manifest_path = _resolve(root, manifest.episode_manifest.path)
        qualification_bytes = _verify_raw(
            _resolve(root, manifest.qualification.path),
            manifest.qualification.sha256,
            name="qualification",
        )
        from latency_meta_mdp.expert_realization.recording_artifacts import (
            load_verified_structured_episode_snapshot,
        )

        episode_snapshot = load_verified_structured_episode_snapshot(episode_manifest_path.parent)
        if episode_snapshot.manifest_sha256 != manifest.episode_manifest.sha256:
            raise ValueError("episode manifest hash mismatch")
        episode = episode_snapshot.episode
        _validate_episode_provenance(
            manifest=manifest,
            metadata=episode.metadata,
            plan=plan,
            task=task,
            row=row,
        )
    return VerifiedAttemptBundle(
        manifest=manifest,
        manifest_bytes=manifest_bytes,
        status_bytes=status_payload,
        qualification_bytes=qualification_bytes,
        episode=episode_snapshot,
        plan=plan_bundle,
    )


def load_verified_attempt(root: Path, *, artifact_root: Path) -> AttemptManifest:
    return load_verified_attempt_bundle(root, artifact_root=artifact_root).manifest
