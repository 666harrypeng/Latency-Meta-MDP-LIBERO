"""Strict non-training contracts for matched-K6 feedback timing calibration."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

import numpy as np
import yaml

from latency_meta_mdp.expert_realization.recording_contracts import ImplementationIdentity

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_QUANTILE_METRICS = (
    "pregrasp_tick",
    "approach_tick",
    "close_tick",
    "lift_start_tick",
    "first_contact_time_us",
    "stable_contact_time_us",
    "handoff_time_us",
    "lift_threshold_time_us",
    "terminal_time_us",
    "close_distance_m",
    "close_relative_speed_mps",
    "handoff_relative_speed_mps",
    "saturation_fraction",
    "maximum_eef_speed_mps",
    "maximum_eef_acceleration_mps2",
    "maximum_eef_jerk_mps3",
)
_CALIBRATION_SOURCE_PATHS = (
    "pyproject.toml",
    "uv.lock",
    "src/latency_meta_mdp/artifacts.py",
    "src/latency_meta_mdp/backend.py",
    "src/latency_meta_mdp/control.py",
    "src/latency_meta_mdp/expert.py",
    "src/latency_meta_mdp/handoff.py",
    "src/latency_meta_mdp/motion.py",
    "src/latency_meta_mdp/outcomes.py",
    "src/latency_meta_mdp/snapshots.py",
    "src/latency_meta_mdp/task.py",
    "src/latency_meta_mdp/timing.py",
    "src/latency_meta_mdp/cli/calibrate_structured_expert_timing.py",
    "src/latency_meta_mdp/expert_realization/artifacts.py",
    "src/latency_meta_mdp/expert_realization/calibration.py",
    "src/latency_meta_mdp/expert_realization/calibration_worker.py",
    "src/latency_meta_mdp/expert_realization/config.py",
    "src/latency_meta_mdp/expert_realization/contracts.py",
    "src/latency_meta_mdp/expert_realization/recording_contracts.py",
    "src/latency_meta_mdp/expert_realization/shared_prefix.py",
    "src/latency_meta_mdp/expert_realization/task_instance.py",
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


def _seed(value: dict[str, Any]) -> int:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def collect_scoped_implementation_identity(
    project_root: Path,
    *,
    source_paths: tuple[str, ...],
) -> ImplementationIdentity:
    root = Path(project_root).resolve()
    if type(source_paths) is not tuple or not source_paths:
        raise ValueError("source_paths must be a non-empty tuple")
    normalized = []
    for value in source_paths:
        if type(value) is not str:
            raise TypeError("source path must be a string")
        pure = PurePosixPath(value)
        if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != value:
            raise ValueError("source paths must be normalized relative POSIX paths")
        path = root / value
        if not path.is_file():
            raise FileNotFoundError(f"scoped implementation source is missing: {value}")
        normalized.append(value)
    if len(set(normalized)) != len(normalized):
        raise ValueError("source_paths contain duplicates")
    ordered = tuple(sorted(normalized))
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    def diff_is_dirty(*arguments: str) -> bool:
        completed = subprocess.run(
            ["git", *arguments, "--", *ordered],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode not in (0, 1):
            raise RuntimeError(f"scoped git status failed: {completed.stderr.strip()}")
        return completed.returncode == 1

    unstaged = diff_is_dirty("diff", "--quiet", "HEAD")
    staged = diff_is_dirty("diff", "--cached", "--quiet", "HEAD")
    untracked = bool(
        subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "--", *ordered],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    digest = hashlib.sha256()
    for relative in ordered:
        payload = (root / relative).read_bytes()
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return ImplementationIdentity(
        revision=revision,
        source_sha256=digest.hexdigest(),
        dirty=bool(unstaged or staged or untracked),
    )


@dataclass(frozen=True)
class TimingCalibrationConfig:
    schema_version: int
    calibration_id: str
    logical_task_index_start: int
    task_instance_count: int
    levels: tuple[int, ...]
    quantiles: tuple[float, ...]
    bounded_review_only: bool
    training_authorized: bool

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("calibration schema_version must equal 1")
        if self.calibration_id != "panda-ball-feedback-calibration-v1":
            raise ValueError("unsupported calibration_id")
        if type(self.logical_task_index_start) is not int or self.logical_task_index_start != 0:
            raise ValueError("calibration logical_task_index_start must equal 0")
        if type(self.task_instance_count) is not int or self.task_instance_count != 50:
            raise ValueError("calibration task_instance_count must equal 50")
        if self.levels != (1, 2, 3):
            raise ValueError("calibration levels must equal L1/L2/L3")
        if self.quantiles != (0.10, 0.50, 0.90):
            raise ValueError("calibration quantiles must equal 0.10/0.50/0.90")
        if self.bounded_review_only is not True or self.training_authorized is not False:
            raise ValueError("calibration must remain bounded review-only and non-training")

    @property
    def logical_task_indices(self) -> tuple[int, ...]:
        return tuple(
            range(
                self.logical_task_index_start,
                self.logical_task_index_start + self.task_instance_count,
            )
        )

    @property
    def attempt_count(self) -> int:
        return self.task_instance_count * len(self.levels)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "calibration_id": self.calibration_id,
            "logical_task_index_start": self.logical_task_index_start,
            "task_instance_count": self.task_instance_count,
            "levels": list(self.levels),
            "quantiles": list(self.quantiles),
            "bounded_review_only": self.bounded_review_only,
            "training_authorized": self.training_authorized,
        }

    @classmethod
    def from_mapping(cls, mapping: Any) -> TimingCalibrationConfig:
        raw = _strict(mapping, {item.name for item in fields(cls)}, name=cls.__name__).copy()
        if type(raw["levels"]) is not list or type(raw["quantiles"]) is not list:
            raise TypeError("calibration levels/quantiles must be JSON lists")
        raw["levels"] = tuple(raw["levels"])
        raw["quantiles"] = tuple(raw["quantiles"])
        return cls(**raw)


def load_timing_calibration_config(path: Path) -> TimingCalibrationConfig:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return TimingCalibrationConfig.from_mapping(value)


@dataclass(frozen=True)
class TimingCalibrationAttemptRequest:
    calibration_id: str
    logical_task_index: int
    level: int
    master_task_seed: int
    task_config_sha256: str
    motion_config_sha256: str
    runtime_config_sha256: str
    controller_config_sha256: str
    expert_config_sha256: str

    def __post_init__(self) -> None:
        if self.calibration_id != "panda-ball-feedback-calibration-v1":
            raise ValueError("unsupported calibration_id")
        if type(self.logical_task_index) is not int or self.logical_task_index < 0:
            raise ValueError("logical_task_index must be non-negative")
        if type(self.level) is not int or self.level not in (1, 2, 3):
            raise ValueError("calibration level must be L1/L2/L3")
        expected_seed = _seed(
            {
                "corpus_id": self.calibration_id,
                "logical_task_index": self.logical_task_index,
            }
        )
        if type(self.master_task_seed) is not int or self.master_task_seed != expected_seed:
            raise ValueError("master_task_seed does not match calibration identity")
        for item in fields(self):
            if item.name.endswith("_sha256"):
                _sha(getattr(self, item.name), name=item.name)

    def to_mapping(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @classmethod
    def from_mapping(cls, mapping: Any) -> TimingCalibrationAttemptRequest:
        return cls(**_strict(mapping, {item.name for item in fields(cls)}, name=cls.__name__))


@dataclass(frozen=True)
class TimingCalibrationRow:
    logical_task_index: int
    level: int
    terminal_status: str
    terminal_reason: str
    pregrasp_tick: int | None
    approach_tick: int | None
    close_tick: int | None
    lift_start_tick: int | None
    first_contact_time_us: int | None
    stable_contact_time_us: int | None
    handoff_time_us: int | None
    lift_threshold_time_us: int | None
    terminal_time_us: int | None
    close_distance_m: float | None
    close_relative_speed_mps: float | None
    handoff_relative_speed_mps: float | None
    saturation_fraction: float
    maximum_eef_speed_mps: float
    maximum_eef_acceleration_mps2: float
    maximum_eef_jerk_mps3: float
    k6_planning_start_sha256: str

    def __post_init__(self) -> None:
        if type(self.logical_task_index) is not int or self.logical_task_index < 0:
            raise ValueError("logical_task_index must be non-negative")
        if type(self.level) is not int or self.level not in (1, 2, 3):
            raise ValueError("calibration level must be L1/L2/L3")
        if self.terminal_status not in {"success", "failure"}:
            raise ValueError("terminal_status must be success or failure")
        if type(self.terminal_reason) is not str or not self.terminal_reason:
            raise ValueError("terminal_reason must be non-empty")
        optional_names = (
            "pregrasp_tick",
            "approach_tick",
            "close_tick",
            "lift_start_tick",
            "first_contact_time_us",
            "stable_contact_time_us",
            "handoff_time_us",
            "lift_threshold_time_us",
            "terminal_time_us",
            "close_distance_m",
            "close_relative_speed_mps",
            "handoff_relative_speed_mps",
        )
        optional_values = tuple(getattr(self, name) for name in optional_names)
        if self.terminal_status == "success" and any(value is None for value in optional_values):
            raise ValueError("success metrics must all be present")
        phase_names = optional_names[:4]
        time_names = optional_names[4:9]
        phase_values = tuple(getattr(self, name) for name in phase_names)
        time_values = tuple(getattr(self, name) for name in time_names)
        present_phases = tuple(value for value in phase_values if value is not None)
        present_times = tuple(value for value in time_values if value is not None)
        if any(type(value) is not int or value < 0 for value in present_phases):
            raise ValueError("phase ticks must be non-negative integers")
        if present_phases != tuple(sorted(present_phases)):
            raise ValueError("phase ticks must be monotonic")
        if any(
            type(value) is not int or value < 0 or value % 2_000
            for value in present_times
        ):
            raise ValueError("physical event times must lie on the 2 ms grid")
        if present_times != tuple(sorted(present_times)):
            raise ValueError("physical event times must be monotonic")
        for name in optional_names[9:]:
            value = getattr(self, name)
            if value is not None and (
                type(value) is not float or not np.isfinite(value) or value < 0.0
            ):
                raise ValueError(f"{name} must be a non-negative finite float")
        for name in (
            "saturation_fraction",
            "maximum_eef_speed_mps",
            "maximum_eef_acceleration_mps2",
            "maximum_eef_jerk_mps3",
        ):
            value = getattr(self, name)
            if type(value) is not float or not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a non-negative finite float")
        if self.saturation_fraction > 1.0:
            raise ValueError("saturation_fraction cannot exceed one")
        _sha(self.k6_planning_start_sha256, name="k6_planning_start_sha256")

    def to_mapping(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @classmethod
    def from_mapping(cls, mapping: Any) -> TimingCalibrationRow:
        return cls(**_strict(mapping, {item.name for item in fields(cls)}, name=cls.__name__))


def _thaw_quantiles(
    value: Mapping[int, Mapping[str, tuple[float, ...]]],
) -> dict[str, dict[str, list[float]]]:
    return {
        str(level): {name: list(values) for name, values in metrics.items()}
        for level, metrics in value.items()
    }


@dataclass(frozen=True)
class TimingCalibrationReport:
    config: TimingCalibrationConfig
    request_sha256: str
    rows: tuple[TimingCalibrationRow, ...]
    per_level_quantiles: Mapping[int, Mapping[str, tuple[float, ...]]]
    success_count: int
    failure_count: int
    per_level_success_count: Mapping[int, int]
    implementation: ImplementationIdentity
    artifact_eligible: bool
    training_authorized: bool

    def __post_init__(self) -> None:
        if not isinstance(self.config, TimingCalibrationConfig):
            raise TypeError("config must be TimingCalibrationConfig")
        if not isinstance(self.implementation, ImplementationIdentity):
            raise TypeError("implementation must be ImplementationIdentity")
        _sha(self.request_sha256, name="request_sha256")
        if type(self.rows) is not tuple or any(
            not isinstance(row, TimingCalibrationRow) for row in self.rows
        ):
            raise TypeError("rows must contain TimingCalibrationRow values")
        expected = tuple(
            (task_index, level)
            for task_index in self.config.logical_task_indices
            for level in self.config.levels
        )
        actual = tuple((row.logical_task_index, row.level) for row in self.rows)
        if actual != expected:
            raise ValueError("calibration row universe is missing, duplicated, or out of order")
        measured_success = sum(row.terminal_status == "success" for row in self.rows)
        if self.success_count != measured_success or (
            self.failure_count != len(self.rows) - measured_success
        ):
            raise ValueError("calibration success/failure counts do not match rows")
        expected_per_level = {
            level: sum(
                row.level == level and row.terminal_status == "success" for row in self.rows
            )
            for level in self.config.levels
        }
        if dict(self.per_level_success_count) != expected_per_level:
            raise ValueError("per-level success counts do not match rows")
        frozen_quantiles: dict[int, Mapping[str, tuple[float, ...]]] = {}
        for level in self.config.levels:
            metrics = self.per_level_quantiles[level]
            if set(metrics) != set(_QUANTILE_METRICS):
                raise ValueError("per-level quantile metric inventory is incomplete")
            frozen_quantiles[level] = MappingProxyType(
                {name: tuple(metrics[name]) for name in _QUANTILE_METRICS}
            )
            if any(
                len(values) != len(self.config.quantiles)
                or any(type(value) is not float or not np.isfinite(value) for value in values)
                for values in frozen_quantiles[level].values()
            ):
                raise ValueError("per-level quantile values are invalid")
        object.__setattr__(self, "per_level_quantiles", MappingProxyType(frozen_quantiles))
        object.__setattr__(
            self,
            "per_level_success_count",
            MappingProxyType(dict(expected_per_level)),
        )
        if type(self.artifact_eligible) is not bool or (
            self.artifact_eligible != (not self.implementation.dirty)
        ):
            raise ValueError("artifact eligibility must reflect implementation cleanliness")
        if self.training_authorized is not False:
            raise ValueError("calibration report must remain non-training")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "config": self.config.to_mapping(),
            "request_sha256": self.request_sha256,
            "rows": [row.to_mapping() for row in self.rows],
            "per_level_quantiles": _thaw_quantiles(self.per_level_quantiles),
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "per_level_success_count": {
                str(level): count for level, count in self.per_level_success_count.items()
            },
            "implementation": self.implementation.to_mapping(),
            "artifact_eligible": self.artifact_eligible,
            "training_authorized": self.training_authorized,
        }

    @classmethod
    def from_mapping(cls, mapping: Any) -> TimingCalibrationReport:
        raw = _strict(mapping, {item.name for item in fields(cls)}, name=cls.__name__).copy()
        raw["config"] = TimingCalibrationConfig.from_mapping(raw["config"])
        if type(raw["rows"]) is not list or type(raw["per_level_quantiles"]) is not dict:
            raise TypeError("calibration rows/quantiles have invalid JSON containers")
        raw["rows"] = tuple(TimingCalibrationRow.from_mapping(row) for row in raw["rows"])
        raw["per_level_quantiles"] = {
            int(level): {name: tuple(values) for name, values in metrics.items()}
            for level, metrics in raw["per_level_quantiles"].items()
        }
        if type(raw["per_level_success_count"]) is not dict:
            raise TypeError("per_level_success_count must be a JSON mapping")
        raw["per_level_success_count"] = {
            int(level): count for level, count in raw["per_level_success_count"].items()
        }
        raw["implementation"] = ImplementationIdentity.from_mapping(raw["implementation"])
        return cls(**raw)


def build_timing_calibration_report(
    *,
    config: TimingCalibrationConfig,
    request_sha256: str,
    rows: tuple[TimingCalibrationRow, ...],
    implementation: ImplementationIdentity,
) -> TimingCalibrationReport:
    if not isinstance(config, TimingCalibrationConfig):
        raise TypeError("config must be TimingCalibrationConfig")
    expected = tuple(
        (task_index, level)
        for task_index in config.logical_task_indices
        for level in config.levels
    )
    actual = tuple((row.logical_task_index, row.level) for row in rows)
    if actual != expected:
        raise ValueError("calibration row universe is missing, duplicated, or out of order")
    per_level_quantiles: dict[int, dict[str, tuple[float, ...]]] = {}
    per_level_success_count = {}
    for level in config.levels:
        successful = [
            row for row in rows if row.level == level and row.terminal_status == "success"
        ]
        if not successful:
            raise ValueError("every level requires at least one successful calibration row")
        per_level_success_count[level] = len(successful)
        per_level_quantiles[level] = {
            name: tuple(
                float(value)
                for value in np.quantile(
                    np.asarray([getattr(row, name) for row in successful], dtype=np.float64),
                    config.quantiles,
                )
            )
            for name in _QUANTILE_METRICS
        }
    success_count = sum(row.terminal_status == "success" for row in rows)
    return TimingCalibrationReport(
        config=config,
        request_sha256=request_sha256,
        rows=rows,
        per_level_quantiles=per_level_quantiles,
        success_count=success_count,
        failure_count=len(rows) - success_count,
        per_level_success_count=per_level_success_count,
        implementation=implementation,
        artifact_eligible=not implementation.dirty,
        training_authorized=False,
    )


def _write_json_mapping(path: Path, mapping: dict[str, Any]) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite calibration protocol file: {target}")
    target.write_text(
        json.dumps(mapping, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_json_mapping(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} must contain one valid JSON mapping") from error
    if type(value) is not dict:
        raise ValueError(f"{name} must contain one valid JSON mapping")
    return value


def write_timing_calibration_attempt_request(
    request: TimingCalibrationAttemptRequest,
    path: Path,
) -> None:
    if not isinstance(request, TimingCalibrationAttemptRequest):
        raise TypeError("request must be TimingCalibrationAttemptRequest")
    _write_json_mapping(path, request.to_mapping())


def write_timing_calibration_attempt_result(
    row: TimingCalibrationRow,
    path: Path,
) -> None:
    if not isinstance(row, TimingCalibrationRow):
        raise TypeError("row must be TimingCalibrationRow")
    _write_json_mapping(path, row.to_mapping())


def load_timing_calibration_attempt_result(path: Path) -> TimingCalibrationRow:
    return TimingCalibrationRow.from_mapping(
        _load_json_mapping(path, name="timing calibration result")
    )


def run_timing_calibration_attempt_process(
    request_path: Path,
    result_path: Path,
    *,
    worker_python: Path,
    timeout_seconds: float = 120.0,
) -> None:
    if type(timeout_seconds) is not float or timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be a positive float")
    completed = subprocess.run(
        [
            str(worker_python),
            "-m",
            "latency_meta_mdp.expert_realization.calibration_worker",
            "--request",
            str(request_path),
            "--result",
            str(result_path),
        ],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        diagnostic = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"timing calibration worker failed: {diagnostic[-4000:]}")


def publish_timing_calibration_report(
    report: TimingCalibrationReport,
    target: Path,
) -> Path:
    from latency_meta_mdp.expert_realization.artifacts import _publish_tree

    if not isinstance(report, TimingCalibrationReport):
        raise TypeError("report must be TimingCalibrationReport")
    report_payload = (
        json.dumps(report.to_mapping(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    report_sha = hashlib.sha256(report_payload).hexdigest()
    manifest = {
        "schema_version": 1,
        "format_id": "structured_expert_timing_calibration_v1",
        "calibration_id": report.config.calibration_id,
        "request_sha256": report.request_sha256,
        "report_sha256": report_sha,
        "row_count": len(report.rows),
        "artifact_eligible": report.artifact_eligible,
        "training_authorized": report.training_authorized,
        "implementation": report.implementation.to_mapping(),
    }
    manifest_payload = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")
    manifest_path = _publish_tree(
        Path(target),
        {"report.json": report_payload, "manifest.json": manifest_payload},
        expected_hashes={"report.json": report_sha},
    )
    return manifest_path


def load_verified_timing_calibration_report(root: Path) -> TimingCalibrationReport:
    from latency_meta_mdp.expert_realization.artifacts import (
        _check_final_root,
        _read_regular_file_nofollow,
    )

    artifact_root = _check_final_root(Path(root))
    if {path.name for path in artifact_root.iterdir()} != {"manifest.json", "report.json"}:
        raise ValueError("timing calibration artifact inventory is invalid")
    manifest = _load_json_mapping(artifact_root / "manifest.json", name="calibration manifest")
    expected_fields = {
        "schema_version",
        "format_id",
        "calibration_id",
        "request_sha256",
        "report_sha256",
        "row_count",
        "artifact_eligible",
        "training_authorized",
        "implementation",
    }
    _strict(manifest, expected_fields, name="calibration manifest")
    if (
        manifest["schema_version"] != 1
        or manifest["format_id"] != "structured_expert_timing_calibration_v1"
        or type(manifest["artifact_eligible"]) is not bool
        or manifest["training_authorized"] is not False
    ):
        raise ValueError("timing calibration manifest semantics are invalid")
    report_payload = _read_regular_file_nofollow(
        artifact_root / "report.json",
        name="timing calibration report",
    )
    if hashlib.sha256(report_payload).hexdigest() != manifest["report_sha256"]:
        raise ValueError("timing calibration report hash mismatch")
    try:
        report_mapping = json.loads(report_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("timing calibration report JSON is invalid") from error
    report = TimingCalibrationReport.from_mapping(report_mapping)
    if (
        manifest["calibration_id"] != report.config.calibration_id
        or manifest["request_sha256"] != report.request_sha256
        or manifest["row_count"] != len(report.rows)
        or manifest["artifact_eligible"] != report.artifact_eligible
        or ImplementationIdentity.from_mapping(manifest["implementation"])
        != report.implementation
    ):
        raise ValueError("timing calibration manifest/report identity mismatch")
    return report


def collect_timing_calibration(
    *,
    project_root: Path,
    config_path: Path,
    target: Path,
    attempt_runner: Callable[[TimingCalibrationAttemptRequest], TimingCalibrationRow],
    on_progress: Callable[[dict[str, int]], None] | None = None,
) -> Path:
    root = Path(project_root).resolve()
    config_source = Path(config_path)
    config = load_timing_calibration_config(config_source)
    common_paths = {
        "task_config_sha256": root / "configs/task/dynamic_grasp_lift_l0.yaml",
        "runtime_config_sha256": root / "configs/runtime/robosuite_v1.yaml",
        "controller_config_sha256": root / "configs/control/panda_osc_pose_delta_v1.yaml",
        "expert_config_sha256": root / "configs/expert/panda_ball_feedback_v1.yaml",
    }
    common_hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in common_paths.items()
    }
    requests = tuple(
        TimingCalibrationAttemptRequest(
            calibration_id=config.calibration_id,
            logical_task_index=task_index,
            level=level,
            master_task_seed=_seed(
                {
                    "corpus_id": config.calibration_id,
                    "logical_task_index": task_index,
                }
            ),
            motion_config_sha256=hashlib.sha256(
                (
                    root / f"configs/motion/dynamic_grasp_lift_l{level}.yaml"
                ).read_bytes()
            ).hexdigest(),
            **common_hashes,
        )
        for task_index in config.logical_task_indices
        for level in config.levels
    )
    request_sha256 = hashlib.sha256(
        json.dumps(
            {
                "config": config.to_mapping(),
                "config_source_sha256": hashlib.sha256(config_source.read_bytes()).hexdigest(),
                "requests": [request.to_mapping() for request in requests],
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    rows = []
    for completed, request in enumerate(requests, start=1):
        row = attempt_runner(request)
        if not isinstance(row, TimingCalibrationRow) or (
            row.logical_task_index != request.logical_task_index or row.level != request.level
        ):
            raise ValueError("calibration runner returned a row for the wrong request")
        rows.append(row)
        if on_progress is not None:
            on_progress({"completed": completed, "total": len(requests)})
    implementation = collect_scoped_implementation_identity(
        root,
        source_paths=_CALIBRATION_SOURCE_PATHS,
    )
    report = build_timing_calibration_report(
        config=config,
        request_sha256=request_sha256,
        rows=tuple(rows),
        implementation=implementation,
    )
    return publish_timing_calibration_report(report, target)
