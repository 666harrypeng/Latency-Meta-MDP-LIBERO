"""Typed CuRobo candidate records and immutable numerical serialization."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tempfile
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from latency_meta_mdp.expert_realization.contracts import (
    ExpertRealizationKey,
    derive_subseed,
)
from latency_meta_mdp.expert_realization.keyposes import InterceptionPlan
from latency_meta_mdp.expert_realization.robot_bridge import PandaPlanningBridge


class PlannerCandidateStatus(str, Enum):
    SUCCESS = "success"
    PLANNER_FAILURE = "planner_failure"
    TIMEOUT = "timeout"


def _seed(payload: dict[str, Any]) -> int:
    value = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def planner_candidate_seed(key: ExpertRealizationKey, candidate_index: int) -> int:
    if not isinstance(key, ExpertRealizationKey):
        raise TypeError("key must be an ExpertRealizationKey")
    if type(candidate_index) is not int or not 0 <= candidate_index < 8:
        raise ValueError("candidate_index must be in 0..7")
    return _seed(
        {
            "planner_subseed": derive_subseed(key.realization_seed, "planner"),
            "candidate_index": candidate_index,
        }
    )


def planner_candidate_seeds(
    key: ExpertRealizationKey, *, candidate_count: int
) -> tuple[int, ...]:
    if candidate_count != 8:
        raise ValueError("the locked planner candidate count is 8")
    return tuple(planner_candidate_seed(key, index) for index in range(candidate_count))


def schedule_candidate_timestamps(
    raw_timestamps_seconds: np.ndarray,
    *,
    arrival_duration_seconds: float,
    time_scaling_profile: str,
) -> np.ndarray:
    """Map one feasible geometric path onto the frozen episode-level arrival schedule."""
    raw = np.asarray(raw_timestamps_seconds)
    if (
        raw.dtype != np.float64
        or raw.ndim != 1
        or len(raw) < 2
        or not np.all(np.isfinite(raw))
        or raw[0] != 0.0
        or np.any(np.diff(raw) <= 0.0)
    ):
        raise ValueError("raw_timestamps_seconds must be increasing finite float64[N] from zero")
    if (
        type(arrival_duration_seconds) is not float
        or not np.isfinite(arrival_duration_seconds)
        or arrival_duration_seconds <= 0.0
    ):
        raise ValueError("arrival_duration_seconds must be a positive finite float")
    if time_scaling_profile not in {
        "nominal",
        "early_smooth",
        "lateral_smooth",
        "minimum_jerk_slow",
    }:
        raise ValueError("unknown time_scaling_profile")

    progress = raw / raw[-1]
    if time_scaling_profile == "nominal":
        scheduled_progress = progress
    else:
        lower = np.zeros_like(progress)
        upper = np.ones_like(progress)
        for _ in range(64):
            midpoint = 0.5 * (lower + upper)
            if time_scaling_profile == "minimum_jerk_slow":
                warped = 10.0 * midpoint**3 - 15.0 * midpoint**4 + 6.0 * midpoint**5
            else:
                warped = 3.0 * midpoint**2 - 2.0 * midpoint**3
            lower = np.where(warped < progress, midpoint, lower)
            upper = np.where(warped < progress, upper, midpoint)
        scheduled_progress = 0.5 * (lower + upper)
        scheduled_progress[0] = 0.0
        scheduled_progress[-1] = 1.0
    scheduled = np.asarray(scheduled_progress * arrival_duration_seconds, dtype=np.float64)
    if np.any(np.diff(scheduled) <= 0.0):
        raise ValueError("time scaling produced non-increasing timestamps")
    return scheduled


def validate_scheduled_joint_path(
    qpos_path: np.ndarray,
    timestamps_seconds: np.ndarray,
    *,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    joint_velocity: np.ndarray,
    joint_acceleration: np.ndarray,
) -> None:
    """Reject retimed planner paths that violate the bound Panda joint contract."""
    qpos = np.asarray(qpos_path)
    timestamps = np.asarray(timestamps_seconds)
    limits = {
        "joint_lower": np.asarray(joint_lower),
        "joint_upper": np.asarray(joint_upper),
        "joint_velocity": np.asarray(joint_velocity),
        "joint_acceleration": np.asarray(joint_acceleration),
    }
    if (
        qpos.dtype != np.float64
        or qpos.ndim != 2
        or qpos.shape[1:] != (7,)
        or len(qpos) < 2
        or not np.all(np.isfinite(qpos))
        or timestamps.dtype != np.float64
        or timestamps.shape != (len(qpos),)
        or not np.all(np.isfinite(timestamps))
        or timestamps[0] != 0.0
        or np.any(np.diff(timestamps) <= 0.0)
    ):
        raise ValueError("scheduled joint path must be aligned finite float64 data")
    if any(
        value.dtype != np.float64
        or value.shape != (7,)
        or not np.all(np.isfinite(value))
        for value in limits.values()
    ):
        raise ValueError("joint limits must be finite float64[7]")
    tolerance = 1.0e-9
    if np.any(qpos < limits["joint_lower"] - tolerance) or np.any(
        qpos > limits["joint_upper"] + tolerance
    ):
        raise ValueError("scheduled path violates joint position limits")
    dt = np.diff(timestamps)
    segment_velocity = np.diff(qpos, axis=0) / dt[:, None]
    if np.any(np.abs(segment_velocity) > limits["joint_velocity"] + tolerance):
        raise ValueError("scheduled path violates joint velocity limits")
    if len(segment_velocity) >= 2:
        midpoint_dt = 0.5 * (dt[:-1] + dt[1:])
        segment_acceleration = np.diff(segment_velocity, axis=0) / midpoint_dt[:, None]
        if np.any(np.abs(segment_acceleration) > limits["joint_acceleration"] + tolerance):
            raise ValueError("scheduled path violates joint acceleration limits")


def _array(value: Any, *, shape_tail: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value)
    if (
        result.dtype != np.float64
        or result.ndim != len(shape_tail) + 1
        or result.shape[1:] != shape_tail
        or not np.all(np.isfinite(result))
    ):
        raise ValueError(f"{name} must be finite float64[N,{','.join(map(str, shape_tail))}]")
    result = np.array(result, copy=True)
    result.setflags(write=False)
    return result


def _candidate_fingerprint(values: dict[str, Any], arrays: tuple[np.ndarray, ...]) -> str:
    digest = hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )
    for array in arrays:
        digest.update(array.dtype.str.encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, eq=False)
class PlannerCandidate:
    expert_realization_key: ExpertRealizationKey
    candidate_index: int
    requested_seed: int
    effective_seed: int
    status: PlannerCandidateStatus
    qpos_path: np.ndarray
    timestamps_seconds: np.ndarray
    eef_positions_world: np.ndarray
    eef_path_length_m: float
    certified_clearance_lower_bound_m: float
    goal_position_error_m: float
    goal_rotation_error_degrees: float
    planner_cost: float
    planning_time_seconds: float
    failure_reason: str | None
    deterministic_replay_verified: bool | None
    trajectory_fingerprint: str = field(init=False)
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.expert_realization_key, ExpertRealizationKey):
            raise TypeError("expert_realization_key must be an ExpertRealizationKey")
        if type(self.candidate_index) is not int or not 0 <= self.candidate_index < 8:
            raise ValueError("candidate_index must be in 0..7")
        expected_seed = planner_candidate_seed(self.expert_realization_key, self.candidate_index)
        for name in ("requested_seed", "effective_seed"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < 2**64:
                raise ValueError(f"{name} must be uint64")
        if self.requested_seed != expected_seed or self.effective_seed != self.requested_seed:
            raise ValueError("planner requested/effective seed contract mismatch")
        try:
            status = PlannerCandidateStatus(self.status)
        except ValueError as error:
            raise ValueError("unknown planner candidate status") from error
        object.__setattr__(self, "status", status)
        qpos = _array(self.qpos_path, shape_tail=(7,), name="qpos_path")
        eef = _array(self.eef_positions_world, shape_tail=(3,), name="eef_positions_world")
        timestamps = np.asarray(self.timestamps_seconds)
        if (
            timestamps.dtype != np.float64
            or timestamps.ndim != 1
            or not np.all(np.isfinite(timestamps))
        ):
            raise ValueError("timestamps_seconds must be finite float64[N]")
        timestamps = np.array(timestamps, copy=True)
        timestamps.setflags(write=False)
        object.__setattr__(self, "qpos_path", qpos)
        object.__setattr__(self, "eef_positions_world", eef)
        object.__setattr__(self, "timestamps_seconds", timestamps)
        metrics = (
            self.eef_path_length_m,
            self.certified_clearance_lower_bound_m,
            self.goal_position_error_m,
            self.goal_rotation_error_degrees,
            self.planner_cost,
            self.planning_time_seconds,
        )
        if any(
            type(value) is not float or not np.isfinite(value) or value < 0
            for value in metrics
        ):
            raise ValueError("planner candidate metrics must be finite non-negative floats")
        if self.deterministic_replay_verified not in (None, True, False):
            raise TypeError("deterministic_replay_verified must be bool or None")
        if status is PlannerCandidateStatus.SUCCESS:
            if len(qpos) < 2 or len(qpos) != len(eef) or len(qpos) != len(timestamps):
                raise ValueError("successful candidate arrays must be aligned and nonempty")
            if timestamps[0] != 0.0 or np.any(np.diff(timestamps) <= 0):
                raise ValueError("candidate timestamps must start at zero and increase")
            if self.failure_reason is not None:
                raise ValueError("successful candidate cannot carry failure_reason")
            measured_length = float(np.linalg.norm(np.diff(eef, axis=0), axis=1).sum())
            if not np.isclose(measured_length, self.eef_path_length_m, atol=1e-9, rtol=0.0):
                raise ValueError("eef_path_length_m does not match numerical path")
        else:
            if len(qpos) or len(eef) or len(timestamps):
                raise ValueError("failed candidate cannot carry numerical path arrays")
            if type(self.failure_reason) is not str or not self.failure_reason:
                raise ValueError("failed candidate requires failure_reason")
        trajectory_values = self.trajectory_mapping()
        trajectory_fingerprint = _candidate_fingerprint(
            trajectory_values,
            (qpos, timestamps, eef),
        )
        object.__setattr__(self, "trajectory_fingerprint", trajectory_fingerprint)
        object.__setattr__(
            self,
            "fingerprint",
            _candidate_fingerprint(self.metadata_mapping(), (qpos, timestamps, eef)),
        )

    def trajectory_mapping(self) -> dict[str, Any]:
        """Numerical identity excluding nondeterministic wall-clock/audit fields."""
        return {
            "expert_realization_key": self.expert_realization_key.to_mapping(),
            "candidate_index": self.candidate_index,
            "requested_seed": self.requested_seed,
            "effective_seed": self.effective_seed,
            "status": self.status.value,
            "eef_path_length_m": self.eef_path_length_m,
            "certified_clearance_lower_bound_m": self.certified_clearance_lower_bound_m,
            "goal_position_error_m": self.goal_position_error_m,
            "goal_rotation_error_degrees": self.goal_rotation_error_degrees,
            "planner_cost": self.planner_cost,
            "failure_reason": self.failure_reason,
        }

    def metadata_mapping(self) -> dict[str, Any]:
        return {
            **self.trajectory_mapping(),
            "planning_time_seconds": self.planning_time_seconds,
            "deterministic_replay_verified": self.deterministic_replay_verified,
            "trajectory_fingerprint": self.trajectory_fingerprint,
        }

    def constructor_mapping(self) -> dict[str, Any]:
        mapping = {
            **self.metadata_mapping(),
            "expert_realization_key": self.expert_realization_key,
            "status": self.status,
            "qpos_path": self.qpos_path,
            "timestamps_seconds": self.timestamps_seconds,
            "eef_positions_world": self.eef_positions_world,
        }
        mapping.pop("trajectory_fingerprint")
        return mapping

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PlannerCandidate):
            return False
        return self.metadata_mapping() == other.metadata_mapping() and all(
            np.array_equal(getattr(self, name), getattr(other, name))
            for name in ("qpos_path", "timestamps_seconds", "eef_positions_world")
        )

    @classmethod
    def failure(
        cls,
        *,
        expert_realization_key: ExpertRealizationKey,
        candidate_index: int,
        requested_seed: int,
        effective_seed: int,
        status: PlannerCandidateStatus,
        reason: str,
        planning_time_seconds: float,
    ) -> PlannerCandidate:
        return cls(
            expert_realization_key=expert_realization_key,
            candidate_index=candidate_index,
            requested_seed=requested_seed,
            effective_seed=effective_seed,
            status=status,
            qpos_path=np.empty((0, 7), dtype=np.float64),
            timestamps_seconds=np.empty((0,), dtype=np.float64),
            eef_positions_world=np.empty((0, 3), dtype=np.float64),
            eef_path_length_m=0.0,
            certified_clearance_lower_bound_m=0.0,
            goal_position_error_m=0.0,
            goal_rotation_error_degrees=0.0,
            planner_cost=0.0,
            planning_time_seconds=float(planning_time_seconds),
            failure_reason=reason,
            deterministic_replay_verified=None,
        )


def _npz_bytes(candidate: PlannerCandidate) -> bytes:
    output = io.BytesIO()
    np.savez(
        output,
        qpos_path=candidate.qpos_path,
        timestamps_seconds=candidate.timestamps_seconds,
        eef_positions_world=candidate.eef_positions_world,
    )
    return output.getvalue()


def write_planner_candidates(root: Path, candidates: tuple[PlannerCandidate, ...]) -> None:
    if type(candidates) is not tuple or len(candidates) != 8:
        raise ValueError("candidate artifact requires exactly eight records")
    ordered = tuple(sorted(candidates, key=lambda item: item.candidate_index))
    if [item.candidate_index for item in ordered] != list(range(8)):
        raise ValueError("candidate indices must cover 0..7")
    if len({item.expert_realization_key for item in ordered}) != 1:
        raise ValueError("candidate records must share one realization key")
    root = Path(root)
    root.mkdir(parents=False, exist_ok=False)
    rows = []
    for candidate in ordered:
        payload = _npz_bytes(candidate)
        name = f"candidate-{candidate.candidate_index:03d}.npz"
        (root / name).write_bytes(payload)
        rows.append(
            {
                **candidate.metadata_mapping(),
                "fingerprint": candidate.fingerprint,
                "array_file": name,
                "array_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    (root / "candidates.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def load_planner_candidates(
    root: Path, *, structured_expert_config_sha256: str
) -> tuple[PlannerCandidate, ...]:
    root = Path(root)
    expected = {"candidates.jsonl"} | {f"candidate-{index:03d}.npz" for index in range(8)}
    if {item.name for item in root.iterdir()} != expected:
        raise ValueError("candidate artifact inventory is invalid")
    rows = [json.loads(line) for line in (root / "candidates.jsonl").read_text().splitlines()]
    if len(rows) != 8:
        raise ValueError("candidate metadata must contain eight rows")
    candidates = []
    for index, row in enumerate(rows):
        if row.get("candidate_index") != index:
            raise ValueError("candidate metadata order is invalid")
        filename = row.pop("array_file")
        expected_hash = row.pop("array_sha256")
        expected_fingerprint = row.pop("fingerprint")
        expected_trajectory_fingerprint = row.pop("trajectory_fingerprint")
        payload = (root / filename).read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_hash:
            raise ValueError("candidate array hash mismatch")
        with np.load(io.BytesIO(payload), allow_pickle=False) as source:
            if set(source.files) != {"qpos_path", "timestamps_seconds", "eef_positions_world"}:
                raise ValueError("candidate NPZ fields are invalid")
            arrays = {name: np.array(source[name], copy=True) for name in source.files}
        key = ExpertRealizationKey.from_mapping(
            row.pop("expert_realization_key")
        )
        status = PlannerCandidateStatus(row.pop("status"))
        candidate = PlannerCandidate(
            expert_realization_key=key,
            status=status,
            qpos_path=arrays["qpos_path"],
            timestamps_seconds=arrays["timestamps_seconds"],
            eef_positions_world=arrays["eef_positions_world"],
            **row,
        )
        if candidate.fingerprint != expected_fingerprint:
            raise ValueError("candidate fingerprint mismatch")
        if candidate.trajectory_fingerprint != expected_trajectory_fingerprint:
            raise ValueError("candidate trajectory fingerprint mismatch")
        candidates.append(candidate)
    return tuple(candidates)


def _plan_mapping(plan: InterceptionPlan) -> dict[str, Any]:
    return {
        "task_instance_id": plan.task_instance_id.to_mapping(),
        "family": plan.family.value,
        "interception_tick": plan.interception_tick,
        "pregrasp_arrival_tick": plan.pregrasp_arrival_tick,
        "reference_start_tick": 5,
        "time_scaling_profile": plan.time_scaling_profile,
        "guide_positions_world": plan.guide_positions_world.tolist(),
        "pregrasp_position_world": plan.pregrasp_position_world.tolist(),
        "fixed_orientation_world": plan.fixed_orientation_world.tolist(),
    }


def write_planner_request(
    *,
    expert_realization_key: ExpertRealizationKey,
    structured_expert_config_sha256: str,
    candidate_index: int,
    bridge: PandaPlanningBridge,
    plan: InterceptionPlan,
    start_qpos: np.ndarray,
    timeout_seconds: float,
    path: Path,
) -> None:
    if plan.task_instance_id != expert_realization_key.task_instance_id:
        raise ValueError("planner request plan/key task identity mismatch")
    seed = planner_candidate_seed(expert_realization_key, candidate_index)
    qpos = np.asarray(start_qpos)
    if qpos.dtype != np.float64 or qpos.shape != (7,) or not np.all(np.isfinite(qpos)):
        raise ValueError("planner request start_qpos must be finite float64[7]")
    if type(timeout_seconds) is not float or timeout_seconds != 5.0:
        raise ValueError("planner request timeout must equal 5.0 seconds")
    payload = {
        "schema_version": 1,
        "format_id": "structured_expert_planner_request_v1",
        "expert_realization_key": expert_realization_key.to_mapping(),
        "structured_expert_config_sha256": structured_expert_config_sha256,
        "candidate_index": candidate_index,
        "requested_seed": seed,
        "bridge": bridge.to_mapping(),
        "plan": _plan_mapping(plan),
        "start_qpos": qpos.tolist(),
        "timeout_seconds": timeout_seconds,
    }
    Path(path).write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def write_single_candidate_result(path: Path, candidate: PlannerCandidate) -> None:
    root = Path(path)
    root.mkdir(parents=False, exist_ok=False)
    array_payload = _npz_bytes(candidate)
    (root / "candidate.npz").write_bytes(array_payload)
    metadata = {
        **candidate.metadata_mapping(),
        "fingerprint": candidate.fingerprint,
        "array_sha256": hashlib.sha256(array_payload).hexdigest(),
    }
    (root / "candidate.json").write_text(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def load_single_candidate_result(
    path: Path, *, structured_expert_config_sha256: str
) -> PlannerCandidate:
    root = Path(path)
    if {item.name for item in root.iterdir()} != {"candidate.json", "candidate.npz"}:
        raise ValueError("single candidate result inventory is invalid")
    metadata = json.loads((root / "candidate.json").read_text())
    array_payload = (root / "candidate.npz").read_bytes()
    if metadata.pop("array_sha256") != hashlib.sha256(array_payload).hexdigest():
        raise ValueError("single candidate array hash mismatch")
    expected_fingerprint = metadata.pop("fingerprint")
    expected_trajectory_fingerprint = metadata.pop("trajectory_fingerprint")
    with np.load(io.BytesIO(array_payload), allow_pickle=False) as source:
        arrays = {name: np.array(source[name], copy=True) for name in source.files}
    key = ExpertRealizationKey.from_mapping(
        metadata.pop("expert_realization_key")
    )
    candidate = PlannerCandidate(
        expert_realization_key=key,
        status=PlannerCandidateStatus(metadata.pop("status")),
        qpos_path=arrays["qpos_path"],
        timestamps_seconds=arrays["timestamps_seconds"],
        eef_positions_world=arrays["eef_positions_world"],
        **metadata,
    )
    if candidate.fingerprint != expected_fingerprint:
        raise ValueError("single candidate fingerprint mismatch")
    if candidate.trajectory_fingerprint != expected_trajectory_fingerprint:
        raise ValueError("single candidate trajectory fingerprint mismatch")
    return candidate


def run_curobo_candidate_process(
    request_path: Path,
    result_path: Path,
    *,
    worker_python: Path = Path(".venv-expert-realization/bin/python"),
    process_timeout_seconds: float = 20.0,
) -> None:
    try:
        completed = subprocess.run(
            [
                str(worker_python),
                "-m",
                "latency_meta_mdp.expert_realization.curobo_worker",
                "--planner-request",
                str(request_path),
                "--planner-result",
                str(result_path),
            ],
            capture_output=True,
            text=True,
            timeout=process_timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        request = json.loads(Path(request_path).read_text(encoding="utf-8"))
        key = ExpertRealizationKey.from_mapping(
            request["expert_realization_key"]
        )
        candidate_index = request["candidate_index"]
        requested_seed = request["requested_seed"]
        if requested_seed != planner_candidate_seed(key, candidate_index):
            raise ValueError("planner request seed mismatch")
        write_single_candidate_result(
            result_path,
            PlannerCandidate.failure(
                expert_realization_key=key,
                candidate_index=candidate_index,
                requested_seed=requested_seed,
                effective_seed=requested_seed,
                status=PlannerCandidateStatus.TIMEOUT,
                reason=f"planner process exceeded {process_timeout_seconds} seconds",
                planning_time_seconds=float(process_timeout_seconds),
            ),
        )
        return
    if completed.returncode != 0:
        diagnostic = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"CuRobo candidate worker failed: {diagnostic[-2000:]}")


def generate_planner_candidates(
    *,
    expert_realization_key: ExpertRealizationKey,
    structured_expert_config_sha256: str,
    bridge: PandaPlanningBridge,
    plan: InterceptionPlan,
    start_qpos: np.ndarray,
    worker_python: Path = Path(".venv-expert-realization/bin/python"),
) -> tuple[PlannerCandidate, ...]:
    candidates = []
    with tempfile.TemporaryDirectory(prefix="structured-candidates-") as directory:
        root = Path(directory)
        def run_index(index: int, *, suffix: str = "") -> PlannerCandidate:
            request = root / f"request-{index:03d}.json"
            result = root / f"result-{index:03d}{suffix}"
            write_planner_request(
                expert_realization_key=expert_realization_key,
                structured_expert_config_sha256=structured_expert_config_sha256,
                candidate_index=index,
                bridge=bridge,
                plan=plan,
                start_qpos=start_qpos,
                timeout_seconds=5.0,
                path=request,
            )
            run_curobo_candidate_process(request, result, worker_python=worker_python)
            return load_single_candidate_result(
                result,
                structured_expert_config_sha256=structured_expert_config_sha256,
            )
        for index in range(8):
            candidates.append(run_index(index))
        repeated_first = run_index(0, suffix="-repeat")
        first = candidates[0]
        deterministic = first.trajectory_fingerprint == repeated_first.trajectory_fingerprint
        candidates[0] = replace(first, deterministic_replay_verified=deterministic)
    return tuple(candidates)
