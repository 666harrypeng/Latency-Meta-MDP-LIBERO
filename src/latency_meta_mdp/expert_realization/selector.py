"""Deterministic complete-set candidate selection and 50 Hz reference freezing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from latency_meta_mdp.expert_realization.contracts import ExpertRealizationKey, TaskInstanceId
from latency_meta_mdp.expert_realization.planner import PlannerCandidate, PlannerCandidateStatus
from latency_meta_mdp.expert_realization.task_instance import (
    MaterializedTaskInstance,
    _readonly_exact,
)


class DiversitySelectionError(RuntimeError):
    """No complete nonduplicate selected-reference assignment exists."""


def _discrete_frechet(left: np.ndarray, right: np.ndarray) -> float:
    cache = np.full((len(left), len(right)), np.nan, dtype=np.float64)

    def distance(i: int, j: int) -> float:
        if np.isfinite(cache[i, j]):
            return float(cache[i, j])
        point = float(np.linalg.norm(left[i] - right[j]))
        if i == 0 and j == 0:
            value = point
        elif i == 0:
            value = max(distance(0, j - 1), point)
        elif j == 0:
            value = max(distance(i - 1, 0), point)
        else:
            value = max(
                min(distance(i - 1, j), distance(i - 1, j - 1), distance(i, j - 1)),
                point,
            )
        cache[i, j] = value
        return value

    return distance(len(left) - 1, len(right) - 1)


def _candidate_set_sha256(
    candidates_by_key: Mapping[ExpertRealizationKey, tuple[PlannerCandidate, ...]],
) -> str:
    rows = []
    for key in sorted(candidates_by_key, key=lambda item: item.realization_index):
        for candidate in sorted(candidates_by_key[key], key=lambda item: item.candidate_index):
            rows.append(
                {
                    "key": key.to_mapping(),
                    "candidate_index": candidate.candidate_index,
                    "fingerprint": candidate.fingerprint,
                }
            )
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class SelectedReference:
    expert_realization_key: ExpertRealizationKey
    source_candidate_fingerprint: str
    qpos_path: np.ndarray
    timestamps_seconds: np.ndarray
    eef_positions_world: np.ndarray
    fixed_orientation_world: np.ndarray
    fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.expert_realization_key, ExpertRealizationKey):
            raise TypeError("expert_realization_key must be an ExpertRealizationKey")
        for name in ("source_candidate_fingerprint", "fingerprint"):
            value = getattr(self, name)
            if type(value) is not str or len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256 digest")
        contracts = {
            "qpos_path": (np.dtype(np.float64), (None, 7)),
            "timestamps_seconds": (np.dtype(np.float64), (None,)),
            "eef_positions_world": (np.dtype(np.float64), (None, 3)),
            "fixed_orientation_world": (np.dtype(np.float64), (3, 3)),
        }
        for name, (dtype, shape) in contracts.items():
            object.__setattr__(
                self,
                name,
                _readonly_exact(getattr(self, name), dtype=dtype, shape=shape, name=name),
            )
        if not (
            len(self.qpos_path) == len(self.timestamps_seconds) == len(self.eef_positions_world)
        ):
            raise ValueError("selected reference arrays must be aligned")


@dataclass(frozen=True)
class FrozenTaskInstancePlanSet:
    task_instance_id: TaskInstanceId
    candidate_set_sha256: str
    selected_candidate_fingerprints: Mapping[int, str]
    references: Mapping[int, SelectedReference]

    def __post_init__(self) -> None:
        if not isinstance(self.task_instance_id, TaskInstanceId):
            raise TypeError("task_instance_id must be a TaskInstanceId")
        if type(self.candidate_set_sha256) is not str or len(self.candidate_set_sha256) != 64:
            raise ValueError("candidate_set_sha256 must be a SHA-256 digest")
        fingerprints_are_complete = set(self.selected_candidate_fingerprints) == set(range(8))
        references_are_complete = set(self.references) == set(range(8))
        if not fingerprints_are_complete or not references_are_complete:
            raise ValueError("frozen plan set must cover realization indices 0..7")
        object.__setattr__(
            self,
            "selected_candidate_fingerprints",
            MappingProxyType(dict(self.selected_candidate_fingerprints)),
        )
        object.__setattr__(self, "references", MappingProxyType(dict(self.references)))


def _resample_reference(
    candidate: PlannerCandidate,
    *,
    fixed_orientation_world: np.ndarray,
) -> SelectedReference:
    end_time = float(candidate.timestamps_seconds[-1])
    target_times = np.arange(0.0, end_time + 1.0e-12, 0.02, dtype=np.float64)
    if target_times[-1] < end_time - 1.0e-12:
        target_times = np.concatenate([target_times, np.array([end_time])])
    qpos = np.stack(
        [
            np.interp(target_times, candidate.timestamps_seconds, candidate.qpos_path[:, index])
            for index in range(7)
        ],
        axis=1,
    )
    eef = np.stack(
        [
            np.interp(
                target_times,
                candidate.timestamps_seconds,
                candidate.eef_positions_world[:, index],
            )
            for index in range(3)
        ],
        axis=1,
    )
    digest = hashlib.sha256()
    digest.update(candidate.fingerprint.encode())
    for value in (target_times, qpos, eef, fixed_orientation_world):
        digest.update(np.asarray(value).tobytes(order="C"))
    return SelectedReference(
        expert_realization_key=candidate.expert_realization_key,
        source_candidate_fingerprint=candidate.fingerprint,
        qpos_path=qpos.astype(np.float64),
        timestamps_seconds=target_times,
        eef_positions_world=eef.astype(np.float64),
        fixed_orientation_world=np.asarray(fixed_orientation_world, dtype=np.float64),
        fingerprint=digest.hexdigest(),
    )


def select_task_instance_plan_set(
    *,
    task_instance: MaterializedTaskInstance,
    candidates_by_key: Mapping[ExpertRealizationKey, tuple[PlannerCandidate, ...]],
) -> FrozenTaskInstancePlanSet:
    if not isinstance(task_instance, MaterializedTaskInstance):
        raise TypeError("task_instance must be a MaterializedTaskInstance")
    task_instance.validate_publication_consistency()
    expected_keys = tuple(
        sorted(candidates_by_key, key=lambda item: item.realization_index)
    )
    if (
        len(expected_keys) != 8
        or [key.realization_index for key in expected_keys] != list(range(8))
        or any(key.task_instance_id != task_instance.task_instance_id for key in expected_keys)
    ):
        raise ValueError("candidate map must cover the complete task-instance realization set")
    for key in expected_keys:
        rows = candidates_by_key[key]
        if type(rows) is not tuple or len(rows) != 8:
            raise ValueError("each realization requires exactly eight candidate records")
        ordered = sorted(rows, key=lambda item: item.candidate_index)
        if [item.candidate_index for item in ordered] != list(range(8)) or any(
            item.expert_realization_key != key for item in ordered
        ):
            raise ValueError("candidate records do not match their realization key/index")
    candidate_set_sha = _candidate_set_sha256(candidates_by_key)
    selected: dict[int, PlannerCandidate] = {}
    selected_paths: list[np.ndarray] = []
    for key in expected_keys:
        successful = sorted(
            (
                item
                for item in candidates_by_key[key]
                if item.status is PlannerCandidateStatus.SUCCESS
            ),
            key=lambda item: (
                item.planner_cost,
                item.eef_path_length_m,
                item.trajectory_fingerprint,
            ),
        )
        if not successful:
            raise ValueError(f"realization {key.realization_index} has no successful candidate")
        choice = next(
            (
                candidate
                for candidate in successful
                if all(
                    _discrete_frechet(candidate.eef_positions_world, path) >= 0.005
                    for path in selected_paths
                )
            ),
            None,
        )
        if choice is None:
            raise DiversitySelectionError(
                f"realization {key.realization_index} has only near-duplicate candidates"
            )
        selected[key.realization_index] = choice
        selected_paths.append(choice.eef_positions_world)
    orientation = task_instance.expected_anchor.anchor_eef_orientation_matrix_world
    references = {
        index: _resample_reference(candidate, fixed_orientation_world=orientation)
        for index, candidate in selected.items()
    }
    return FrozenTaskInstancePlanSet(
        task_instance_id=task_instance.task_instance_id,
        candidate_set_sha256=candidate_set_sha,
        selected_candidate_fingerprints={
            index: candidate.fingerprint for index, candidate in selected.items()
        },
        references=references,
    )
