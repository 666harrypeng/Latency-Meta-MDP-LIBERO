"""Formal source planning and paired master-block orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from latency_meta_mdp.expert_realization.contracts import ExpertRealizationKey
from latency_meta_mdp.expert_realization.planner import (
    PlannerCandidate,
    PlannerCandidateStatus,
)
from latency_meta_mdp.expert_realization.selector import (
    DiversitySelectionError,
    SelectedReference,
    discrete_frechet,
    freeze_selected_reference,
)
from latency_meta_mdp.expert_realization.source_corpus.config import (
    SourceExecutionConfig,
)


class PlannerInfrastructureError(RuntimeError):
    """The planner request did not complete for an infrastructure reason."""


class PlanningAttemptsExhausted(RuntimeError):
    """Every bounded semantic candidate failed qualification."""


class PlannerDeterminismError(RuntimeError):
    """An exact replay changed the selected planner result."""


class DrawRejected(RuntimeError):
    """One semantic realization draw failed an admission gate."""

    def __init__(self, failure_class: str, reason: str) -> None:
        if failure_class not in {
            "planner_failure",
            "task_failure",
            "safety_failure",
            "diversity_rejection",
        }:
            raise ValueError("failure_class is not a semantic draw rejection")
        if type(reason) is not str or not reason:
            raise ValueError("draw rejection reason must be non-empty")
        self.failure_class = failure_class
        self.reason = reason
        super().__init__(reason)


class LevelQuotaExhausted(RuntimeError):
    """A task level could not fill its bounded successful-realization quota."""


@dataclass(frozen=True)
class SourceSuccessPayloadRef:
    logical_master_task_index: int
    level: int
    realization_draw_index: int
    accepted_slot: int
    payload_path: Path

    def __post_init__(self) -> None:
        if type(self.logical_master_task_index) is not int or self.logical_master_task_index < 0:
            raise ValueError("success payload master task index must be non-negative")
        if self.level not in (1, 2, 3):
            raise ValueError("success payload level must be L1, L2, or L3")
        if type(self.realization_draw_index) is not int or self.realization_draw_index < 0:
            raise ValueError("success payload draw index must be non-negative")
        if type(self.accepted_slot) is not int or self.accepted_slot < 0:
            raise ValueError("success payload accepted slot must be non-negative")
        path = Path(self.payload_path)
        if not path.is_absolute():
            raise ValueError("success payload path must be absolute")
        object.__setattr__(self, "payload_path", path)

    @property
    def realization_index(self) -> int:
        """Compatibility alias for consumers that order final accepted episodes."""
        return self.accepted_slot


@dataclass(frozen=True)
class CompletedLevelQuota:
    logical_master_task_index: int
    level: int
    successes: tuple[SourceSuccessPayloadRef, ...]
    next_draw_index: int

    def __post_init__(self) -> None:
        if type(self.logical_master_task_index) is not int or self.logical_master_task_index < 0:
            raise ValueError("logical master task index must be non-negative")
        if self.level not in (1, 2, 3):
            raise ValueError("level quota must belong to L1, L2, or L3")
        if type(self.successes) is not tuple or not self.successes:
            raise ValueError("completed level quota requires successes")
        if [row.accepted_slot for row in self.successes] != list(range(len(self.successes))):
            raise ValueError("completed level quota requires dense accepted slots")
        if any(
            row.logical_master_task_index != self.logical_master_task_index
            or row.level != self.level
            for row in self.successes
        ):
            raise ValueError("level quota successes do not share their parent identity")
        if len({row.realization_draw_index for row in self.successes}) != len(self.successes):
            raise ValueError("level quota draw indices must be unique")
        if type(self.next_draw_index) is not int or self.next_draw_index <= max(
            row.realization_draw_index for row in self.successes
        ):
            raise ValueError("next draw index must follow every admitted draw")


@dataclass(frozen=True)
class CompletedMasterBlock:
    logical_master_task_index: int
    successes: tuple[SourceSuccessPayloadRef, ...]

    def __post_init__(self) -> None:
        if type(self.logical_master_task_index) is not int or (
            self.logical_master_task_index < 0
        ):
            raise ValueError("logical master task index must be non-negative")
        if type(self.successes) is not tuple or any(
            not isinstance(value, SourceSuccessPayloadRef) for value in self.successes
        ):
            raise ValueError(
                "completed master block requires typed successes and complete accepted slots"
            )
        by_level = {
            level: sorted(
                row.accepted_slot for row in self.successes if row.level == level
            )
            for level in (1, 2, 3)
        }
        counts = {len(slots) for slots in by_level.values()}
        if (
            len(counts) != 1
            or counts == {0}
            or any(slots != list(range(len(slots))) for slots in by_level.values())
            or any(
                row.logical_master_task_index != self.logical_master_task_index
                for row in self.successes
            )
        ):
            raise ValueError("completed master block requires complete accepted slots")


def fill_level_success_quota(
    *,
    logical_master_task_index: int,
    level: int,
    realization_quota: int,
    maximum_draws: int,
    start_draw_index: int,
    existing_successes: tuple[SourceSuccessPayloadRef, ...],
    plan_draw: Callable[[int, tuple[SourceSuccessPayloadRef, ...]], object],
    execute_draw: Callable[[object, int], SourceSuccessPayloadRef],
    on_rejected: Callable[[int, str, str], None],
) -> CompletedLevelQuota:
    """Fill one level with successes while replacing only rejected semantic draws."""
    for name, value in (
        ("logical_master_task_index", logical_master_task_index),
        ("level", level),
        ("realization_quota", realization_quota),
        ("maximum_draws", maximum_draws),
        ("start_draw_index", start_draw_index),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if level not in (1, 2, 3) or realization_quota <= 0 or maximum_draws < realization_quota:
        raise ValueError("level/quota/draw budget is invalid")
    if not all(callable(value) for value in (plan_draw, execute_draw, on_rejected)):
        raise TypeError("quota callbacks must be callable")
    successes = list(existing_successes)
    if [row.accepted_slot for row in successes] != list(range(len(successes))):
        raise ValueError("existing successes must have dense accepted slots")
    draw_index = start_draw_index
    while len(successes) < realization_quota and draw_index < maximum_draws:
        try:
            planned = plan_draw(draw_index, tuple(successes))
            accepted = execute_draw(planned, len(successes))
        except DrawRejected as error:
            on_rejected(draw_index, error.failure_class, error.reason)
            draw_index += 1
            continue
        if not isinstance(accepted, SourceSuccessPayloadRef):
            raise TypeError("execute_draw must return SourceSuccessPayloadRef")
        if (
            accepted.logical_master_task_index != logical_master_task_index
            or accepted.level != level
            or accepted.realization_draw_index != draw_index
            or accepted.accepted_slot != len(successes)
        ):
            raise ValueError("execute_draw returned a mismatched success identity")
        successes.append(accepted)
        draw_index += 1
    if len(successes) != realization_quota:
        raise LevelQuotaExhausted(
            f"master {logical_master_task_index} L{level} could not collect "
            f"{realization_quota} successes within {maximum_draws} draws"
        )
    return CompletedLevelQuota(
        logical_master_task_index=logical_master_task_index,
        level=level,
        successes=tuple(successes),
        next_draw_index=draw_index,
    )


@dataclass(frozen=True)
class FirstQualifiedPlan:
    candidate: PlannerCandidate
    reference: SelectedReference
    attempted_candidate_indices: tuple[int, ...]
    semantic_failures: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, PlannerCandidate) or (
            self.candidate.status is not PlannerCandidateStatus.SUCCESS
        ):
            raise ValueError("first-qualified plan requires one successful candidate")
        if not isinstance(self.reference, SelectedReference) or (
            self.reference.expert_realization_key
            != self.candidate.expert_realization_key
        ):
            raise ValueError("selected reference does not match its candidate")
        if (
            type(self.attempted_candidate_indices) is not tuple
            or self.attempted_candidate_indices
            != tuple(range(self.candidate.candidate_index + 1))
        ):
            raise ValueError("attempted candidate indices must be contiguous from zero")
        if type(self.semantic_failures) is not tuple or len(self.semantic_failures) != (
            len(self.attempted_candidate_indices) - 1
        ):
            raise ValueError("semantic failure history does not match attempted candidates")
        if any(type(value) is not str or not value for value in self.semantic_failures):
            raise ValueError("semantic failure history must contain non-empty strings")


def generate_single_draw_plan(
    *,
    realization_key: ExpertRealizationKey,
    intent: object,
    execution: SourceExecutionConfig,
    run_candidate: Callable[[int], PlannerCandidate],
    on_progress: Callable[[str], None],
) -> FirstQualifiedPlan:
    """Plan one semantic draw; only infrastructure failure repeats candidate zero."""
    if not isinstance(realization_key, ExpertRealizationKey):
        raise TypeError("realization_key must be ExpertRealizationKey")
    if not isinstance(execution, SourceExecutionConfig):
        raise TypeError("execution must be SourceExecutionConfig")
    if execution.planner_candidates_per_draw != 1:
        raise ValueError("one semantic draw requires exactly one planner candidate")
    if not callable(run_candidate) or not callable(on_progress):
        raise TypeError("planner runner and progress callback must be callable")
    infrastructure_failures = 0
    while True:
        on_progress("candidate=0 start")
        try:
            candidate = run_candidate(0)
        except PlannerInfrastructureError as error:
            infrastructure_failures += 1
            on_progress(f"candidate=0 infrastructure_retry={infrastructure_failures}")
            if infrastructure_failures > execution.infrastructure_retry_limit:
                raise PlannerInfrastructureError(
                    "candidate 0 infrastructure retry limit exhausted"
                ) from error
            continue
        break
    if not isinstance(candidate, PlannerCandidate):
        raise TypeError("planner runner must return PlannerCandidate")
    if candidate.expert_realization_key != realization_key or candidate.candidate_index != 0:
        raise ValueError("planner candidate identity does not match its draw")
    if candidate.status is not PlannerCandidateStatus.SUCCESS:
        on_progress(f"candidate=0 {candidate.status.value}")
        raise PlanningAttemptsExhausted(
            f"semantic draw {realization_key.realization_index} candidate 0 failed: "
            f"{candidate.failure_reason}"
        )
    orientation = getattr(getattr(intent, "approach", None), "fixed_orientation_world", None)
    on_progress("candidate=0 qualified frozen=true")
    return FirstQualifiedPlan(
        candidate=candidate,
        reference=freeze_selected_reference(
            candidate,
            fixed_orientation_world=orientation,
        ),
        attempted_candidate_indices=(0,),
        semantic_failures=(),
    )


def qualify_draw_diversity(
    candidate_path: np.ndarray,
    admitted_paths: tuple[np.ndarray, ...],
    *,
    minimum_frechet_m: float,
) -> None:
    """Reject one draw when its path duplicates an already admitted realization."""
    candidate = np.asarray(candidate_path, dtype=np.float64)
    if candidate.ndim != 2 or candidate.shape[1] != 3 or not np.isfinite(candidate).all():
        raise ValueError("candidate path must be a finite [T,3] array")
    if type(admitted_paths) is not tuple:
        raise TypeError("admitted_paths must be a tuple")
    if type(minimum_frechet_m) is not float or minimum_frechet_m <= 0.0:
        raise ValueError("minimum_frechet_m must be a positive float")
    for path in admitted_paths:
        previous = np.asarray(path, dtype=np.float64)
        if previous.ndim != 2 or previous.shape[1] != 3 or not np.isfinite(previous).all():
            raise ValueError("admitted path must be a finite [T,3] array")
        if discrete_frechet(candidate, previous) < minimum_frechet_m:
            raise DiversitySelectionError(
                f"semantic draw violates the {minimum_frechet_m * 1000:g} mm diversity gate"
            )


@dataclass(frozen=True)
class PlannerCanaryResult:
    level: int
    candidate_index: int
    requested_seed: int
    passed: bool

    def __post_init__(self) -> None:
        if type(self.level) is not int or self.level not in (1, 2, 3):
            raise ValueError("canary level must be L1, L2, or L3")
        if type(self.candidate_index) is not int or not 0 <= self.candidate_index < 8:
            raise ValueError("canary candidate index must be in 0..7")
        if type(self.requested_seed) is not int or not 0 <= self.requested_seed < 2**64:
            raise ValueError("canary requested seed must be uint64")
        if self.passed is not True:
            raise ValueError("published planner canary result must have passed")


def verify_level_planner_canary(
    selected: FirstQualifiedPlan,
    *,
    replay_candidate: Callable[[], PlannerCandidate],
) -> PlannerCanaryResult:
    """Require an exact numerical replay while ignoring nondeterministic wall time."""
    if not isinstance(selected, FirstQualifiedPlan):
        raise TypeError("selected must be FirstQualifiedPlan")
    if not callable(replay_candidate):
        raise TypeError("replay_candidate must be callable")
    expected = selected.candidate
    replay = replay_candidate()
    if not isinstance(replay, PlannerCandidate):
        raise TypeError("planner canary must return PlannerCandidate")
    scalar_equal = (
        replay.expert_realization_key == expected.expert_realization_key
        and replay.candidate_index == expected.candidate_index
        and replay.requested_seed == expected.requested_seed
        and replay.effective_seed == expected.effective_seed
        and replay.status is expected.status
        and replay.trajectory_fingerprint == expected.trajectory_fingerprint
        and replay.goal_position_error_m == expected.goal_position_error_m
        and replay.goal_rotation_error_degrees == expected.goal_rotation_error_degrees
        and replay.planner_cost == expected.planner_cost
    )
    arrays_equal = all(
        np.array_equal(getattr(replay, name), getattr(expected, name))
        for name in (
            "geometric_seed_qpos_path",
            "qpos_path",
            "timestamps_seconds",
            "eef_positions_world",
        )
    )
    if not scalar_equal or not arrays_equal:
        raise PlannerDeterminismError("planner canary numerical replay changed")
    return PlannerCanaryResult(
        level=expected.expert_realization_key.task_instance_id.level,
        candidate_index=expected.candidate_index,
        requested_seed=expected.requested_seed,
        passed=True,
    )


def schedule_success_quota_master_blocks(
    *,
    master_task_indices: tuple[int, ...],
    target_block_count: int,
    collect_block: Callable[[int], CompletedMasterBlock],
    publish_blocks: Callable[[tuple[CompletedMasterBlock, ...]], Path],
) -> Path:
    """Collect complete quota blocks and activate reserves only after quota exhaustion."""
    if (
        type(master_task_indices) is not tuple
        or not master_task_indices
        or any(type(value) is not int or value < 0 for value in master_task_indices)
        or tuple(sorted(set(master_task_indices))) != master_task_indices
    ):
        raise ValueError("master_task_indices must be a sorted unique non-empty tuple")
    if (
        type(target_block_count) is not int
        or target_block_count <= 0
        or target_block_count > len(master_task_indices)
    ):
        raise ValueError("target_block_count is outside the declared master-task universe")
    if not callable(collect_block) or not callable(publish_blocks):
        raise TypeError("block collector and publisher must be callable")
    admitted = []
    for logical_master_task_index in master_task_indices:
        if len(admitted) == target_block_count:
            break
        try:
            completed = collect_block(logical_master_task_index)
        except LevelQuotaExhausted:
            continue
        if not isinstance(completed, CompletedMasterBlock) or (
            completed.logical_master_task_index != logical_master_task_index
        ):
            raise ValueError("block collector returned a mismatched completed block")
        admitted.append(completed)
    if len(admitted) != target_block_count:
        raise RuntimeError("formal source reserve blocks were exhausted before target admission")
    result = publish_blocks(tuple(admitted))
    if not isinstance(result, Path):
        raise TypeError("block publisher must return a manifest Path")
    return result
