"""Formal source planning and paired master-block orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from latency_meta_mdp.expert_realization.contracts import ExpertRealizationKey
from latency_meta_mdp.expert_realization.planner import (
    PlannerCandidate,
    PlannerCandidateStatus,
)
from latency_meta_mdp.expert_realization.selector import (
    SelectedReference,
    freeze_selected_reference,
)
from latency_meta_mdp.expert_realization.source_corpus.config import (
    SourceExecutionConfig,
)


class PlannerInfrastructureError(RuntimeError):
    """The planner request did not complete for an infrastructure reason."""


class PlanningAttemptsExhausted(RuntimeError):
    """Every bounded semantic candidate failed qualification."""


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


def generate_first_qualified_plan(
    *,
    realization_key: ExpertRealizationKey,
    intent: object,
    execution: SourceExecutionConfig,
    run_candidate: Callable[[int], PlannerCandidate],
    on_progress: Callable[[str], None],
) -> FirstQualifiedPlan:
    """Generate candidates sequentially and stop at the first qualified result."""
    if not isinstance(realization_key, ExpertRealizationKey):
        raise TypeError("realization_key must be ExpertRealizationKey")
    if not isinstance(execution, SourceExecutionConfig):
        raise TypeError("execution must be SourceExecutionConfig")
    if not callable(run_candidate) or not callable(on_progress):
        raise TypeError("planner runner and progress callback must be callable")
    orientation = getattr(getattr(intent, "approach", None), "fixed_orientation_world", None)
    semantic_failures = []
    attempted_indices = []
    for candidate_index in range(execution.maximum_candidate_attempts_per_realization):
        infrastructure_failures = 0
        while True:
            on_progress(f"candidate={candidate_index} start")
            try:
                candidate = run_candidate(candidate_index)
            except PlannerInfrastructureError as error:
                infrastructure_failures += 1
                on_progress(
                    f"candidate={candidate_index} infrastructure_retry="
                    f"{infrastructure_failures}"
                )
                if infrastructure_failures > execution.infrastructure_retry_limit:
                    raise PlannerInfrastructureError(
                        f"candidate {candidate_index} infrastructure retry limit exhausted"
                    ) from error
                continue
            break
        if not isinstance(candidate, PlannerCandidate):
            raise TypeError("planner runner must return PlannerCandidate")
        if candidate.expert_realization_key != realization_key or (
            candidate.candidate_index != candidate_index
        ):
            raise ValueError("planner candidate identity does not match its request")
        attempted_indices.append(candidate_index)
        if candidate.status is PlannerCandidateStatus.SUCCESS:
            on_progress(f"candidate={candidate_index} qualified frozen=true")
            return FirstQualifiedPlan(
                candidate=candidate,
                reference=freeze_selected_reference(
                    candidate,
                    fixed_orientation_world=orientation,
                ),
                attempted_candidate_indices=tuple(attempted_indices),
                semantic_failures=tuple(semantic_failures),
            )
        semantic_failures.append(
            f"candidate={candidate_index} status={candidate.status.value}: "
            f"{candidate.failure_reason}"
        )
        on_progress(f"candidate={candidate_index} {candidate.status.value}")
    maximum = execution.maximum_candidate_attempts_per_realization - 1
    raise PlanningAttemptsExhausted(
        f"all bounded planner candidates 0..{maximum} failed qualification"
    )
