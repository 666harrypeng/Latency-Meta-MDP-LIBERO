from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


def _key(slot: int = 0):
    from latency_meta_mdp.expert_realization.contracts import (
        ExpertRealizationKey,
        TaskInstanceId,
    )

    return ExpertRealizationKey(
        TaskInstanceId(1, 4000, "a" * 64, "b" * 64), slot, "c" * 64
    )


def _success(index: int, *, slot: int = 0, offset: float = 0.0):
    from latency_meta_mdp.expert_realization.planner import (
        PlannerCandidate,
        PlannerCandidateStatus,
        planner_candidate_seed,
    )

    key = _key(slot)
    seed = planner_candidate_seed(key, index)
    qpos = np.tile(np.arange(7, dtype=np.float64), (3, 1)) + offset
    eef = np.array(
        [[0.0, 0.0, 1.0], [0.05, 0.0, 1.0], [0.10, 0.0, 1.0]],
        dtype=np.float64,
    ) + offset
    return PlannerCandidate(
        expert_realization_key=key,
        candidate_index=index,
        requested_seed=seed,
        effective_seed=seed,
        status=PlannerCandidateStatus.SUCCESS,
        geometric_seed_qpos_path=qpos,
        qpos_path=qpos,
        timestamps_seconds=np.array([0.0, 0.02, 0.04], dtype=np.float64),
        eef_positions_world=eef,
        eef_path_length_m=0.10,
        certified_clearance_lower_bound_m=0.0,
        goal_position_error_m=1.0e-6,
        goal_rotation_error_degrees=1.0e-4,
        planner_cost=1.0,
        planning_time_seconds=0.5,
        failure_reason=None,
        deterministic_replay_verified=None,
    )


def _failure(index: int, *, timeout: bool = False):
    from latency_meta_mdp.expert_realization.planner import (
        PlannerCandidate,
        PlannerCandidateStatus,
        planner_candidate_seed,
    )

    key = _key()
    seed = planner_candidate_seed(key, index)
    return PlannerCandidate.failure(
        expert_realization_key=key,
        candidate_index=index,
        requested_seed=seed,
        effective_seed=seed,
        status=(
            PlannerCandidateStatus.TIMEOUT
            if timeout
            else PlannerCandidateStatus.PLANNER_FAILURE
        ),
        reason="planner did not produce a qualified path",
        planning_time_seconds=5.0,
    )


def _execution():
    from latency_meta_mdp.expert_realization.source_corpus.config import (
        load_source_execution_config,
    )

    return load_source_execution_config(
        "configs/source_corpus/panda_ball_formal_source_execution.yaml"
    )


def _intent():
    return SimpleNamespace(
        approach=SimpleNamespace(fixed_orientation_world=np.eye(3, dtype=np.float64))
    )


def _task_instance():
    from latency_meta_mdp.expert_realization.task_instance import MaterializedTaskInstance

    value = object.__new__(MaterializedTaskInstance)
    object.__setattr__(value, "task_instance_id", _key().task_instance_id)
    object.__setattr__(
        value,
        "expected_anchor",
        SimpleNamespace(anchor_eef_orientation_matrix_world=np.eye(3, dtype=np.float64)),
    )
    return value


@pytest.fixture(autouse=True)
def _validated_task(monkeypatch: pytest.MonkeyPatch) -> None:
    from latency_meta_mdp.expert_realization.task_instance import MaterializedTaskInstance

    monkeypatch.setattr(
        MaterializedTaskInstance,
        "validate_publication_consistency",
        lambda self: None,
    )


def test_first_candidate_success_stops_without_building_a_candidate_bank() -> None:
    """Break caught: all eight candidates are generated before the first success is frozen."""
    from latency_meta_mdp.expert_realization.source_corpus.formal_collection import (
        generate_first_qualified_plan,
    )

    calls = []

    def run(index: int):
        calls.append(index)
        return _success(index)

    plan = generate_first_qualified_plan(
        realization_key=_key(),
        intent=_intent(),
        execution=_execution(),
        run_candidate=run,
        on_progress=lambda _message: None,
    )

    assert calls == [0]
    assert plan.candidate.candidate_index == 0
    assert plan.attempted_candidate_indices == (0,)
    assert plan.semantic_failures == ()


def test_semantic_failures_advance_monotonically_until_first_success() -> None:
    """Break caught: a failed plan terminates immediately or candidate indices are skipped."""
    from latency_meta_mdp.expert_realization.source_corpus.formal_collection import (
        generate_first_qualified_plan,
    )

    calls = []

    def run(index: int):
        calls.append(index)
        return _success(index) if index == 2 else _failure(index, timeout=index == 1)

    plan = generate_first_qualified_plan(
        realization_key=_key(),
        intent=_intent(),
        execution=_execution(),
        run_candidate=run,
        on_progress=lambda _message: None,
    )

    assert calls == [0, 1, 2]
    assert plan.candidate.candidate_index == 2
    assert plan.attempted_candidate_indices == (0, 1, 2)
    assert len(plan.semantic_failures) == 2


def test_exhaustion_never_requests_candidate_index_eight() -> None:
    """Break caught: an impossible strategy enters an unbounded planning loop."""
    from latency_meta_mdp.expert_realization.source_corpus.formal_collection import (
        PlanningAttemptsExhausted,
        generate_first_qualified_plan,
    )

    calls = []

    def run(index: int):
        calls.append(index)
        return _failure(index)

    with pytest.raises(PlanningAttemptsExhausted, match="0..7"):
        generate_first_qualified_plan(
            realization_key=_key(),
            intent=_intent(),
            execution=_execution(),
            run_candidate=run,
            on_progress=lambda _message: None,
        )
    assert calls == list(range(8))


def test_infrastructure_retry_repeats_the_same_candidate_identity() -> None:
    """Break caught: a worker crash advances the semantic candidate seed."""
    from latency_meta_mdp.expert_realization.source_corpus.formal_collection import (
        PlannerInfrastructureError,
        generate_first_qualified_plan,
    )

    calls = []

    def run(index: int):
        calls.append(index)
        if len(calls) == 1:
            raise PlannerInfrastructureError("worker exited")
        return _success(index)

    plan = generate_first_qualified_plan(
        realization_key=_key(),
        intent=_intent(),
        execution=_execution(),
        run_candidate=run,
        on_progress=lambda _message: None,
    )

    assert calls == [0, 0]
    assert plan.attempted_candidate_indices == (0,)
    assert plan.candidate.candidate_index == 0


def test_infrastructure_retry_limit_aborts_without_advancing_candidate() -> None:
    """Break caught: repeated infrastructure failure is mislabeled semantic infeasibility."""
    from latency_meta_mdp.expert_realization.source_corpus.formal_collection import (
        PlannerInfrastructureError,
        generate_first_qualified_plan,
    )

    calls = []

    def run(index: int):
        calls.append(index)
        raise PlannerInfrastructureError("worker exited")

    with pytest.raises(PlannerInfrastructureError, match="retry limit"):
        generate_first_qualified_plan(
            realization_key=_key(),
            intent=_intent(),
            execution=_execution(),
            run_candidate=run,
            on_progress=lambda _message: None,
        )
    assert calls == [0, 0, 0]


def _first_plan(*, slot: int, offset: float):
    from latency_meta_mdp.expert_realization.selector import freeze_selected_reference
    from latency_meta_mdp.expert_realization.source_corpus.formal_collection import (
        FirstQualifiedPlan,
    )

    candidate = _success(0, slot=slot, offset=offset)
    return FirstQualifiedPlan(
        candidate=candidate,
        reference=freeze_selected_reference(
            candidate, fixed_orientation_world=np.eye(3, dtype=np.float64)
        ),
        attempted_candidate_indices=(0,),
        semantic_failures=(),
    )


def test_first_qualified_selector_requires_four_numerically_distinct_plans() -> None:
    """Break caught: a formal four-realization group is incomplete or fingerprint-only diverse."""
    from latency_meta_mdp.expert_realization.selector import (
        DiversitySelectionError,
        select_first_qualified_plan_set,
    )

    plans = {_key(slot): _first_plan(slot=slot, offset=0.02 * slot) for slot in range(4)}
    selected = select_first_qualified_plan_set(
        task_instance=_task_instance(), plans_by_key=plans
    )
    assert tuple(selected.references) == (0, 1, 2, 3)
    assert selected.selected_candidate_fingerprints == {
        slot: plans[_key(slot)].candidate.fingerprint for slot in range(4)
    }

    with pytest.raises(ValueError, match="four"):
        select_first_qualified_plan_set(
            task_instance=_task_instance(),
            plans_by_key={key: value for key, value in plans.items() if key.realization_index < 3},
        )
    duplicate = dict(plans)
    duplicate[_key(1)] = _first_plan(slot=1, offset=0.0)
    with pytest.raises(DiversitySelectionError, match="5 mm"):
        select_first_qualified_plan_set(
            task_instance=_task_instance(), plans_by_key=duplicate
        )


def test_level_canary_ignores_wall_time_but_requires_exact_trajectory() -> None:
    """Break caught: deterministic replay compares wall time or misses numerical path drift."""
    from dataclasses import replace

    from latency_meta_mdp.expert_realization.source_corpus.formal_collection import (
        PlannerDeterminismError,
        verify_level_planner_canary,
    )

    selected = _first_plan(slot=0, offset=0.0)
    replay = replace(selected.candidate, planning_time_seconds=9.0)
    result = verify_level_planner_canary(selected, replay_candidate=lambda: replay)
    assert result.level == 1
    assert result.candidate_index == 0
    assert result.requested_seed == selected.candidate.requested_seed
    assert result.passed is True

    changed = _success(0, slot=0, offset=0.001)
    with pytest.raises(PlannerDeterminismError, match="numerical replay"):
        verify_level_planner_canary(selected, replay_candidate=lambda: changed)
