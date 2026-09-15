from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest


def _key(draw_index: int = 0):
    from latency_meta_mdp.data.collection.contracts import ExpertRealizationKey, TaskInstanceId

    return ExpertRealizationKey(
        TaskInstanceId(1, 4000, "a" * 64, "b" * 64),
        draw_index,
        "c" * 64,
    )


def _candidate(*, draw_index: int = 0, offset: float = 0.0, success: bool = True):
    from latency_meta_mdp.data.collection.planner import (
        PlannerCandidate,
        PlannerCandidateStatus,
        planner_candidate_seed,
    )

    key = _key(draw_index)
    seed = planner_candidate_seed(key, 0)
    if not success:
        return PlannerCandidate.failure(
            expert_realization_key=key,
            candidate_index=0,
            requested_seed=seed,
            effective_seed=seed,
            status=PlannerCandidateStatus.PLANNER_FAILURE,
            reason="infeasible",
            planning_time_seconds=1.0,
        )
    qpos = np.tile(np.arange(7, dtype=np.float64), (3, 1)) + offset
    eef = (
        np.array(
            [[0.0, 0.0, 1.0], [0.05, 0.0, 1.0], [0.10, 0.0, 1.0]],
            dtype=np.float64,
        )
        + offset
    )
    return PlannerCandidate(
        expert_realization_key=key,
        candidate_index=0,
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


def _intent():
    return SimpleNamespace(
        approach=SimpleNamespace(fixed_orientation_world=np.eye(3, dtype=np.float64))
    )


def _execution():
    from latency_meta_mdp.data.source.config import (
        load_source_execution_config,
    )

    return load_source_execution_config(
        "configs/data/source_corpus/panda_ball_formal_source_execution.yaml"
    )


def test_one_semantic_draw_requests_only_candidate_zero() -> None:
    from latency_meta_mdp.data.source.formal_collection import (
        generate_single_draw_plan,
    )

    calls = []
    plan = generate_single_draw_plan(
        realization_key=_key(4),
        intent=_intent(),
        execution=_execution(),
        run_candidate=lambda index: calls.append(index) or _candidate(draw_index=4),
        on_progress=lambda _message: None,
    )

    assert calls == [0]
    assert plan.candidate.candidate_index == 0
    assert plan.candidate.expert_realization_key.realization_index == 4


def test_planner_semantic_failure_rejects_draw_without_candidate_one() -> None:
    from latency_meta_mdp.data.source.formal_collection import (
        PlanningAttemptsExhausted,
        generate_single_draw_plan,
    )

    calls = []
    with pytest.raises(PlanningAttemptsExhausted, match="semantic draw"):
        generate_single_draw_plan(
            realization_key=_key(2),
            intent=_intent(),
            execution=_execution(),
            run_candidate=lambda index: calls.append(index)
            or _candidate(draw_index=2, success=False),
            on_progress=lambda _message: None,
        )
    assert calls == [0]


def test_infrastructure_retry_repeats_candidate_zero_for_same_draw() -> None:
    from latency_meta_mdp.data.source.formal_collection import (
        PlannerInfrastructureError,
        generate_single_draw_plan,
    )

    calls = []

    def run(index: int):
        calls.append(index)
        if len(calls) == 1:
            raise PlannerInfrastructureError("worker crashed")
        return _candidate(draw_index=3)

    plan = generate_single_draw_plan(
        realization_key=_key(3),
        intent=_intent(),
        execution=_execution(),
        run_candidate=run,
        on_progress=lambda _message: None,
    )
    assert calls == [0, 0]
    assert plan.candidate.expert_realization_key == _key(3)


def test_draw_diversity_is_checked_against_already_admitted_paths() -> None:
    from latency_meta_mdp.data.collection.selector import DiversitySelectionError
    from latency_meta_mdp.data.source.formal_collection import (
        qualify_draw_diversity,
    )

    baseline = _candidate().eef_positions_world
    with pytest.raises(DiversitySelectionError, match="5 mm"):
        qualify_draw_diversity(
            baseline + 0.001,
            (baseline,),
            minimum_frechet_m=0.005,
        )
    qualify_draw_diversity(
        baseline + 0.02,
        (baseline,),
        minimum_frechet_m=0.005,
    )


def test_level_canary_ignores_wall_time_but_requires_exact_trajectory() -> None:
    from latency_meta_mdp.data.source.formal_collection import (
        PlannerDeterminismError,
        generate_single_draw_plan,
        verify_level_planner_canary,
    )

    selected = generate_single_draw_plan(
        realization_key=_key(),
        intent=_intent(),
        execution=_execution(),
        run_candidate=lambda _index: _candidate(),
        on_progress=lambda _message: None,
    )
    replay = replace(selected.candidate, planning_time_seconds=9.0)
    result = verify_level_planner_canary(selected, replay_candidate=lambda: replay)
    assert result.passed is True

    changed = _candidate(offset=0.001)
    with pytest.raises(PlannerDeterminismError, match="numerical replay"):
        verify_level_planner_canary(selected, replay_candidate=lambda: changed)
