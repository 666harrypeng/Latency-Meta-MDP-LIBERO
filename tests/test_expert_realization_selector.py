from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest
from test_expert_realization_planner import _candidate, _key


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


def _candidate_map():
    return {
        _key(key_index): tuple(
            _candidate(index, key_index=key_index, offset=0.02 * key_index + 0.001 * index)
            for index in range(8)
        )
        for key_index in range(8)
    }


def test_complete_selection_is_input_order_invariant_and_50hz() -> None:
    """Break caught: worker completion order changes selected plans or reference timing."""
    from latency_meta_mdp.expert_realization.selector import select_task_instance_plan_set

    candidates = _candidate_map()
    expected = select_task_instance_plan_set(
        task_instance=_task_instance(),
        candidates_by_key=candidates,
    )
    items = list(candidates.items())
    random.Random(12).shuffle(items)
    shuffled = {key: tuple(reversed(rows)) for key, rows in items}
    actual = select_task_instance_plan_set(
        task_instance=_task_instance(),
        candidates_by_key=shuffled,
    )

    assert actual.candidate_set_sha256 == expected.candidate_set_sha256
    assert actual.selected_candidate_fingerprints == expected.selected_candidate_fingerprints
    assert len(actual.references) == 8
    for reference in actual.references.values():
        assert np.allclose(np.diff(reference.timestamps_seconds), 0.02)
        assert reference.eef_positions_world.shape[1] == 3
        assert reference.fixed_orientation_world.shape == (3, 3)
        source = candidates[_key(reference.expert_realization_key.realization_index)][0]
        assert np.array_equal(reference.qpos_path, source.qpos_path)
        assert np.array_equal(reference.timestamps_seconds, source.timestamps_seconds)


def test_selection_accepts_any_positive_contiguous_realization_count() -> None:
    """Break caught: a general R-realization request is forced back to the old eight-slot pilot."""
    from latency_meta_mdp.expert_realization.selector import select_task_instance_plan_set

    candidates = {
        _key(key_index): tuple(
            _candidate(index, key_index=key_index, offset=0.02 * key_index + 0.001 * index)
            for index in range(8)
        )
        for key_index in range(3)
    }
    selected = select_task_instance_plan_set(
        task_instance=_task_instance(),
        candidates_by_key=candidates,
    )

    assert set(selected.references) == {0, 1, 2}
    assert set(selected.selected_candidate_fingerprints) == {0, 1, 2}


def test_selection_rejects_incomplete_candidate_universe() -> None:
    """Break caught: a missing realization/candidate silently receives canonical fallback."""
    from latency_meta_mdp.expert_realization.selector import select_task_instance_plan_set

    candidates = _candidate_map()
    del candidates[_key(3)]
    with pytest.raises(ValueError, match="complete"):
        select_task_instance_plan_set(
            task_instance=_task_instance(),
            candidates_by_key=candidates,
        )


def test_all_eight_planner_failures_do_not_fallback_to_canonical() -> None:
    """Break caught: a failed realization is silently replaced by a canonical plan."""
    from latency_meta_mdp.expert_realization.planner import (
        PlannerCandidate,
        PlannerCandidateStatus,
        planner_candidate_seed,
    )
    from latency_meta_mdp.expert_realization.selector import select_task_instance_plan_set

    candidates = _candidate_map()
    key = _key(3)
    candidates[key] = tuple(
        PlannerCandidate.failure(
            expert_realization_key=key,
            candidate_index=index,
            requested_seed=planner_candidate_seed(key, index),
            effective_seed=planner_candidate_seed(key, index),
            status=PlannerCandidateStatus.PLANNER_FAILURE,
            reason="no feasible plan",
            planning_time_seconds=0.5,
        )
        for index in range(8)
    )
    with pytest.raises(ValueError, match="no successful candidate"):
        select_task_instance_plan_set(
            task_instance=_task_instance(),
            candidates_by_key=candidates,
        )


def test_near_duplicate_eef_candidates_cannot_fill_distinct_realization_slots() -> None:
    """Break caught: different joint paths masquerade as distinct task-space behaviors."""
    from latency_meta_mdp.expert_realization.selector import (
        DiversitySelectionError,
        select_task_instance_plan_set,
    )

    candidates = _candidate_map()
    duplicate = tuple(_candidate(index, key_index=1, offset=0.0) for index in range(8))
    candidates[_key(1)] = duplicate
    with pytest.raises(DiversitySelectionError):
        select_task_instance_plan_set(
            task_instance=_task_instance(),
            candidates_by_key=candidates,
        )
