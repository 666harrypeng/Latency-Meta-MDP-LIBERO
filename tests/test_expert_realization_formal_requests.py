from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from latency_meta_mdp.expert_realization.config import load_formal_corpus_config
from latency_meta_mdp.expert_realization.contracts import TaskInstanceId

FORMAL_CONFIG = Path("configs/collection/panda_ball_structured_formal.yaml")
STRUCTURED_CONFIG = Path("configs/expert_realization/panda_ball_structured.yaml")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build(config=None):
    from latency_meta_mdp.expert_realization.contracts import build_formal_request_universe

    return build_formal_request_universe(
        load_formal_corpus_config(FORMAL_CONFIG) if config is None else config,
        corpus_config_sha256=_sha(FORMAL_CONFIG),
        structured_expert_config_sha256=_sha(STRUCTURED_CONFIG),
    )


def test_formal_request_universe_is_complete_stable_and_order_independent() -> None:
    """Break caught: worker order or repeated construction changes semantic requests."""
    first = _build()
    second = _build()

    assert len(first.primary_tasks) == 100
    assert len(first.reserve_tasks) == 20
    assert [row.logical_task_index for row in first.primary_tasks] == list(range(100))
    assert [row.logical_task_index for row in first.reserve_tasks] == list(range(100, 120))
    assert first.primary_tasks[0].master_task_seed == 3484810939061999107
    assert first.primary_tasks[1].master_task_seed == 7577166538116206848
    assert first.reserve_tasks[0].master_task_seed == 14309134202372629248
    assert first.to_mapping() == second.to_mapping()
    assert json.dumps(first.to_mapping(), sort_keys=True, separators=(",", ":")) == json.dumps(
        second.to_mapping(), sort_keys=True, separators=(",", ":")
    )
    assert len(first.request_sha256) == 64


def test_formal_request_universe_round_trip_rejects_mapping_corruption() -> None:
    """Break caught: a stored request can change inventory or hash while remaining loadable."""
    from latency_meta_mdp.expert_realization.contracts import FormalRequestUniverse

    universe = _build()
    mapping = universe.to_mapping()
    assert FormalRequestUniverse.from_mapping(mapping) == universe

    corrupted_hash = {**mapping, "request_sha256": "0" * 64}
    with pytest.raises(ValueError, match="request_sha256"):
        FormalRequestUniverse.from_mapping(corrupted_hash)
    with pytest.raises(ValueError, match="unknown"):
        FormalRequestUniverse.from_mapping({**mapping, "unexpected": 1})
    missing = dict(mapping)
    del missing["reserve_tasks"]
    with pytest.raises(ValueError, match="missing"):
        FormalRequestUniverse.from_mapping(missing)


def test_hidden_family_allocation_is_independent_uniform_and_allows_repeated_modes() -> None:
    """Break caught: each task is forced into a family quota, not sampled behavior."""
    from latency_meta_mdp.expert_realization.contracts import StrategyFamily

    universe = _build()
    assignments0 = universe.family_assignments[0]
    assignments1 = universe.family_assignments[1]

    assert [row.realization_slot for row in assignments0] == list(range(4))
    assert tuple(row.family for row in assignments1) != tuple(row.family for row in assignments0)
    assert any(
        len({row.family for row in rows}) < len(rows)
        for rows in universe.family_assignments.values()
    )
    assert {row.family for rows in universe.family_assignments.values() for row in rows} == set(
        StrategyFamily
    )

    three = replace(
        load_formal_corpus_config(FORMAL_CONFIG),
        task_instance_count=2,
        realizations_per_task=3,
        reserve_task_instance_count=0,
    )
    three_universe = _build(three)
    assert all(len(rows) == 3 for rows in three_universe.family_assignments.values())


def test_formal_realization_requests_bind_task_family_namespace_and_unique_seed() -> None:
    """Break caught: one task's hidden family assignment is not part of its semantic request."""
    from latency_meta_mdp.expert_realization.contracts import (
        build_formal_realization_requests,
    )

    universe = _build()
    task = TaskInstanceId(
        level=1,
        task_instance_seed=universe.primary_tasks[0].master_task_seed,
        motion_profile_sha256="a" * 64,
        initial_state_sha256="b" * 64,
    )
    requests = build_formal_realization_requests(universe, task)

    assert [row.realization_slot for row in requests] == list(range(4))
    assert [row.assigned_family for row in requests] == [
        row.family for row in universe.family_assignments[0]
    ]
    assert len({row.realization_seed for row in requests}) == 4
    assert len({row.realization_namespace_sha256 for row in requests}) == 1
    assert all(row.task_instance_id == task for row in requests)
    assert tuple(type(row).from_mapping(row.to_mapping()) for row in requests) == requests


@pytest.mark.parametrize(
    ("existing", "requested", "valid"),
    [
        (((0, 100),), (99, 120), False),
        (((0, 100),), (100, 200), True),
        (((0, 100), (200, 300)), (150, 210), False),
        ((), (0, 100), True),
    ],
)
def test_corpus_extension_rejects_interval_overlap(existing, requested, valid: bool) -> None:
    """Break caught: a later collection shard silently repeats prior task identities."""
    from latency_meta_mdp.expert_realization.contracts import (
        validate_nonoverlapping_extension,
    )

    if valid:
        validate_nonoverlapping_extension(existing, requested)
    else:
        with pytest.raises(ValueError, match="overlap"):
            validate_nonoverlapping_extension(existing, requested)


def test_formal_request_types_reject_seed_slot_and_inventory_drift() -> None:
    """Break caught: caller-authored rows bypass canonical seed and complete-slot derivation."""
    from latency_meta_mdp.expert_realization.contracts import (
        FamilySlotAssignment,
        MasterTaskRequest,
        StrategyFamily,
    )

    with pytest.raises(ValueError, match="seed"):
        MasterTaskRequest(
            corpus_id="panda-ball-structured-formal-v1",
            logical_task_index=0,
            master_task_seed=0,
            reserve=False,
        )
    with pytest.raises(ValueError, match="slot"):
        FamilySlotAssignment(realization_slot=-1, family=StrategyFamily.CANONICAL_DIRECT)
