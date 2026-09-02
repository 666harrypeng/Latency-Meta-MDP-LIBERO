from __future__ import annotations

from dataclasses import replace

import pytest

from latency_meta_mdp.expert_realization.contracts import TaskInstanceId

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64


def _task() -> TaskInstanceId:
    return TaskInstanceId(1, 4000, SHA_A, SHA_B)


def _key(index: int):
    from latency_meta_mdp.expert_realization.contracts import ExpertRealizationKey

    return ExpertRealizationKey(_task(), index, SHA_C)


def _row(index: int):
    from latency_meta_mdp.expert_realization.artifacts import ArtifactRef, FrozenPlanRow

    return FrozenPlanRow(
        expert_realization_key=_key(index),
        strategy=ArtifactRef(f"strategies/realization-{index:03d}.json", SHA_A),
        planner_candidates=ArtifactRef(
            f"planner_candidates/realization-{index:03d}.jsonl", SHA_B
        ),
        selection_status="selected",
        selected_candidate_fingerprint=SHA_C,
        selected_reference=ArtifactRef(
            f"selected_references/realization-{index:03d}.npz", SHA_D
        ),
    )


def _implementation():
    from latency_meta_mdp.expert_realization.recording_contracts import ImplementationIdentity

    return ImplementationIdentity(revision="test", source_sha256=SHA_A, dirty=False)


def test_realization_key_serializes_namespace_and_accepts_request_bound_slot() -> None:
    """Break caught: a formal slot is rejected by the old global 0..7 key contract."""
    from latency_meta_mdp.expert_realization.contracts import ExpertRealizationKey

    key = _key(12)
    mapping = key.to_mapping()

    assert mapping["realization_namespace_sha256"] == SHA_C
    assert ExpertRealizationKey.from_mapping(mapping) == key
    assert key.realization_index == 12


def test_four_and_eight_slot_universes_bind_exact_plan_rows() -> None:
    """Break caught: a frozen plan silently assumes eight rows or accepts a missing formal row."""
    from latency_meta_mdp.expert_realization.artifacts import (
        ArtifactRef,
        FrozenPlanSetManifest,
    )
    from latency_meta_mdp.expert_realization.contracts import (
        build_realization_universe_identity,
    )

    for slots in (tuple(range(4)), tuple(range(8))):
        universe = build_realization_universe_identity(
            request_sha256=SHA_D,
            task_instance_id=_task(),
            realization_slots=slots,
        )
        manifest = FrozenPlanSetManifest(
            schema_version=1,
            format_id="structured_expert_plan_set_v1",
            task_instance_id=_task(),
            task_instance_manifest=ArtifactRef(
                "task_instances/L1/seed-4000/task_instance/manifest.json",
                SHA_A,
            ),
            structured_expert_config_sha256=SHA_B,
            curobo_planner_config_sha256=SHA_C,
            realization_universe_sha256=universe.universe_sha256,
            realization_slots=slots,
            implementation=_implementation(),
            realizations=tuple(_row(index) for index in slots),
        )
        assert FrozenPlanSetManifest.from_mapping(manifest.to_mapping()) == manifest
        with pytest.raises(ValueError, match="universe"):
            replace(manifest, realizations=manifest.realizations[:-1])


def test_realization_universe_rejects_duplicate_or_noncanonical_slots() -> None:
    """Break caught: duplicate/out-of-order slots produce an ambiguous plan-set identity."""
    from latency_meta_mdp.expert_realization.contracts import (
        build_realization_universe_identity,
    )

    for slots in ((0, 0, 1), (1, 0), (-1, 0)):
        with pytest.raises(ValueError, match="slots"):
            build_realization_universe_identity(
                request_sha256=SHA_D,
                task_instance_id=_task(),
                realization_slots=slots,
            )


def test_realization_universe_mapping_round_trip_rejects_false_hash() -> None:
    """Break caught: a serialized universe digest does not bind its request/task/slot payload."""
    from latency_meta_mdp.expert_realization.contracts import (
        RealizationUniverseIdentity,
        build_realization_universe_identity,
    )

    universe = build_realization_universe_identity(
        request_sha256=SHA_D,
        task_instance_id=_task(),
        realization_slots=tuple(range(4)),
    )
    assert RealizationUniverseIdentity.from_mapping(universe.to_mapping()) == universe
    with pytest.raises(ValueError, match="universe_sha256"):
        RealizationUniverseIdentity.from_mapping(
            {**universe.to_mapping(), "universe_sha256": SHA_E}
        )


def test_realization_paths_round_trip_beyond_three_digits_without_aliases() -> None:
    """Break caught: a valid variable slot cannot be represented by canonical artifact paths."""
    row = _row(1000)
    assert row.strategy.path == "strategies/realization-1000.json"
    assert row.planner_candidates.path == "planner_candidates/realization-1000.jsonl"
    assert row.selected_reference is not None
    assert row.selected_reference.path == "selected_references/realization-1000.npz"
