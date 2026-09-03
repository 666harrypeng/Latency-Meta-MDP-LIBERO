from __future__ import annotations

from pathlib import Path

import pytest


def _universe():
    from latency_meta_mdp.expert_realization.config import FormalCorpusConfig
    from latency_meta_mdp.expert_realization.contracts import build_formal_request_universe

    config = FormalCorpusConfig(
        schema_version=2,
        corpus_id="formal-source-quota-workspace-test",
        logical_task_index_start=0,
        task_instance_count=3,
        levels=(1, 2, 3),
        realizations_per_task=4,
        families=(
            "canonical_direct",
            "early_high_arc",
            "lateral_arc",
            "time_shifted_smooth",
        ),
        family_allocation="iid_uniform_seeded",
        reserve_task_instance_count=3,
        require_complete_realization_block=True,
        group_unit="logical_master_task_index",
    )
    return build_formal_request_universe(
        config,
        corpus_config_sha256="a" * 64,
        structured_expert_config_sha256="b" * 64,
    )


def _identity(*, source_sha: str = "d" * 64):
    return {
        "schema_version": 2,
        "format_id": "formal_source_success_quota_identity_v1",
        "source_config_sha256": source_sha,
        "execution_config_sha256": "e" * 64,
        "qualification_gate_sha256": "f" * 64,
        "planner_environment_sha256": "0" * 64,
        "implementation_sha256": "1" * 64,
    }


def _workspace(tmp_path: Path):
    from latency_meta_mdp.expert_realization.source_corpus.workspace import CollectionWorkspace

    return CollectionWorkspace.create_formal(
        tmp_path / "work",
        _universe(),
        collection_identity=_identity(),
    )


def _draw(universe, *, level: int, index: int):
    from latency_meta_mdp.expert_realization.contracts import (
        TaskInstanceId,
        build_formal_realization_draw_request,
    )

    task = TaskInstanceId(
        level=level,
        task_instance_seed=universe.primary_tasks[0].master_task_seed,
        motion_profile_sha256=f"{level}" * 64,
        initial_state_sha256=f"{level + 3}" * 64,
    )
    return build_formal_realization_draw_request(universe, task, index)


def test_formal_workspace_predeclares_master_and_level_quotas_only(tmp_path: Path) -> None:
    """Break caught: four fixed child draws are materialized before any semantic sampling."""
    workspace = _workspace(tmp_path)

    assert len(workspace.formal_block_statuses()) == 6
    levels = workspace.level_quota_statuses()
    assert len(levels) == 18
    assert {(row.status, row.next_draw_index, row.accepted_count) for row in levels} == {
        ("pending", 0, 0)
    }
    assert workspace.draw_statuses() == ()


def test_failed_draw_advances_without_erasing_prior_acceptance(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    universe = workspace.request
    workspace.begin_formal_block(0)

    draw0 = workspace.begin_next_draw(
        logical_master_task_index=0,
        request=_draw(universe, level=1, index=0),
    )
    workspace.record_draw_failure(draw0, failure_class="task_failure", reason="missed")
    level = workspace.level_quota_status(0, 1)
    assert (level.next_draw_index, level.accepted_count) == (1, 0)

    draw1 = workspace.begin_next_draw(
        logical_master_task_index=0,
        request=_draw(universe, level=1, index=1),
    )
    workspace.record_draw_plan_qualified(
        draw1,
        selected_candidate_fingerprint="a" * 64,
        plan_path="payloads/plans/L1/task-0/draw-1",
    )
    workspace.record_draw_running(draw1)
    slot = workspace.record_draw_accepted(
        draw1,
        payload_path="payloads/successful/L1/task-0/draw-1",
        terminal_reason="lift_succeeded",
    )
    assert slot == 0
    level = workspace.level_quota_status(0, 1)
    assert (level.next_draw_index, level.accepted_count) == (2, 1)
    assert [row.realization_draw_index for row in workspace.accepted_draws(0, 1)] == [1]


def test_noncontiguous_draws_receive_dense_immutable_slots(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    universe = workspace.request
    workspace.begin_formal_block(0)

    accepted = []
    for index in range(8):
        draw = workspace.begin_next_draw(
            logical_master_task_index=0,
            request=_draw(universe, level=1, index=index),
        )
        if index not in {0, 2, 5, 7}:
            workspace.record_draw_failure(
                draw,
                failure_class="diversity_rejection",
                reason="near duplicate",
            )
            continue
        workspace.record_draw_plan_qualified(
            draw,
            selected_candidate_fingerprint=f"{index:x}" * 64,
            plan_path=f"payloads/plans/L1/task-0/draw-{index}",
        )
        workspace.record_draw_running(draw)
        accepted.append(
            workspace.record_draw_accepted(
                draw,
                payload_path=f"payloads/successful/L1/task-0/draw-{index}",
                terminal_reason="lift_succeeded",
            )
        )

    assert accepted == [0, 1, 2, 3]
    rows = workspace.accepted_draws(0, 1)
    assert [(row.realization_draw_index, row.accepted_slot) for row in rows] == [
        (0, 0),
        (2, 1),
        (5, 2),
        (7, 3),
    ]
    assert workspace.level_quota_status(0, 1).status == "complete"
    with pytest.raises(ValueError, match="complete"):
        workspace.begin_next_draw(
            logical_master_task_index=0,
            request=_draw(universe, level=1, index=8),
        )


def test_infrastructure_retry_preserves_draw_identity(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    workspace.begin_formal_block(0)
    request = _draw(workspace.request, level=2, index=0)
    draw = workspace.begin_next_draw(logical_master_task_index=0, request=request)

    retried = workspace.record_draw_infrastructure_retry(draw, reason="worker crash")

    assert retried.realization_draw_index == draw.realization_draw_index
    assert retried.realization_seed == draw.realization_seed
    assert retried.family == draw.family
    assert retried.attempt_index == 1
    assert workspace.level_quota_status(0, 2).next_draw_index == 0


def test_master_admission_requires_three_complete_level_quotas(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    workspace.begin_formal_block(0)
    with pytest.raises(ValueError, match="three complete level quotas"):
        workspace.admit_formal_block(0)

    for level in (1, 2, 3):
        for index in range(4):
            draw = workspace.begin_next_draw(
                logical_master_task_index=0,
                request=_draw(workspace.request, level=level, index=index),
            )
            workspace.record_draw_plan_qualified(
                draw,
                selected_candidate_fingerprint=f"{level}{index}".ljust(64, "a"),
                plan_path=f"payloads/plans/L{level}/task-0/draw-{index}",
            )
            workspace.record_draw_running(draw)
            workspace.record_draw_accepted(
                draw,
                payload_path=f"payloads/successful/L{level}/task-0/draw-{index}",
                terminal_reason="lift_succeeded",
            )
    workspace.admit_formal_block(0)
    assert workspace.formal_block_status(0).status == "admitted"


def test_formal_resume_requires_exact_collection_identity_and_state(tmp_path: Path) -> None:
    from latency_meta_mdp.expert_realization.source_corpus.workspace import CollectionWorkspace

    root = tmp_path / "work"
    workspace = CollectionWorkspace.create_formal(
        root,
        _universe(),
        collection_identity=_identity(),
    )
    workspace.begin_formal_block(0)
    draw = workspace.begin_next_draw(
        logical_master_task_index=0,
        request=_draw(workspace.request, level=3, index=0),
    )
    workspace.record_draw_failure(draw, failure_class="planner_failure", reason="infeasible")

    resumed = CollectionWorkspace.resume_formal(
        root,
        _universe(),
        collection_identity=_identity(),
    )
    assert resumed.level_quota_status(0, 3).next_draw_index == 1
    with pytest.raises(ValueError, match="collection identity"):
        CollectionWorkspace.resume_formal(
            root,
            _universe(),
            collection_identity=_identity(source_sha="2" * 64),
        )
