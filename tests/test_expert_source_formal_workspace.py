from __future__ import annotations

from pathlib import Path

import pytest


def _universe():
    from latency_meta_mdp.expert_realization.config import FormalCorpusConfig
    from latency_meta_mdp.expert_realization.contracts import build_formal_request_universe

    config = FormalCorpusConfig(
        schema_version=1,
        corpus_id="formal-source-workspace-test",
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
        split_unit="master_task_index",
    )
    return build_formal_request_universe(
        config,
        corpus_config_sha256="a" * 64,
        structured_expert_config_sha256="b" * 64,
    )


def _identity(*, split_sha: str = "c" * 64):
    return {
        "schema_version": 1,
        "format_id": "formal_source_collection_identity_v1",
        "source_config_sha256": "d" * 64,
        "split_plan_sha256": split_sha,
        "execution_config_sha256": "e" * 64,
        "qualification_gate_sha256": "f" * 64,
        "planner_environment_sha256": "0" * 64,
        "implementation_sha256": "1" * 64,
    }


def _workspace(tmp_path: Path):
    from latency_meta_mdp.expert_realization.source_corpus.workspace import (
        CollectionWorkspace,
    )

    return CollectionWorkspace.create_formal(
        tmp_path / "work",
        _universe(),
        collection_identity=_identity(),
    )


def test_formal_workspace_materializes_six_blocks_and_72_plan_slots(tmp_path: Path) -> None:
    """Break caught: reserve blocks or cross-level plan identities are created lazily."""
    workspace = _workspace(tmp_path)

    blocks = workspace.formal_block_statuses()
    plans = workspace.formal_plan_statuses()
    assert len(blocks) == 6
    assert len(plans) == 72
    assert {row.status for row in blocks} == {"pending"}
    assert {row.status for row in plans} == {"pending"}
    expected = {
        (task, level, realization)
        for task in range(6)
        for level in (1, 2, 3)
        for realization in range(4)
    }
    assert {
        (row.logical_master_task_index, row.level, row.realization_index)
        for row in plans
    } == expected


def test_candidate_failure_advances_but_infrastructure_retry_preserves_index(
    tmp_path: Path,
) -> None:
    """Break caught: semantic and infrastructure planner failures share one retry transition."""
    workspace = _workspace(tmp_path)
    workspace.begin_formal_block(0)
    workspace.record_formal_candidate_failure(
        logical_master_task_index=0,
        level=1,
        realization_index=0,
        candidate_index=0,
        reason="infeasible",
    )
    advanced = workspace.formal_plan_status(0, 1, 0)
    assert advanced.current_candidate_index == 1
    assert advanced.infrastructure_retry_count == 0
    assert advanced.semantic_failures == ("candidate=0: infeasible",)

    workspace.record_formal_infrastructure_retry(
        logical_master_task_index=0,
        level=1,
        realization_index=0,
        candidate_index=1,
        reason="worker crash",
    )
    retried = workspace.formal_plan_status(0, 1, 0)
    assert retried.current_candidate_index == 1
    assert retried.infrastructure_retry_count == 1


def test_candidate_seven_exhaustion_rejects_block_and_forbids_index_eight(
    tmp_path: Path,
) -> None:
    """Break caught: an impossible realization creates an unbounded candidate sequence."""
    workspace = _workspace(tmp_path)
    workspace.begin_formal_block(0)
    for candidate_index in range(8):
        workspace.record_formal_candidate_failure(
            logical_master_task_index=0,
            level=1,
            realization_index=0,
            candidate_index=candidate_index,
            reason="infeasible",
        )
    plan = workspace.formal_plan_status(0, 1, 0)
    assert plan.status == "exhausted"
    assert workspace.formal_block_status(0).status == "rejected"
    with pytest.raises(ValueError, match="candidate index"):
        workspace.record_formal_candidate_failure(
            logical_master_task_index=0,
            level=1,
            realization_index=0,
            candidate_index=8,
            reason="invalid",
        )
    with pytest.raises(ValueError, match="planning formal block"):
        workspace.record_formal_infrastructure_retry(
            logical_master_task_index=0,
            level=1,
            realization_index=1,
            candidate_index=0,
            reason="late retry",
        )
    with pytest.raises(ValueError, match="planning formal block"):
        workspace.record_formal_plan_success(
            logical_master_task_index=0,
            level=1,
            realization_index=1,
            candidate_index=0,
            selected_candidate_fingerprint="a" * 64,
        )


def _qualify_all_plans(workspace, logical: int) -> None:
    for level in (1, 2, 3):
        for realization in range(4):
            if workspace.formal_plan_status(logical, level, realization).status == "qualified":
                continue
            workspace.record_formal_plan_success(
                logical_master_task_index=logical,
                level=level,
                realization_index=realization,
                candidate_index=0,
                selected_candidate_fingerprint=f"{level}{realization}".ljust(64, "a"),
            )


def test_plan_ready_and_admitted_require_all_twelve_children(tmp_path: Path) -> None:
    """Break caught: a partial level or realization is admitted as a paired master block."""
    from latency_meta_mdp.expert_realization.source_corpus.workspace import (
        RealizationRunStatus,
    )

    workspace = _workspace(tmp_path)
    workspace.begin_formal_block(0)
    workspace.record_formal_plan_success(
        logical_master_task_index=0,
        level=1,
        realization_index=0,
        candidate_index=0,
        selected_candidate_fingerprint="a" * 64,
    )
    with pytest.raises(ValueError, match="twelve"):
        workspace.mark_formal_block_plan_ready(0)
    _qualify_all_plans(workspace, 0)
    workspace.mark_formal_block_plan_ready(0)
    workspace.begin_formal_block_execution(0)
    with pytest.raises(ValueError, match="twelve"):
        workspace.admit_formal_block(0)

    for level in (1, 2, 3):
        for realization in range(4):
            identity = dict(
                logical_master_task_index=0,
                level=level,
                realization_index=realization,
            )
            workspace.record_formal_execution_status(
                RealizationRunStatus(**identity, status="running", attempt_index=0)
            )
            workspace.record_formal_execution_status(
                RealizationRunStatus(
                    **identity,
                    status="success",
                    attempt_index=0,
                    payload_path=(
                        f"payloads/successful/L{level}/task-0/r-{realization}"
                    ),
                    terminal_reason="lift_succeeded",
                )
            )
    workspace.admit_formal_block(0)
    assert workspace.formal_block_status(0).status == "admitted"


def test_semantic_execution_failure_rejects_the_parent_block(tmp_path: Path) -> None:
    """Break caught: scheduler continues a block that can no longer be paired and admitted."""
    from latency_meta_mdp.expert_realization.source_corpus.workspace import (
        RealizationRunStatus,
    )

    workspace = _workspace(tmp_path)
    workspace.begin_formal_block(0)
    _qualify_all_plans(workspace, 0)
    workspace.mark_formal_block_plan_ready(0)
    workspace.begin_formal_block_execution(0)
    identity = dict(logical_master_task_index=0, level=1, realization_index=0)
    workspace.record_formal_execution_status(
        RealizationRunStatus(**identity, status="running", attempt_index=0)
    )
    workspace.record_formal_execution_status(
        RealizationRunStatus(
            **identity,
            status="task_failure",
            attempt_index=0,
            terminal_reason="grasp failed",
        )
    )
    assert workspace.formal_block_status(0).status == "rejected"


def test_formal_resume_requires_exact_collection_identity(tmp_path: Path) -> None:
    """Break caught: a split/gate/implementation change resumes old successful payloads."""
    from latency_meta_mdp.expert_realization.source_corpus.workspace import (
        CollectionWorkspace,
    )

    root = tmp_path / "work"
    CollectionWorkspace.create_formal(
        root, _universe(), collection_identity=_identity()
    )
    resumed = CollectionWorkspace.resume_formal(
        root, _universe(), collection_identity=_identity()
    )
    assert len(resumed.formal_block_statuses()) == 6
    with pytest.raises(ValueError, match="collection identity"):
        CollectionWorkspace.resume_formal(
            root,
            _universe(),
            collection_identity=_identity(split_sha="2" * 64),
        )
