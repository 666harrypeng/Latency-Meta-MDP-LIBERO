from __future__ import annotations

from pathlib import Path

import pytest


def _universe(*, corpus_id: str = "panda-ball-source-workspace-test"):
    from latency_meta_mdp.expert_realization.config import FormalCorpusConfig
    from latency_meta_mdp.expert_realization.contracts import build_formal_request_universe

    config = FormalCorpusConfig(
        schema_version=1,
        corpus_id=corpus_id,
        logical_task_index_start=0,
        task_instance_count=1,
        levels=(1, 2, 3),
        realizations_per_task=2,
        families=(
            "canonical_direct",
            "early_high_arc",
            "lateral_arc",
            "time_shifted_smooth",
        ),
        family_allocation="iid_uniform_seeded",
        reserve_task_instance_count=1,
        require_complete_realization_block=True,
        split_unit="master_task_index",
    )
    return build_formal_request_universe(
        config,
        corpus_config_sha256="a" * 64,
        structured_expert_config_sha256="b" * 64,
    )


def _advance_success(workspace, *, logical_index: int, level: int, realization: int) -> None:
    from latency_meta_mdp.expert_realization.source_corpus.workspace import (
        RealizationRunStatus,
    )

    identity = {
        "logical_master_task_index": logical_index,
        "level": level,
        "realization_index": realization,
    }
    workspace.record_status(
        RealizationRunStatus(**identity, status="planned", attempt_index=0)
    )
    workspace.record_status(
        RealizationRunStatus(**identity, status="running", attempt_index=0)
    )
    workspace.record_status(
        RealizationRunStatus(
            **identity,
            status="success",
            attempt_index=0,
            payload_path=f"payloads/successful/L{level}/task-{logical_index}/r-{realization}",
            terminal_reason="lift_succeeded",
        )
    )


def test_workspace_materializes_the_complete_primary_and_reserve_universe(tmp_path: Path) -> None:
    """Break caught: scheduling or a failure creates semantic identities lazily."""
    from latency_meta_mdp.expert_realization.source_corpus.workspace import CollectionWorkspace

    universe = _universe()
    workspace = CollectionWorkspace.create(tmp_path / "work", universe)

    rows = workspace.status_rows()
    assert len(rows) == 12
    assert {(row.logical_master_task_index, row.level) for row in rows} == {
        (task, level) for task in (0, 1) for level in (1, 2, 3)
    }
    assert {row.status for row in rows} == {"requested"}
    assert (tmp_path / "work" / "request.json").is_file()
    assert (tmp_path / "work" / "run_state.sqlite").is_file()
    assert workspace.payload_root == tmp_path / "work" / "payloads"
    assert workspace.payload_root.is_dir()


def test_workspace_resume_requires_the_exact_same_request(tmp_path: Path) -> None:
    """Break caught: resume silently reuses statuses under a different corpus identity."""
    from latency_meta_mdp.expert_realization.source_corpus.workspace import CollectionWorkspace

    root = tmp_path / "work"
    first = _universe()
    CollectionWorkspace.create(root, first)
    (root / "payloads" / "resume-marker").write_text("owned work")
    resumed = CollectionWorkspace.resume(root, first)
    assert len(resumed.status_rows()) == 12

    with pytest.raises(ValueError, match="request identity"):
        CollectionWorkspace.resume(root, _universe(corpus_id="different-corpus"))


def test_workspace_enforces_status_transitions_and_same_identity_retry(tmp_path: Path) -> None:
    """Break caught: semantic failure is retried or retry attempt order changes."""
    from latency_meta_mdp.expert_realization.source_corpus.workspace import (
        CollectionWorkspace,
        RealizationRunStatus,
    )

    workspace = CollectionWorkspace.create(tmp_path / "work", _universe())
    identity = dict(logical_master_task_index=0, level=1, realization_index=0)
    with pytest.raises(ValueError, match="invalid status transition"):
        workspace.record_status(
            RealizationRunStatus(
                **identity,
                status="success",
                attempt_index=0,
                payload_path="payloads/successful/L1/task-0/r-0",
                terminal_reason="lift_succeeded",
            )
        )
    workspace.record_status(
        RealizationRunStatus(**identity, status="planned", attempt_index=0)
    )
    workspace.record_status(
        RealizationRunStatus(**identity, status="running", attempt_index=0)
    )
    workspace.record_status(
        RealizationRunStatus(
            **identity,
            status="infrastructure_failure",
            attempt_index=0,
            terminal_reason="renderer process exited",
        )
    )
    with pytest.raises(ValueError, match="attempt_index"):
        workspace.record_status(
            RealizationRunStatus(**identity, status="running", attempt_index=0)
        )
    workspace.record_status(
        RealizationRunStatus(**identity, status="running", attempt_index=1)
    )
    workspace.record_status(
        RealizationRunStatus(
            **identity,
            status="task_failure",
            attempt_index=1,
            terminal_reason="grasp deadline missed",
        )
    )
    with pytest.raises(ValueError, match="terminal semantic status"):
        workspace.record_status(
            RealizationRunStatus(**identity, status="running", attempt_index=2)
        )


def test_workspace_keeps_only_lightweight_failure_state(tmp_path: Path) -> None:
    """Break caught: failed trajectory payload becomes a durable workspace record."""
    from latency_meta_mdp.expert_realization.source_corpus.workspace import (
        CollectionWorkspace,
        RealizationRunStatus,
    )

    workspace = CollectionWorkspace.create(tmp_path / "work", _universe())
    identity = dict(logical_master_task_index=0, level=1, realization_index=0)
    workspace.record_status(
        RealizationRunStatus(
            **identity,
            status="planner_failure",
            attempt_index=0,
            terminal_reason="no feasible candidate",
        )
    )
    row = workspace.status_for(**identity)

    assert row.status == "planner_failure"
    assert row.payload_path is None
    assert row.terminal_reason == "no feasible candidate"
    assert {path.name for path in (tmp_path / "work").iterdir()} == {
        "request.json",
        "run_state.sqlite",
        "payloads",
    }
    assert not list((tmp_path / "work" / "payloads").iterdir())


def test_complete_blocks_require_every_level_and_realization_success(tmp_path: Path) -> None:
    """Break caught: a partially successful task leaks into formal source publication."""
    from latency_meta_mdp.expert_realization.source_corpus.workspace import CollectionWorkspace

    workspace = CollectionWorkspace.create(tmp_path / "work", _universe())
    for level in (1, 2, 3):
        for realization in (0, 1):
            if (level, realization) != (3, 1):
                _advance_success(
                    workspace,
                    logical_index=0,
                    level=level,
                    realization=realization,
                )
    assert workspace.admitted_complete_blocks() == ()
    _advance_success(workspace, logical_index=0, level=3, realization=1)
    assert workspace.admitted_complete_blocks() == (0,)


def test_complete_block_result_is_independent_of_worker_completion_order(tmp_path: Path) -> None:
    """Break caught: parallel completion order changes which semantic task is admitted."""
    from latency_meta_mdp.expert_realization.source_corpus.workspace import CollectionWorkspace

    forward = CollectionWorkspace.create(tmp_path / "forward", _universe())
    reverse = CollectionWorkspace.create(tmp_path / "reverse", _universe())
    identities = [
        (level, realization) for level in (1, 2, 3) for realization in (0, 1)
    ]
    for level, realization in identities:
        _advance_success(forward, logical_index=0, level=level, realization=realization)
    for level, realization in reversed(identities):
        _advance_success(reverse, logical_index=0, level=level, realization=realization)

    assert forward.admitted_complete_blocks() == reverse.admitted_complete_blocks() == (0,)
