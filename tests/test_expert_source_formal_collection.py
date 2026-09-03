from __future__ import annotations

from pathlib import Path

import pytest


def _ref(logical: int, level: int, draw: int, slot: int):
    from latency_meta_mdp.expert_realization.source_corpus.formal_collection import (
        SourceSuccessPayloadRef,
    )

    return SourceSuccessPayloadRef(
        logical_master_task_index=logical,
        level=level,
        realization_draw_index=draw,
        accepted_slot=slot,
        payload_path=Path(
            f"/workspace/payloads/successful/L{level}/task-{logical:06d}/draw-{draw:04d}"
        ),
    )


def test_level_quota_replaces_failed_draws_without_losing_successes() -> None:
    from latency_meta_mdp.expert_realization.source_corpus.formal_collection import (
        DrawRejected,
        fill_level_success_quota,
    )

    outcomes = {
        0: "task_failure",
        1: "success",
        2: "diversity_rejection",
        3: "success",
        4: "success",
        5: "success",
    }
    rejected = []

    def plan(draw: int, _accepted):
        if outcomes[draw] == "diversity_rejection":
            raise DrawRejected("diversity_rejection", "near duplicate")
        return draw

    def execute(draw: int, slot: int):
        if outcomes[draw] == "task_failure":
            raise DrawRejected("task_failure", "missed grasp")
        return _ref(0, 1, draw, slot)

    completed = fill_level_success_quota(
        logical_master_task_index=0,
        level=1,
        realization_quota=4,
        maximum_draws=16,
        start_draw_index=0,
        existing_successes=(),
        plan_draw=plan,
        execute_draw=execute,
        on_rejected=lambda draw, kind, reason: rejected.append((draw, kind, reason)),
    )

    assert [(row.realization_draw_index, row.accepted_slot) for row in completed.successes] == [
        (1, 0),
        (3, 1),
        (4, 2),
        (5, 3),
    ]
    assert [row[:2] for row in rejected] == [
        (0, "task_failure"),
        (2, "diversity_rejection"),
    ]
    assert completed.next_draw_index == 6


def test_level_quota_resume_preserves_dense_slots() -> None:
    from latency_meta_mdp.expert_realization.source_corpus.formal_collection import (
        fill_level_success_quota,
    )

    existing = (_ref(0, 2, 1, 0), _ref(0, 2, 3, 1))
    planned = []
    completed = fill_level_success_quota(
        logical_master_task_index=0,
        level=2,
        realization_quota=4,
        maximum_draws=16,
        start_draw_index=4,
        existing_successes=existing,
        plan_draw=lambda draw, _accepted: planned.append(draw) or draw,
        execute_draw=lambda draw, slot: _ref(0, 2, draw, slot),
        on_rejected=lambda _draw, _kind, _reason: None,
    )

    assert planned == [4, 5]
    assert [(row.realization_draw_index, row.accepted_slot) for row in completed.successes] == [
        (1, 0),
        (3, 1),
        (4, 2),
        (5, 3),
    ]


def test_level_quota_exhaustion_is_typed_and_never_returns_partial() -> None:
    from latency_meta_mdp.expert_realization.source_corpus.formal_collection import (
        DrawRejected,
        LevelQuotaExhausted,
        fill_level_success_quota,
    )

    def fail(draw: int, _accepted):
        raise DrawRejected("planner_failure", f"draw {draw} infeasible")

    with pytest.raises(LevelQuotaExhausted, match="L3.*4 successes.*5 draws"):
        fill_level_success_quota(
            logical_master_task_index=7,
            level=3,
            realization_quota=4,
            maximum_draws=5,
            start_draw_index=0,
            existing_successes=(),
            plan_draw=fail,
            execute_draw=lambda _draw, _slot: None,
            on_rejected=lambda _draw, _kind, _reason: None,
        )


def test_completed_master_requires_dense_slots_per_level() -> None:
    from latency_meta_mdp.expert_realization.source_corpus.formal_collection import (
        CompletedMasterBlock,
    )

    successes = tuple(
        _ref(0, level, draw=slot + level, slot=slot)
        for level in (1, 2, 3)
        for slot in range(4)
    )
    block = CompletedMasterBlock(0, successes)
    assert len(block.successes) == 12

    with pytest.raises(ValueError, match="complete accepted slots"):
        CompletedMasterBlock(0, successes[:-1])


def test_master_scheduler_uses_reserve_only_after_level_quota_exhaustion(
    tmp_path: Path,
) -> None:
    from latency_meta_mdp.expert_realization.source_corpus.formal_collection import (
        CompletedMasterBlock,
        LevelQuotaExhausted,
        schedule_success_quota_master_blocks,
    )

    attempted = []

    def collect(logical: int):
        attempted.append(logical)
        if logical == 0:
            raise LevelQuotaExhausted("master 0 L2 could not fill quota")
        return CompletedMasterBlock(
            logical,
            tuple(
                _ref(logical, level, draw=slot, slot=slot)
                for level in (1, 2, 3)
                for slot in range(4)
            ),
        )

    published = []
    result = schedule_success_quota_master_blocks(
        master_task_indices=(0, 1, 2),
        target_block_count=2,
        collect_block=collect,
        publish_blocks=lambda blocks: published.extend(blocks) or (tmp_path / "manifest.json"),
    )

    assert result == tmp_path / "manifest.json"
    assert attempted == [0, 1, 2]
    assert [row.logical_master_task_index for row in published] == [1, 2]


def test_reserve_exhaustion_never_publishes() -> None:
    from latency_meta_mdp.expert_realization.source_corpus.formal_collection import (
        LevelQuotaExhausted,
        schedule_success_quota_master_blocks,
    )

    published = []
    with pytest.raises(RuntimeError, match="reserve"):
        schedule_success_quota_master_blocks(
            master_task_indices=(0, 1),
            target_block_count=1,
            collect_block=lambda _logical: (_ for _ in ()).throw(
                LevelQuotaExhausted("quota exhausted")
            ),
            publish_blocks=lambda blocks: published.append(blocks),
        )
    assert published == []
