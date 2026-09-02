from __future__ import annotations

from pathlib import Path

import pytest


def _twelve(logical: int) -> tuple[str, ...]:
    return tuple(
        f"task={logical}/L{level}/r{realization}"
        for level in (1, 2, 3)
        for realization in range(4)
    )


def test_scheduler_requires_complete_planning_before_ordered_execution(
    tmp_path: Path,
) -> None:
    """Break caught: a rollout starts before all twelve paired plans are frozen."""
    from latency_meta_mdp.expert_realization.source_corpus.formal_collection import (
        CompletedMasterBlock,
        PlannedMasterBlock,
        schedule_paired_master_blocks,
    )

    events = []

    def plan(logical: int):
        identities = _twelve(logical)
        events.extend(("plan", identity) for identity in identities)
        return PlannedMasterBlock(logical, identities)

    def execute(block):
        assert len([event for event in events if event[0] == "plan"]) % 12 == 0
        events.extend(("execute", identity) for identity in block.plan_identities)
        return CompletedMasterBlock(block.logical_master_task_index, block.plan_identities)

    published = []

    def publish(blocks):
        published.extend(blocks)
        return tmp_path / "manifest.json"

    result = schedule_paired_master_blocks(
        master_task_indices=(0, 1, 2),
        target_block_count=3,
        plan_block=plan,
        execute_block=execute,
        publish_blocks=publish,
    )

    assert result == tmp_path / "manifest.json"
    assert [block.logical_master_task_index for block in published] == [0, 1, 2]
    assert [event[1] for event in events if event[0] == "execute"] == (
        list(_twelve(0)) + list(_twelve(1)) + list(_twelve(2))
    )


def test_failed_block_is_discarded_and_next_reserve_replaces_it(tmp_path: Path) -> None:
    """Break caught: partial successes from a rejected block leak into final publication."""
    from latency_meta_mdp.expert_realization.source_corpus.formal_collection import (
        BlockExecutionFailure,
        CompletedMasterBlock,
        PlannedMasterBlock,
        schedule_paired_master_blocks,
    )

    executed = []

    def plan(logical: int):
        return PlannedMasterBlock(logical, _twelve(logical))

    def execute(block):
        executed.append(block.logical_master_task_index)
        if block.logical_master_task_index == 0:
            raise BlockExecutionFailure(
                "grasp failed", partial_successes=block.plan_identities[:2]
            )
        return CompletedMasterBlock(block.logical_master_task_index, block.plan_identities)

    published = []
    result = schedule_paired_master_blocks(
        master_task_indices=(0, 1, 2, 3),
        target_block_count=3,
        plan_block=plan,
        execute_block=execute,
        publish_blocks=lambda blocks: published.extend(blocks) or (tmp_path / "manifest.json"),
    )

    assert result == tmp_path / "manifest.json"
    assert executed == [0, 1, 2, 3]
    assert [block.logical_master_task_index for block in published] == [1, 2, 3]
    assert all("task=0" not in item for block in published for item in block.successes)


def test_reserve_replacement_preserves_primary_split_quotas(tmp_path: Path) -> None:
    """Break caught: a failed validation primary is replaced by a train reserve."""
    from latency_meta_mdp.expert_realization.source_corpus.formal_collection import (
        BlockExecutionFailure,
        CompletedMasterBlock,
        PlannedMasterBlock,
        schedule_paired_master_blocks,
    )

    split_by_master = {
        0: "train",
        1: "train",
        2: "validation",
        3: "train",
        4: "train",
        5: "validation",
    }

    def execute(block):
        if block.logical_master_task_index == 2:
            raise BlockExecutionFailure("validation grasp failed", partial_successes=())
        return CompletedMasterBlock(block.logical_master_task_index, block.plan_identities)

    published = []
    result = schedule_paired_master_blocks(
        master_task_indices=(0, 1, 2, 3, 4, 5),
        target_block_count=3,
        target_block_counts={"train": 2, "validation": 1},
        split_for=split_by_master.__getitem__,
        plan_block=lambda logical: PlannedMasterBlock(logical, _twelve(logical)),
        execute_block=execute,
        publish_blocks=lambda blocks: published.extend(blocks) or (tmp_path / "manifest.json"),
    )

    assert result == tmp_path / "manifest.json"
    assert [block.logical_master_task_index for block in published] == [0, 1, 5]
    assert [split_by_master[block.logical_master_task_index] for block in published] == [
        "train",
        "train",
        "validation",
    ]


def test_split_specific_reserve_exhaustion_never_publishes() -> None:
    """Break caught: total count passes even though one required split is missing."""
    from latency_meta_mdp.expert_realization.source_corpus.formal_collection import (
        BlockPlanningFailure,
        CompletedMasterBlock,
        PlannedMasterBlock,
        schedule_paired_master_blocks,
    )

    split_by_master = {0: "train", 1: "validation", 2: "train"}
    published = []

    def plan(logical: int):
        if logical == 1:
            raise BlockPlanningFailure("validation diversity failed")
        return PlannedMasterBlock(logical, _twelve(logical))

    with pytest.raises(RuntimeError, match="reserve"):
        schedule_paired_master_blocks(
            master_task_indices=(0, 1, 2),
            target_block_count=2,
            target_block_counts={"train": 1, "validation": 1},
            split_for=split_by_master.__getitem__,
            plan_block=plan,
            execute_block=lambda block: CompletedMasterBlock(
                block.logical_master_task_index, block.plan_identities
            ),
            publish_blocks=lambda blocks: published.append(blocks),
        )
    assert published == []


def test_planning_failure_produces_no_rollout_for_rejected_block(tmp_path: Path) -> None:
    """Break caught: execution begins after a task-level planning group has failed."""
    from latency_meta_mdp.expert_realization.source_corpus.formal_collection import (
        BlockPlanningFailure,
        CompletedMasterBlock,
        PlannedMasterBlock,
        schedule_paired_master_blocks,
    )

    executed = []

    def plan(logical: int):
        if logical == 0:
            raise BlockPlanningFailure("no four-plan diversity")
        return PlannedMasterBlock(logical, _twelve(logical))

    def execute(block):
        executed.append(block.logical_master_task_index)
        return CompletedMasterBlock(block.logical_master_task_index, block.plan_identities)

    schedule_paired_master_blocks(
        master_task_indices=(0, 1),
        target_block_count=1,
        plan_block=plan,
        execute_block=execute,
        publish_blocks=lambda _blocks: tmp_path / "manifest.json",
    )
    assert executed == [1]


def test_reserve_exhaustion_never_invokes_publication() -> None:
    """Break caught: an incomplete pilot is published after reserve exhaustion."""
    from latency_meta_mdp.expert_realization.source_corpus.formal_collection import (
        BlockPlanningFailure,
        schedule_paired_master_blocks,
    )

    published = []
    with pytest.raises(RuntimeError, match="reserve"):
        schedule_paired_master_blocks(
            master_task_indices=(0, 1),
            target_block_count=1,
            plan_block=lambda _logical: (_ for _ in ()).throw(
                BlockPlanningFailure("infeasible")
            ),
            execute_block=lambda _block: None,
            publish_blocks=lambda blocks: published.append(blocks),
        )
    assert published == []


def test_scheduler_rejects_malformed_completed_block() -> None:
    """Break caught: fewer than twelve success payloads are treated as a complete block."""
    from latency_meta_mdp.expert_realization.source_corpus.formal_collection import (
        CompletedMasterBlock,
        PlannedMasterBlock,
        schedule_paired_master_blocks,
    )

    with pytest.raises(ValueError, match="twelve"):
        schedule_paired_master_blocks(
            master_task_indices=(0,),
            target_block_count=1,
            plan_block=lambda logical: PlannedMasterBlock(logical, _twelve(logical)),
            execute_block=lambda block: CompletedMasterBlock(
                block.logical_master_task_index, block.plan_identities[:-1]
            ),
            publish_blocks=lambda _blocks: Path("manifest.json"),
        )
