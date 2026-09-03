from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from expert_realization_test_support import make_formal_source_episode


def _publication_fixture(*, boundary_count: int = 2):
    from latency_meta_mdp.expert_realization.config import FormalCorpusConfig
    from latency_meta_mdp.expert_realization.contracts import (
        ExpertRealizationId,
        TaskInstanceId,
        build_formal_realization_requests,
        build_formal_request_universe,
    )
    from latency_meta_mdp.expert_realization.source_corpus.config import (
        load_source_corpus_config,
    )
    from latency_meta_mdp.expert_realization.source_corpus.metadata import (
        SourceTaskMetadataEntry,
    )

    formal = FormalCorpusConfig(
        schema_version=2,
        corpus_id="panda-ball-structured-source-pilot",
        logical_task_index_start=0,
        task_instance_count=1,
        levels=(1,),
        realizations_per_task=1,
        families=(
            "canonical_direct",
            "early_high_arc",
            "lateral_arc",
            "time_shifted_smooth",
        ),
        family_allocation="iid_uniform_seeded",
        reserve_task_instance_count=1,
        require_complete_realization_block=True,
        group_unit="logical_master_task_index",
    )
    request = build_formal_request_universe(
        formal,
        corpus_config_sha256="e" * 64,
        structured_expert_config_sha256="f" * 64,
    )
    source_config = load_source_corpus_config(
        Path("configs/source_corpus/panda_ball_source_parquet.yaml")
    )
    motion = b'{"level":1,"profile":"constant_velocity"}\n'
    initial = b"exact-initial-state-npz"
    task_id = TaskInstanceId(
        level=1,
        task_instance_seed=request.primary_tasks[0].master_task_seed,
        motion_profile_sha256=hashlib.sha256(motion).hexdigest(),
        initial_state_sha256=hashlib.sha256(initial).hexdigest(),
    )
    source = make_formal_source_episode(
        camera_height=256,
        camera_width=256,
        boundary_count=boundary_count,
    )
    formal_realization = build_formal_realization_requests(request, task_id)[0]
    key = formal_realization.to_expert_realization_key()
    metadata = replace(
        source.metadata,
        corpus_id=formal.corpus_id,
        task_instance_id=task_id,
        expert_realization_id=ExpertRealizationId(
            key,
            source.metadata.expert_realization_id.task_instance_plan_set_sha256,
        ),
        strategy_family=formal_realization.assigned_family,
        formal_corpus_config_sha256=request.corpus_config_sha256,
        source_corpus_config_sha256=source_config.sha256,
    )
    episode = replace(
        source,
        metadata=metadata,
        transitions=tuple(
            replace(
                transition,
                expert_audit=replace(
                    transition.expert_audit,
                    expert_realization_id=metadata.expert_realization_id,
                ),
            )
            for transition in source.transitions
        ),
    )
    task = SourceTaskMetadataEntry(
        task_instance_id=task_id,
        corpus_id=formal.corpus_id,
        logical_master_task_index=0,
        instruction=metadata.instruction,
        motion_profile_json=motion.decode(),
        initial_state_npz=initial,
        admitted_realization_count=1,
    )
    return request, source_config, task, episode


def _admitted(episode):
    from latency_meta_mdp.expert_realization.source_corpus.collection import (
        AdmittedSourceEpisode,
    )

    return AdmittedSourceEpisode(
        episode=episode,
        strategy_parameters={"prediction_lead_seconds": 0.12},
        selected_planner_fingerprint="c" * 64,
        qualification={
            "eligible": True,
            "failures": [],
            "actual_rollout_safety": {"terminal_success": True},
        },
    )


def _summary():
    from latency_meta_mdp.expert_realization.source_corpus.collection import CollectionSummary

    return CollectionSummary(
        requested_realizations=2,
        planned_realizations=1,
        executed_attempts=1,
        successful_realizations=1,
        admitted_realizations=1,
        failures_by_class={},
    )


def test_source_publication_contains_only_success_data_and_central_metadata(
    tmp_path: Path,
) -> None:
    """Break caught: failed/workspace artifacts or per-episode JSON enter the source corpus."""
    from latency_meta_mdp.expert_realization.source_corpus.collection import publish_source_corpus

    request, config, task, episode = _publication_fixture()
    target = tmp_path / "formal_source_corpus"
    manifest_path = publish_source_corpus(
        target=target,
        request=request,
        source_config=config,
        task_entries=(task,),
        admitted_episodes=(_admitted(episode),),
        collection_summary=_summary(),
    )

    assert manifest_path == target / "manifest.json"
    files = {path.relative_to(target).as_posix() for path in target.rglob("*") if path.is_file()}
    assert files == {
        "README.md",
        "manifest.json",
        "schema.json",
        "provenance.json",
        "meta/task_instances.parquet",
        "meta/episodes.parquet",
        "meta/events.parquet",
        "meta/collection_summary.json",
        "data/level-1/shard-00000.parquet",
    }
    assert not any(
        token in path
        for path in files
        for token in ("failure", "attempt", "candidate", "workspace", "video")
    )
    manifest = json.loads(manifest_path.read_text())
    provenance = json.loads((target / "provenance.json").read_text())
    assert manifest["complete"] is True
    assert manifest["episode_count"] == 1
    assert manifest["master_task_count"] == 1
    assert manifest["level_task_instance_count"] == 1
    assert manifest["episodes_by_level"] == {"1": 1}
    assert set(manifest["artifacts"]) == files - {"manifest.json"}
    assert provenance["formal_request"]["request_sha256"] == request.request_sha256
    assert provenance["source_config"] == config.to_mapping()
    assert "split_plan" not in provenance
    assert pq.read_table(target / "meta/episodes.parquet")["episode_id"].to_pylist() == [
        episode.metadata.episode_id
    ]


def test_source_publication_rejects_incomplete_blocks(tmp_path: Path) -> None:
    """Break caught: a partial task block is finalized as complete."""
    from latency_meta_mdp.expert_realization.source_corpus.collection import publish_source_corpus

    request, config, task, episode = _publication_fixture()
    with pytest.raises(ValueError, match="complete admitted master-task blocks"):
        publish_source_corpus(
            target=tmp_path / "empty",
            request=request,
            source_config=config,
            task_entries=(task,),
            admitted_episodes=(),
            collection_summary=_summary(),
        )


def test_source_publication_requires_episode_and_task_table_identity_match(
    tmp_path: Path,
) -> None:
    """Break caught: a same-seed episode joins a different task profile or initial state."""
    from latency_meta_mdp.expert_realization.contracts import (
        ExpertRealizationId,
        ExpertRealizationKey,
    )
    from latency_meta_mdp.expert_realization.source_corpus.collection import publish_source_corpus

    request, config, task, episode = _publication_fixture()
    wrong_task_id = replace(
        episode.metadata.task_instance_id,
        motion_profile_sha256="0" * 64,
    )
    old_key = episode.metadata.expert_realization_id.expert_realization_key
    wrong_key = ExpertRealizationKey(
        wrong_task_id,
        old_key.realization_index,
        old_key.realization_namespace_sha256,
    )
    wrong_metadata = replace(
        episode.metadata,
        task_instance_id=wrong_task_id,
        expert_realization_id=ExpertRealizationId(
            wrong_key,
            episode.metadata.expert_realization_id.task_instance_plan_set_sha256,
        ),
    )
    wrong_episode = replace(
        episode,
        metadata=wrong_metadata,
        transitions=tuple(
            replace(
                transition,
                expert_audit=replace(
                    transition.expert_audit,
                    expert_realization_id=wrong_metadata.expert_realization_id,
                ),
            )
            for transition in episode.transitions
        ),
    )

    with pytest.raises(ValueError, match="exact task metadata"):
        publish_source_corpus(
            target=tmp_path / "wrong-task",
            request=request,
            source_config=config,
            task_entries=(task,),
            admitted_episodes=(_admitted(wrong_episode),),
            collection_summary=_summary(),
        )


def test_source_publication_is_no_overwrite_and_manifest_is_last(tmp_path: Path) -> None:
    """Break caught: a rerun mutates an immutable source corpus or exposes partial completion."""
    from latency_meta_mdp.expert_realization.source_corpus.collection import publish_source_corpus

    request, config, task, episode = _publication_fixture()
    kwargs = dict(
        request=request,
        source_config=config,
        task_entries=(task,),
        admitted_episodes=(_admitted(episode),),
        collection_summary=_summary(),
    )
    target = tmp_path / "corpus"
    publish_source_corpus(target=target, **kwargs)
    first_manifest = (target / "manifest.json").read_bytes()

    with pytest.raises(FileExistsError):
        publish_source_corpus(target=target, **kwargs)
    assert (target / "manifest.json").read_bytes() == first_manifest


def test_failed_publication_removes_only_its_owned_building_tree(tmp_path: Path) -> None:
    """Break caught: a failed finalization leaves ambiguous hidden corpus payloads."""
    from latency_meta_mdp.expert_realization.source_corpus.collection import publish_source_corpus

    request, config, task, episode = _publication_fixture()
    success_event = tuple(event for event in episode.physical_events if event.kind == "success")
    incomplete_events = replace(episode, physical_events=success_event)
    target = tmp_path / "corpus"

    with pytest.raises(ValueError, match="physical-event inventory"):
        publish_source_corpus(
            target=target,
            request=request,
            source_config=config,
            task_entries=(task,),
            admitted_episodes=(_admitted(incomplete_events),),
            collection_summary=_summary(),
        )

    assert not target.exists()
    assert not list(tmp_path.glob(".corpus.building-*"))


def test_collection_summary_is_aggregate_only_and_monotonic() -> None:
    """Break caught: aggregate accounting accepts impossible counts or failure payload paths."""
    from latency_meta_mdp.expert_realization.source_corpus.collection import CollectionSummary

    with pytest.raises(ValueError, match="monotonic"):
        CollectionSummary(
            requested_realizations=1,
            planned_realizations=2,
            executed_attempts=1,
            successful_realizations=1,
            admitted_realizations=1,
            failures_by_class={},
        )
    with pytest.raises(ValueError, match="failure class"):
        CollectionSummary(
            requested_realizations=2,
            planned_realizations=2,
            executed_attempts=1,
            successful_realizations=1,
            admitted_realizations=1,
            failures_by_class={"path/to/failed-trial": 1},
        )
    summary = CollectionSummary(
        requested_realizations=2,
        planned_realizations=1,
        executed_attempts=1,
        successful_realizations=1,
        admitted_realizations=1,
        failures_by_class={},
    )
    with pytest.raises(TypeError):
        summary.failures_by_class["task_failure"] = 1  # type: ignore[index]
