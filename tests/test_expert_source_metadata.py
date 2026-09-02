from __future__ import annotations

import hashlib

import pyarrow as pa
import pytest
from expert_realization_test_support import make_formal_source_episode


def _task_entry(*, logical_index: int = 0, split: str = "train"):
    from latency_meta_mdp.expert_realization.contracts import TaskInstanceId
    from latency_meta_mdp.expert_realization.source_corpus.metadata import (
        SourceTaskMetadataEntry,
    )

    motion = b'{"level":1,"profile":"constant_velocity"}\n'
    initial = b"exact-initial-state-npz"
    task_id = TaskInstanceId(
        level=1,
        task_instance_seed=4000 + logical_index,
        motion_profile_sha256=hashlib.sha256(motion).hexdigest(),
        initial_state_sha256=hashlib.sha256(initial).hexdigest(),
    )
    return SourceTaskMetadataEntry(
        task_instance_id=task_id,
        corpus_id="panda-ball-structured-source-pilot",
        logical_master_task_index=logical_index,
        split=split,
        instruction="Grasp the moving ball and lift it.",
        motion_profile_json=motion.decode(),
        initial_state_npz=initial,
        admitted_realization_count=4,
    )


def _episode_entry(*, episode_id: str = "source-l1-task000-r000", split: str = "train"):
    from dataclasses import replace

    from latency_meta_mdp.expert_realization.source_corpus.metadata import (
        SourceEpisodeMetadataEntry,
    )
    from latency_meta_mdp.expert_realization.source_corpus.parquet import EpisodeLocation

    episode = make_formal_source_episode()
    if episode_id != episode.metadata.episode_id:
        episode = replace(episode, metadata=replace(episode.metadata, episode_id=episode_id))
    return SourceEpisodeMetadataEntry(
        episode=episode,
        logical_master_task_index=0,
        split=split,
        strategy_parameters={"prediction_lead_seconds": 0.12},
        selected_planner_fingerprint="c" * 64,
        qualification={"eligible": True, "failures": []},
        location=EpisodeLocation(
            episode_id=episode_id,
            row_group_index=0,
            row_offset=0,
            row_count=2,
        ),
        data_shard="data/level-1/shard-00000.parquet",
    )


def test_task_instance_table_centralizes_exact_task_payloads() -> None:
    """Break caught: task identity/config payload is repeated or detached from its hashes."""
    from latency_meta_mdp.expert_realization.source_corpus.metadata import (
        build_task_instance_table,
    )
    from latency_meta_mdp.expert_realization.source_corpus.schema import TASK_INSTANCE_SCHEMA

    table = build_task_instance_table((_task_entry(),))

    assert table.schema == TASK_INSTANCE_SCHEMA
    assert table.num_rows == 1
    assert table["logical_master_task_index"].to_pylist() == [0]
    assert table["split"].to_pylist() == ["train"]
    assert table["admitted_realization_count"].to_pylist() == [4]
    assert table["initial_state_npz"][0].as_py() == b"exact-initial-state-npz"


def test_task_metadata_rejects_hash_split_and_identity_drift() -> None:
    """Break caught: centralized task metadata accepts corrupt bytes or group leakage."""
    from dataclasses import replace

    entry = _task_entry()
    with pytest.raises(ValueError, match="motion profile hash"):
        replace(entry, motion_profile_json='{"changed":true}\n')
    with pytest.raises(ValueError, match="initial state hash"):
        replace(entry, initial_state_npz=b"changed")
    with pytest.raises(ValueError, match="split"):
        replace(entry, split="holdout")


def test_episode_table_is_success_only_and_locates_one_row_group() -> None:
    """Break caught: a source episode needs per-episode JSON or ambiguous shard offsets."""
    from latency_meta_mdp.expert_realization.source_corpus.metadata import build_episode_table
    from latency_meta_mdp.expert_realization.source_corpus.schema import EPISODE_SCHEMA

    table = build_episode_table((_episode_entry(),))

    assert table.schema == EPISODE_SCHEMA
    assert table.num_rows == 1
    assert table["episode_id"].to_pylist() == ["source-l1-task000-r000"]
    assert table["frame_count"].to_pylist() == [2]
    assert table["terminal_tick"].to_pylist() == [1]
    assert table["success_time_us"].to_pylist() == [20_000]
    assert table["data_shard"].to_pylist() == ["data/level-1/shard-00000.parquet"]
    assert table["row_group_index"].to_pylist() == [0]
    assert table["row_count"].to_pylist() == [2]


def test_episode_table_rejects_duplicate_ids_and_cross_level_paths() -> None:
    """Break caught: two logical episodes alias one identity or a level partition."""
    from dataclasses import replace

    from latency_meta_mdp.expert_realization.source_corpus.metadata import build_episode_table

    entry = _episode_entry()
    with pytest.raises(ValueError, match="episode IDs must be unique"):
        build_episode_table((entry, entry))
    with pytest.raises(ValueError, match="level partition"):
        replace(entry, data_shard="data/level-2/shard-00000.parquet")
    with pytest.raises(ValueError, match="location"):
        replace(entry, location=replace(entry.location, episode_id="other"))


def test_event_table_is_a_single_relational_log() -> None:
    """Break caught: physical events are fragmented into one JSON file per episode."""
    from latency_meta_mdp.expert_realization.source_corpus.metadata import build_event_table
    from latency_meta_mdp.expert_realization.source_corpus.schema import EVENT_SCHEMA

    episode = make_formal_source_episode()
    table = build_event_table((episode,))

    assert table.schema == EVENT_SCHEMA
    assert table.num_rows == 5
    assert table["episode_id"].to_pylist() == [episode.metadata.episode_id] * 5
    assert table["event_index"].to_pylist() == list(range(5))
    assert table["kind"].to_pylist() == [
        "first_contact",
        "stable_grasp",
        "handoff",
        "lift_threshold",
        "success",
    ]
    assert table["terminal_reason"].to_pylist() == [None, None, None, None, "lift_succeeded"]


def test_metadata_builders_reject_empty_or_duplicate_inputs() -> None:
    """Break caught: an empty metadata table can masquerade as a complete source corpus."""
    from latency_meta_mdp.expert_realization.source_corpus.metadata import (
        build_episode_table,
        build_event_table,
        build_task_instance_table,
    )

    with pytest.raises(ValueError, match="non-empty"):
        build_task_instance_table(())
    with pytest.raises(ValueError, match="non-empty"):
        build_episode_table(())
    with pytest.raises(ValueError, match="non-empty"):
        build_event_table(())


def test_schema_and_provenance_documents_are_centralized_and_model_agnostic() -> None:
    """Break caught: source metadata embeds a training transform or omits collection identity."""
    from latency_meta_mdp.expert_realization.source_corpus.metadata import (
        build_provenance_document,
        build_schema_document,
    )

    episode = make_formal_source_episode()
    schema = build_schema_document()
    provenance = build_provenance_document((episode.metadata,))

    assert schema["format_id"] == "structured_expert_source_parquet_v1"
    assert provenance == {
        "schema_version": 1,
        "format_id": "structured_expert_source_provenance_v1",
        "corpus_id": "panda-ball-structured-source-pilot",
        "record_profile": "formal_source",
        "task_id": "dynamic_grasp_lift",
        "instruction": "Grasp the moving ball and lift it.",
        "physics_dt_us": 2000,
        "formal_tick_us": 20000,
        "camera_height": 2,
        "camera_width": 3,
        "action_contract_id": "panda_osc_pose_delta_v1",
        "action_dim": 7,
        "actuator_dim": 9,
        "expert_id": "panda_ball_smooth_approach_funnel_v1",
        "formal_corpus_config_sha256": "e" * 64,
        "source_corpus_config_sha256": "f" * 64,
        "master_task_split_plan_sha256": "0" * 64,
        "task_config_sha256": "1" * 64,
        "motion_config_sha256_by_level": {"1": "2" * 64},
        "runtime_config_sha256": "3" * 64,
        "controller_config_sha256": "4" * 64,
        "structured_expert_config_sha256": "f" * 64,
        "curobo_planner_config_sha256": "5" * 64,
        "implementation": {
            "revision": "1" * 40,
            "source_sha256": "b" * 64,
            "dirty": False,
        },
    }
    assert "normalization" not in provenance
    assert "history_window" not in provenance


def test_provenance_rejects_mixed_dataset_level_contracts() -> None:
    """Break caught: one corpus silently mixes camera or runtime contracts."""
    from dataclasses import replace

    from latency_meta_mdp.expert_realization.source_corpus.metadata import (
        build_provenance_document,
    )

    metadata = make_formal_source_episode().metadata
    with pytest.raises(ValueError, match="dataset-level provenance"):
        build_provenance_document((metadata, replace(metadata, camera_width=4)))


def test_provenance_allows_one_explicit_motion_config_per_level() -> None:
    """Break caught: one multi-level source repo incorrectly requires identical motion configs."""
    from dataclasses import replace

    from latency_meta_mdp.expert_realization.contracts import (
        ExpertRealizationId,
        ExpertRealizationKey,
    )
    from latency_meta_mdp.expert_realization.source_corpus.metadata import (
        build_provenance_document,
    )

    first = make_formal_source_episode().metadata
    second_task = replace(first.task_instance_id, level=2)
    first_key = first.expert_realization_id.expert_realization_key
    second_key = ExpertRealizationKey(
        second_task,
        first_key.realization_index,
        first_key.realization_namespace_sha256,
    )
    second = replace(
        first,
        episode_id="source-l2-task000-r000",
        task_instance_id=second_task,
        expert_realization_id=ExpertRealizationId(
            second_key,
            first.expert_realization_id.task_instance_plan_set_sha256,
        ),
        motion_config_sha256="a" * 64,
    )

    provenance = build_provenance_document((first, second))

    assert provenance["motion_config_sha256_by_level"] == {
        "1": "2" * 64,
        "2": "a" * 64,
    }


def test_metadata_tables_are_plain_arrow_without_framework_specific_metadata() -> None:
    """Break caught: source tables require an OpenPI or LeRobot custom loader."""
    from latency_meta_mdp.expert_realization.source_corpus.metadata import build_episode_table

    table = build_episode_table((_episode_entry(),))
    assert isinstance(table, pa.Table)
    assert table.schema.metadata is None
