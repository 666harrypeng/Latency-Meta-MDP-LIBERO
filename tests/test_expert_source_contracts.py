from __future__ import annotations

from dataclasses import replace

import pytest
from expert_realization_test_support import (
    make_formal_source_metadata,
    make_formal_source_records,
)


def test_formal_source_metadata_has_no_pilot_or_attempt_identity() -> None:
    """Break caught: canonical source provenance inherits pilot/attempt-only fields."""
    metadata = make_formal_source_metadata()
    mapping = metadata.to_mapping()

    assert metadata.record_profile == "formal_source"
    assert mapping["formal_corpus_config_sha256"] == "e" * 64
    assert mapping["source_corpus_config_sha256"] == "f" * 64
    assert "master_task_split_plan_sha256" not in mapping
    assert (
        metadata.expert_realization_id.expert_realization_key.realization_namespace_sha256
        != metadata.structured_expert_config_sha256
    )
    assert not {"pilot_config_sha256", "pilot_gate_config_sha256", "attempt_id"} & set(mapping)
    from latency_meta_mdp.expert_realization.source_corpus.contracts import (
        FormalSourceEpisodeMetadata,
    )

    assert FormalSourceEpisodeMetadata.from_mapping(mapping) == metadata


def test_formal_source_metadata_rejects_identity_and_protocol_drift() -> None:
    """Break caught: a source episode can detach from its formal corpus or frozen plan."""
    metadata = make_formal_source_metadata()

    with pytest.raises(ValueError, match="record_profile"):
        replace(metadata, record_profile="pilot_debug")
    with pytest.raises(ValueError, match="logical_master_task_index"):
        replace(metadata, logical_master_task_index=-1)
    with pytest.raises(ValueError, match="plan-set"):
        replace(metadata, frozen_plan_set_manifest_sha256="1" * 64)
    with pytest.raises(ValueError, match="clock"):
        replace(metadata, formal_tick_us=10_000)


def test_formal_source_episode_reuses_records_but_accepts_only_success() -> None:
    """Break caught: a failed execution becomes a canonical source episode payload."""
    from latency_meta_mdp.expert_realization.source_corpus.contracts import (
        FormalSourceSynchronizedEpisode,
    )

    metadata = make_formal_source_metadata()
    boundaries, transitions, events = make_formal_source_records(metadata)
    episode = FormalSourceSynchronizedEpisode(
        metadata=metadata,
        boundaries=boundaries,
        transitions=transitions,
        physical_events=events,
        terminal_reason="lift_succeeded",
    )

    assert episode.terminal_status == "success"
    assert len(episode.boundaries) == 2
    assert len(episode.transitions) == 1
    with pytest.raises(TypeError, match="StructuredBoundaryRecord"):
        replace(episode, boundaries=(object(),))
    with pytest.raises(ValueError, match="terminal boundary"):
        replace(
            episode,
            boundaries=(boundaries[0], replace(boundaries[1], outcome_status="running")),
        )


def test_source_episode_rejects_clock_and_realization_join_drift() -> None:
    """Break caught: source transitions or expert audits refer to another tick/realization."""
    from latency_meta_mdp.expert_realization.source_corpus.contracts import (
        FormalSourceSynchronizedEpisode,
    )

    metadata = make_formal_source_metadata()
    boundaries, transitions, events = make_formal_source_records(metadata)

    with pytest.raises(ValueError, match="source tick"):
        FormalSourceSynchronizedEpisode(
            metadata=metadata,
            boundaries=boundaries,
            transitions=(replace(transitions[0], target_formal_tick=2),),
            physical_events=events,
            terminal_reason="lift_succeeded",
        )
    wrong_audit = replace(
        transitions[0].expert_audit,
        expert_realization_id=replace(
            metadata.expert_realization_id,
            task_instance_plan_set_sha256="1" * 64,
        ),
    )
    with pytest.raises(ValueError, match="realization identity"):
        FormalSourceSynchronizedEpisode(
            metadata=metadata,
            boundaries=boundaries,
            transitions=(replace(transitions[0], expert_audit=wrong_audit),),
            physical_events=events,
            terminal_reason="lift_succeeded",
        )


def test_source_episode_requires_one_terminal_success_event_at_the_final_boundary() -> None:
    """Break caught: an admitted source episode lacks its causal success timestamp."""
    from latency_meta_mdp.expert_realization.source_corpus.contracts import (
        FormalSourceSynchronizedEpisode,
    )

    metadata = make_formal_source_metadata()
    boundaries, transitions, events = make_formal_source_records(metadata)

    with pytest.raises(ValueError, match="one terminal success event"):
        FormalSourceSynchronizedEpisode(
            metadata=metadata,
            boundaries=boundaries,
            transitions=transitions,
            physical_events=(),
            terminal_reason="lift_succeeded",
        )
    from latency_meta_mdp.expert_realization.recording_contracts import (
        StructuredPhysicalEventRecord,
    )

    early = StructuredPhysicalEventRecord(
        kind="success",
        physics_step_index=0,
        time_us=0,
        payload={"lift_height_m": 0.1},
        terminal_reason="lift_succeeded",
    )
    with pytest.raises(ValueError, match="terminal boundary time"):
        FormalSourceSynchronizedEpisode(
            metadata=metadata,
            boundaries=boundaries,
            transitions=transitions,
            physical_events=(early,),
            terminal_reason="lift_succeeded",
        )
