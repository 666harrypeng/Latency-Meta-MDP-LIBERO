from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from latency_meta_mdp.data.vision.cache import EpisodeVisionFeatureCache
from latency_meta_mdp.legacy.belief.causal_return.motion_aware_data import (
    build_motion_aware_history_corpus,
)
from latency_meta_mdp.legacy.vision_probe_corpus import (
    VisionProbeCorpus,
    VisionProbeEpisodeRecord,
)
from latency_meta_mdp.legacy.vision_probe_data import (
    ProbeSplit,
    build_probe_sample_indices,
)
from latency_meta_mdp.runtime.temporal_contract import TemporalContract


def _episode(*, level: int, seed: int, count: int = 81, transitions=(40,)):
    ticks = np.arange(count)
    segment_index = np.zeros(count, dtype=np.int64)
    for index, transition in enumerate(transitions, start=1):
        segment_index[transition:] = index
    deployment = SimpleNamespace(
        robot_qpos=np.repeat((ticks[:, None] / 100.0), 7, axis=1),
        robot_qvel=np.repeat((ticks[:, None] / 200.0), 7, axis=1),
        gripper_qpos=np.stack((ticks / 1000.0, -ticks / 1000.0), axis=1),
        gripper_qvel=np.stack((ticks / 2000.0, -ticks / 2000.0), axis=1),
    )
    supervision = SimpleNamespace(
        object_pose=np.concatenate(
            (
                np.stack((ticks, ticks + 1, ticks + 2), axis=1) / 100.0,
                np.zeros((count, 4)),
            ),
            axis=1,
        ),
        object_velocity=np.concatenate(
            (
                np.stack((ticks + 3, ticks + 4, ticks + 5), axis=1) / 200.0,
                np.zeros((count, 3)),
            ),
            axis=1,
        ),
        handoff_state=np.full(count, "driven"),
        commanded_motion_segment_index=segment_index,
    )
    return SimpleNamespace(
        episode_id=f"l{level}-seed-{seed:06d}-attempt-000",
        level=level,
        scene_seed=seed,
        boundary_count=count,
        transition_count=count - 1,
        deployment=deployment,
        supervision=supervision,
        expert_phase=np.full(count, "pregrasp"),
    )


def _record(
    *,
    level: int,
    seed: int,
    split: ProbeSplit,
    count: int = 81,
    transitions=(40,),
) -> VisionProbeEpisodeRecord:
    episode = _episode(level=level, seed=seed, count=count, transitions=transitions)
    features = np.zeros((count, 2, 196, 384), dtype=np.float16)
    features[:, 0, 0, 0] = np.arange(count, dtype=np.float16)
    indices = build_probe_sample_indices(
        episode=episode,
        history_sample_count=6,
        split=split,
    )
    return VisionProbeEpisodeRecord(
        episode=episode,
        cache=EpisodeVisionFeatureCache(
            manifest={"episode_id": episode.episode_id},
            features=features,
        ),
        indices=indices,
    )


def _source(*records: VisionProbeEpisodeRecord) -> VisionProbeCorpus:
    return VisionProbeCorpus(
        level=records[0].episode.level,
        history_sample_count=6,
        records=records,
        sample_references={split: () for split in ProbeSplit},
    )


def _temporal_contract() -> TemporalContract:
    return TemporalContract(
        schema_version=1,
        contract_id="h50_e25_d20_k6_v1",
        formal_tick_us=20_000,
        control_frequency_hz=50,
        prediction_horizon=50,
        launch_trigger_horizon=25,
        maximum_delay_ticks=20,
        history_sample_count=6,
    )


def test_motion_aware_corpus_filters_shared_sources_and_preserves_splits() -> None:
    source = _source(
        _record(level=3, seed=1000, split=ProbeSplit.TRAIN),
        _record(level=3, seed=1180, split=ProbeSplit.VALIDATION),
    )
    corpus = build_motion_aware_history_corpus(
        source=source,
        temporal_contract=_temporal_contract(),
    )
    assert corpus.sample_counts[ProbeSplit.TRAIN] == 31
    assert corpus.sample_counts[ProbeSplit.VALIDATION] == 31
    assert corpus.sample_counts[ProbeSplit.HOLDOUT] == 0
    assert corpus.episode_counts[ProbeSplit.TRAIN] == 1
    assert corpus.episode_counts[ProbeSplit.VALIDATION] == 1
    assert corpus.seeds[ProbeSplit.TRAIN] == frozenset({1000})
    assert corpus.seeds[ProbeSplit.VALIDATION] == frozenset({1180})


def test_corpus_rejects_mixed_record_split_and_cross_split_seed_overlap() -> None:
    record = _record(level=3, seed=1000, split=ProbeSplit.TRAIN)
    indices = list(record.indices)
    indices[0] = replace(indices[0], split=ProbeSplit.VALIDATION)
    with np.testing.assert_raises_regex(ValueError, "split"):
        build_motion_aware_history_corpus(
            source=_source(replace(record, indices=tuple(indices))),
            temporal_contract=_temporal_contract(),
        )

    with np.testing.assert_raises_regex(ValueError, "split"):
        build_motion_aware_history_corpus(
            source=_source(
                _record(level=3, seed=1000, split=ProbeSplit.TRAIN),
                _record(level=3, seed=1000, split=ProbeSplit.VALIDATION),
            ),
            temporal_contract=_temporal_contract(),
        )


def test_materialized_l3_sample_has_exact_k6_and_signed_transition_metadata() -> None:
    record = _record(level=3, seed=1000, split=ProbeSplit.TRAIN)
    corpus = build_motion_aware_history_corpus(
        source=_source(record),
        temporal_contract=_temporal_contract(),
    )
    before = corpus.materialize(ProbeSplit.TRAIN, 14)  # source tick 39
    exact = corpus.materialize(ProbeSplit.TRAIN, 15)  # source tick 40
    after = corpus.materialize(ProbeSplit.TRAIN, 16)  # source tick 41

    assert before.source_tick == 39
    assert before.transition_offset_valid
    assert before.transition_offset_ticks == -1
    assert exact.transition_offset_ticks == 0
    assert after.transition_offset_ticks == 1
    np.testing.assert_array_equal(
        exact.vision_history,
        record.cache.features[35:41],
    )
    np.testing.assert_array_equal(
        exact.robot_history[:, :7],
        record.episode.deployment.robot_qpos[35:41].astype(np.float32),
    )
    np.testing.assert_allclose(
        exact.object_position_target,
        record.episode.supervision.object_pose[40, :3],
    )
    np.testing.assert_allclose(
        exact.object_velocity_target,
        record.episode.supervision.object_velocity[40, :3],
    )
    assert set(exact.deployment_inputs()) == {
        "vision_history",
        "robot_history",
        "history_valid_mask",
    }


def test_l1_has_no_transition_metadata_and_repeated_materialization_is_stable() -> None:
    record = _record(level=1, seed=1000, split=ProbeSplit.TRAIN, transitions=())
    corpus = build_motion_aware_history_corpus(
        source=_source(record),
        temporal_contract=_temporal_contract(),
    )
    sample = corpus.materialize(ProbeSplit.TRAIN, 0)
    repeated = corpus.materialize(ProbeSplit.TRAIN, 0)
    assert not sample.transition_offset_valid
    assert sample.transition_offset_ticks == 0
    for name in sample.deployment_inputs():
        np.testing.assert_array_equal(
            sample.deployment_inputs()[name],
            repeated.deployment_inputs()[name],
        )


def test_state_values_match_materialized_targets_without_changing_identity() -> None:
    record = _record(level=3, seed=1000, split=ProbeSplit.TRAIN)
    corpus = build_motion_aware_history_corpus(
        source=_source(record),
        temporal_contract=_temporal_contract(),
    )
    sample = corpus.materialize(ProbeSplit.TRAIN, 15)
    robot, position, velocity = corpus.state_values(ProbeSplit.TRAIN, 15)
    offset, valid = corpus.transition_metadata(ProbeSplit.TRAIN, 15)
    np.testing.assert_array_equal(robot, sample.robot_history)
    np.testing.assert_array_equal(position, sample.object_position_target)
    np.testing.assert_array_equal(velocity, sample.object_velocity_target)
    assert (offset, valid) == (
        sample.transition_offset_ticks,
        sample.transition_offset_valid,
    )


def test_equal_distance_transition_tie_selects_earlier_transition() -> None:
    record = _record(
        level=3,
        seed=1000,
        split=ProbeSplit.TRAIN,
        count=121,
        transitions=(40, 80),
    )
    corpus = build_motion_aware_history_corpus(
        source=_source(record),
        temporal_contract=_temporal_contract(),
    )
    sample = corpus.materialize(ProbeSplit.TRAIN, 35)  # source tick 60
    assert sample.source_tick == 60
    assert sample.transition_offset_ticks == 20


def test_formal_l3_seed_1000_exact_transition_observability_oracle() -> None:
    path = (
        "outputs/bulk/expert/panda-ball-formal-train-1000-1199-7571a4c/"
        "episodes/L3/seed_001000/arrays.npz"
    )
    with np.load(path, allow_pickle=False) as source:
        position = np.asarray(source["object_pose"][:, :3])
        velocity = np.asarray(source["object_velocity"][:, :3])
        commanded_position = np.asarray(source["commanded_motion_position"])
        commanded_velocity = np.asarray(source["commanded_motion_velocity"])
        segment_index = np.asarray(source["commanded_motion_segment_index"])

    np.testing.assert_array_equal(
        np.flatnonzero(segment_index[1:] != segment_index[:-1]) + 1,
        [44, 79],
    )
    for window in (slice(43, 46), slice(78, 81)):
        np.testing.assert_allclose(
            position[window],
            commanded_position[window],
            atol=0.0,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            velocity[window],
            commanded_velocity[window],
            atol=0.0,
            rtol=0.0,
        )
    np.testing.assert_allclose(
        (position[44] - position[43]) / 0.02,
        velocity[43],
        atol=1e-3,
    )
    assert np.linalg.norm(velocity[44] - velocity[43]) > 0.05
    np.testing.assert_allclose(
        (position[45] - position[44]) / 0.02,
        velocity[45],
        atol=1e-3,
    )
