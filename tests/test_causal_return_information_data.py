from types import SimpleNamespace

import numpy as np

from latency_meta_mdp.data.vision.cache import EpisodeVisionFeatureCache
from latency_meta_mdp.legacy.belief.causal_return.information_data import (
    build_information_state_corpus,
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


def _episode(*, level: int, seed: int, count: int = 81):
    ticks = np.arange(count)
    transition_count = count - 1
    deployment = SimpleNamespace(
        robot_qpos=np.repeat((ticks[:, None] / 100.0), 7, axis=1),
        robot_qvel=np.repeat((ticks[:, None] / 200.0), 7, axis=1),
        gripper_qpos=np.stack((ticks / 1000.0, -ticks / 1000.0), axis=1),
        gripper_qvel=np.stack((ticks / 2000.0, -ticks / 2000.0), axis=1),
    )
    supervision = SimpleNamespace(
        object_pose=np.concatenate(
            (np.stack((ticks, ticks + 1, ticks + 2), axis=1) / 100.0, np.zeros((count, 4))),
            axis=1,
        ),
        object_velocity=np.concatenate(
            (np.stack((ticks + 3, ticks + 4, ticks + 5), axis=1) / 200.0, np.zeros((count, 3))),
            axis=1,
        ),
        handoff_state=np.full(count, "driven"),
        commanded_motion_velocity=np.stack(
            (np.ones(count), ticks / 100.0, np.zeros(count)), axis=1
        ),
        commanded_motion_acceleration=np.stack(
            (np.zeros(count), np.ones(count) / 100.0, np.zeros(count)), axis=1
        ),
        commanded_motion_segment_index=np.where(ticks < 40, 0, 1),
    )
    return SimpleNamespace(
        episode_id=f"l{level}-seed-{seed:06d}-attempt-000",
        level=level,
        scene_seed=seed,
        boundary_count=count,
        transition_count=transition_count,
        deployment=deployment,
        supervision=supervision,
        expert_phase=np.asarray(
            ["pregrasp"] * 30 + ["approach"] * 20 + ["close"] * 15 + ["lift"] * 15
        ),
    )


def _record(*, level: int, seed: int, split: ProbeSplit) -> VisionProbeEpisodeRecord:
    episode = _episode(level=level, seed=seed)
    size = episode.boundary_count * 2 * 196 * 384
    features = (
        (np.arange(size, dtype=np.int32) % 1024)
        .astype(np.float16)
        .reshape(
            episode.boundary_count,
            2,
            196,
            384,
        )
    )
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


def test_information_corpus_filters_to_shared_causal_return_sources() -> None:
    records = (
        _record(level=1, seed=1000, split=ProbeSplit.TRAIN),
        _record(level=1, seed=1180, split=ProbeSplit.VALIDATION),
    )
    source = VisionProbeCorpus(
        level=1,
        history_sample_count=6,
        records=records,
        sample_references={split: () for split in ProbeSplit},
    )

    corpus = build_information_state_corpus(
        source=source,
        temporal_contract=_temporal_contract(),
    )

    assert corpus.sample_counts[ProbeSplit.TRAIN] == 31
    assert corpus.sample_counts[ProbeSplit.VALIDATION] == 31
    assert corpus.sample_counts[ProbeSplit.HOLDOUT] == 0
    assert corpus.episode_counts[ProbeSplit.TRAIN] == 1
    assert corpus.episode_counts[ProbeSplit.VALIDATION] == 1


def test_materialized_information_sample_has_exact_history_and_target_order() -> None:
    record = _record(level=1, seed=1000, split=ProbeSplit.TRAIN)
    source = VisionProbeCorpus(
        level=1,
        history_sample_count=6,
        records=(record,),
        sample_references={split: () for split in ProbeSplit},
    )
    corpus = build_information_state_corpus(
        source=source,
        temporal_contract=_temporal_contract(),
    )

    sample = corpus.materialize(ProbeSplit.TRAIN, 0)
    repeated = corpus.materialize(ProbeSplit.TRAIN, 0)

    assert sample.source_tick == 25
    assert sample.source_phase == "pregrasp"
    assert sample.motion_curvature > 0.0
    assert sample.motion_transition_distance_ticks == 15
    np.testing.assert_array_equal(sample.history_time_ms, [-100, -80, -60, -40, -20, 0])
    np.testing.assert_array_equal(sample.vision_history, record.cache.features[20:26])
    np.testing.assert_array_equal(
        sample.robot_history[:, :7],
        record.episode.deployment.robot_qpos[20:26].astype(np.float32),
    )
    np.testing.assert_allclose(
        sample.object_state_target,
        np.concatenate(
            (
                record.episode.supervision.object_pose[25, :3],
                record.episode.supervision.object_velocity[25, :3],
            )
        ),
    )
    assert sample.model_inputs().keys() == repeated.model_inputs().keys()
    for name in sample.model_inputs():
        np.testing.assert_array_equal(sample.model_inputs()[name], repeated.model_inputs()[name])
    assert all("buffer" not in key and "latency" not in key for key in sample.model_inputs())
