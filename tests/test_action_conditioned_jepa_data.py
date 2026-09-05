from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from latency_meta_mdp.vision_feature_cache import EpisodeVisionFeatureCache


def _record(
    tmp_path: Path,
    *,
    episode_id: str = "episode-train",
    split: str = "train",
    terminal_tick: int = 10,
    future_offset: float = 0.0,
):
    from latency_meta_mdp.belief.action_conditioned_jepa.data_adapter import JepaEpisodeRecord

    feature_path = tmp_path / f"{episode_id}-{future_offset}.npy"
    features = np.lib.format.open_memmap(
        feature_path,
        mode="w+",
        dtype=np.float16,
        shape=(terminal_tick + 1, 2, 196, 384),
    )
    for tick in range(terminal_tick + 1):
        features[tick] = tick + (future_offset if tick >= 6 else 0.0)
    features.flush()
    del features
    features = np.load(feature_path, mmap_mode="r", allow_pickle=False)

    proprio = np.stack(
        [np.arange(16, dtype=np.float32) + tick for tick in range(terminal_tick + 1)]
    )
    proprio[:, 0] = 5.0
    if future_offset:
        proprio[6:] += future_offset
        proprio[:, 0] = 5.0
    controls = np.stack(
        [
            np.array(
                [
                    tick / terminal_tick,
                    (tick + 1) / (terminal_tick + 1),
                    -(tick + 1) / (terminal_tick + 1),
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
                dtype=np.float32,
            )
            for tick in range(terminal_tick)
        ]
    )
    phases = tuple(
        "approach" if tick < terminal_tick else None for tick in range(terminal_tick + 1)
    )
    statuses = tuple(
        "running" if tick < terminal_tick else "success" for tick in range(terminal_tick + 1)
    )
    return JepaEpisodeRecord(
        episode_id=episode_id,
        task_instance_id=f"task-{episode_id}",
        logical_master_task_index=0 if split == "train" else 1,
        level=3,
        split=split,
        terminal_tick=terminal_tick,
        cache=EpisodeVisionFeatureCache(manifest={}, features=features),
        proprio_physical=proprio,
        controls=controls,
        phases=phases,
        statuses=statuses,
    )


def _normalization(record):
    from latency_meta_mdp.belief.action_conditioned_jepa.data_adapter import (
        compute_jepa_proprio_normalization,
    )

    return compute_jepa_proprio_normalization(
        records=(record,),
        source_manifest_sha256="a" * 64,
        split_manifest_sha256="b" * 64,
    )


def test_nominal_sample_uses_exact_boundary_action_alignment(tmp_path: Path) -> None:
    from latency_meta_mdp.belief.action_conditioned_jepa.data_adapter import (
        materialize_jepa_sample,
    )

    record = _record(tmp_path)
    sample = materialize_jepa_sample(
        record,
        source_tick=5,
        normalization=_normalization(record),
    )

    np.testing.assert_array_equal(sample.boundary_ticks, np.arange(0, 6))
    np.testing.assert_array_equal(sample.executed_control_ticks, np.arange(0, 5))
    np.testing.assert_array_equal(sample.executable_control_ticks, np.arange(5, 25))
    np.testing.assert_array_equal(sample.target_ticks, np.arange(6, 26))
    np.testing.assert_array_equal(sample.executed_controls, record.controls[0:5])
    np.testing.assert_array_equal(sample.executable_controls[:5], record.controls[5:10])
    assert sample.launch_context.vision_history.shape == (1, 6, 2, 196, 384)
    assert sample.future_rollout.future_visual_latents.shape == (1, 20, 2, 196, 384)


def test_future_target_mutation_cannot_change_launch_context(tmp_path: Path) -> None:
    from latency_meta_mdp.belief.action_conditioned_jepa.data_adapter import (
        materialize_jepa_sample,
    )

    original_record = _record(tmp_path, episode_id="original")
    changed_record = _record(tmp_path, episode_id="changed", future_offset=100.0)
    normalization = _normalization(original_record)
    original = materialize_jepa_sample(
        original_record,
        source_tick=5,
        normalization=normalization,
    )
    changed = materialize_jepa_sample(
        changed_record,
        source_tick=5,
        normalization=normalization,
    )

    np.testing.assert_array_equal(original.vision_history, changed.vision_history)
    np.testing.assert_array_equal(original.proprio_history, changed.proprio_history)
    np.testing.assert_array_equal(original.executed_controls, changed.executed_controls)
    np.testing.assert_array_equal(original.executable_controls, changed.executable_controls)
    assert not np.array_equal(original.future_visual_latents, changed.future_visual_latents)
    assert not np.array_equal(original.future_proprio, changed.future_proprio)


def test_terminal_absorbing_targets_preserve_real_terminal_then_hold(tmp_path: Path) -> None:
    from latency_meta_mdp.belief.action_conditioned_jepa.data_adapter import (
        formal_hold_suffix,
        materialize_jepa_sample,
    )

    record = _record(tmp_path, terminal_tick=10)
    sample = materialize_jepa_sample(
        record,
        source_tick=8,
        normalization=_normalization(record),
    )
    terminal_offset = int(np.flatnonzero(sample.target_ticks == record.terminal_tick)[0])

    assert terminal_offset == 1
    np.testing.assert_array_equal(
        sample.future_visual_latents[terminal_offset],
        record.cache.features[record.terminal_tick],
    )
    np.testing.assert_array_equal(
        sample.future_proprio[terminal_offset],
        record.proprio_physical[record.terminal_tick],
    )
    assert not sample.target_absorbing[terminal_offset]
    assert np.all(sample.target_absorbing[terminal_offset + 1 :])
    np.testing.assert_array_equal(
        sample.future_visual_latents[-1],
        record.cache.features[record.terminal_tick],
    )
    np.testing.assert_array_equal(sample.future_proprio[-1, 7:14], 0.0)
    assert sample.future_proprio[-1, 15] == 0.0
    np.testing.assert_array_equal(
        sample.executable_controls[terminal_offset],
        record.controls[record.terminal_tick - 1],
    )
    np.testing.assert_array_equal(
        sample.executable_controls[terminal_offset + 1 :],
        formal_hold_suffix(
            record.last_real_gripper_command,
            20 - terminal_offset - 1,
        ),
    )


def test_feature_cache_stays_memory_mapped_and_samples_are_stride_one(tmp_path: Path) -> None:
    from latency_meta_mdp.belief.action_conditioned_jepa.data_adapter import (
        ActionConditionedJepaCorpus,
        compute_jepa_proprio_normalization,
    )

    first = _record(tmp_path, episode_id="first", terminal_tick=10)
    second = _record(tmp_path, episode_id="second", terminal_tick=12)
    normalization = compute_jepa_proprio_normalization(
        records=(first, second),
        source_manifest_sha256="a" * 64,
        split_manifest_sha256="b" * 64,
    )
    corpus = ActionConditionedJepaCorpus(
        records=(first, second),
        normalization=normalization,
        source_manifest_sha256="a" * 64,
        cache_manifest_sha256="d" * 64,
        split_manifest_sha256="b" * 64,
    )

    assert all(isinstance(record.cache.features, np.memmap) for record in corpus.records)
    assert corpus.sample_indices_for_episode("first") == tuple(
        replace(corpus.indices[0], source_tick=tick) for tick in range(5, 10)
    )
    assert tuple(
        index.source_tick for index in corpus.sample_indices_for_episode("second")
    ) == tuple(range(5, 12))
    assert len(corpus) == 12


def test_normalization_counts_each_train_boundary_once_and_rejects_validation(
    tmp_path: Path,
) -> None:
    from latency_meta_mdp.belief.action_conditioned_jepa.data_adapter import (
        compute_jepa_proprio_normalization,
    )

    train = _record(tmp_path, episode_id="train", split="train", terminal_tick=10)
    validation = _record(
        tmp_path,
        episode_id="validation",
        split="validation",
        terminal_tick=10,
        future_offset=100.0,
    )
    normalization = compute_jepa_proprio_normalization(
        records=(train,),
        source_manifest_sha256="a" * 64,
        split_manifest_sha256="b" * 64,
    )

    np.testing.assert_allclose(normalization.mean, train.proprio_physical.mean(axis=0))
    assert normalization.boundary_count == train.terminal_tick + 1
    assert normalization.episode_ids == (train.episode_id,)
    assert normalization.constant_dimension_mask[0]
    assert normalization.scale[0] == 1.0
    restored = normalization.denormalize(normalization.normalize(train.proprio_physical))
    np.testing.assert_allclose(restored, train.proprio_physical, atol=1e-5, rtol=1e-6)

    with pytest.raises(ValueError, match="train"):
        compute_jepa_proprio_normalization(
            records=(train, validation),
            source_manifest_sha256="a" * 64,
            split_manifest_sha256="b" * 64,
        )


def test_normalization_artifact_round_trips_and_is_no_overwrite(tmp_path: Path) -> None:
    from latency_meta_mdp.belief.action_conditioned_jepa.data_adapter import (
        load_jepa_proprio_normalization,
        write_jepa_proprio_normalization,
    )

    record = _record(tmp_path, episode_id="normalization-source")
    normalization = _normalization(record)
    path = tmp_path / "normalization.json"

    assert write_jepa_proprio_normalization(path, normalization) == path
    loaded = load_jepa_proprio_normalization(path)
    assert loaded.level == normalization.level
    assert loaded.episode_ids == normalization.episode_ids
    assert loaded.boundary_count == normalization.boundary_count
    assert loaded.sample_index_sha256 == normalization.sample_index_sha256
    np.testing.assert_array_equal(loaded.mean, normalization.mean)
    np.testing.assert_array_equal(loaded.scale, normalization.scale)
    np.testing.assert_array_equal(
        loaded.constant_dimension_mask,
        normalization.constant_dimension_mask,
    )
    with pytest.raises(FileExistsError):
        write_jepa_proprio_normalization(path, normalization)


def test_corpus_rejects_normalization_from_another_source_or_split(tmp_path: Path) -> None:
    from latency_meta_mdp.belief.action_conditioned_jepa.data_adapter import (
        ActionConditionedJepaCorpus,
    )

    record = _record(tmp_path, episode_id="train")
    wrong = replace(_normalization(record), source_manifest_sha256="d" * 64)

    with pytest.raises(ValueError, match="normalization provenance"):
        ActionConditionedJepaCorpus(
            records=(record,),
            normalization=wrong,
            source_manifest_sha256="a" * 64,
            cache_manifest_sha256="e" * 64,
            split_manifest_sha256="b" * 64,
        )


def test_shared_temporal_indices_use_the_strictest_history_for_all_candidates(
    tmp_path: Path,
) -> None:
    """Catches candidate-specific launch occupancy entering configuration selection."""

    from latency_meta_mdp.belief.action_conditioned_jepa.config import (
        load_jepa_temporal_sampling,
    )
    from latency_meta_mdp.belief.action_conditioned_jepa.temporal_view import (
        build_shared_temporal_indices,
    )

    root = Path("configs/belief/action_conditioned_jepa")
    samplings = tuple(
        load_jepa_temporal_sampling(root / name)
        for name in (
            "dense_20ms_history_100ms.yaml",
            "stride2_40ms_history_120ms.yaml",
            "stride4_80ms_history_160ms.yaml",
            "stride5_100ms_history_200ms.yaml",
        )
    )
    record = _record(tmp_path, terminal_tick=30)

    indices = build_shared_temporal_indices(records=(record,), samplings=samplings)

    assert tuple(index.source_tick for index in indices) == tuple(range(10, 30))
    assert all(index.episode_id == record.episode_id for index in indices)
    assert indices[0].boundary_disposition == "recorded_complete"
    assert indices[-1].boundary_disposition == "certified_absorbing_extension"


def test_stride2_temporal_sample_preserves_macro_control_order_and_absorbing_tail(
    tmp_path: Path,
) -> None:
    """Catches strided endpoint sampling that drops intervening controls or pads nonphysically."""

    from latency_meta_mdp.belief.action_conditioned_jepa.config import (
        load_jepa_temporal_sampling,
    )
    from latency_meta_mdp.belief.action_conditioned_jepa.temporal_view import (
        SharedJepaSampleIndex,
        materialize_temporal_jepa_sample,
    )

    sampling = load_jepa_temporal_sampling(
        Path("configs/belief/action_conditioned_jepa/stride2_40ms_history_120ms.yaml")
    )
    record = _record(tmp_path, terminal_tick=30)
    sample = materialize_temporal_jepa_sample(
        record=record,
        index=SharedJepaSampleIndex(
            level=3,
            split="train",
            episode_id=record.episode_id,
            source_tick=20,
            boundary_disposition="certified_absorbing_extension",
        ),
        sampling=sampling,
        normalization=_normalization(record),
    )

    np.testing.assert_array_equal(sample.history_ticks, np.array([14, 16, 18, 20]))
    np.testing.assert_array_equal(
        sample.past_macro_control_ticks,
        np.array([[14, 15], [16, 17], [18, 19]]),
    )
    np.testing.assert_array_equal(
        sample.future_macro_control_ticks,
        np.arange(20, 40).reshape(10, 2),
    )
    np.testing.assert_array_equal(sample.native_target_ticks, np.arange(22, 41, 2))
    np.testing.assert_array_equal(sample.native_target_rows, np.minimum(np.arange(22, 41, 2), 30))
    np.testing.assert_array_equal(sample.target_absorbing, np.arange(22, 41, 2) > 30)
    assert sample.vision_history.shape == (4, 2, 196, 384)
    assert sample.past_macro_controls.shape == (3, 2, 7)
    assert sample.future_macro_controls.shape == (10, 2, 7)
    assert sample.future_visual_latents.shape == (10, 2, 196, 384)
    assert sample.future_proprio.shape == (10, 16)
    assert sample.launch_context.executed_controls.shape == (1, 3, 2, 7)
    assert sample.launch_context.executable_controls.shape == (1, 10, 2, 7)
    assert torch.equal(
        sample.future_rollout.native_delay_ticks,
        torch.arange(2, 21, 2, dtype=torch.int64),
    )
    np.testing.assert_array_equal(
        sample.future_macro_controls[:5],
        record.controls[20:30].reshape(5, 2, 7),
    )
    np.testing.assert_array_equal(
        sample.future_macro_controls[5:, :, :6].numpy(),
        np.zeros((5, 2, 6), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        sample.future_macro_controls[5:, :, 6].numpy(),
        np.full((5, 2), record.last_real_gripper_command, dtype=np.float32),
    )
    absorbing_rows = np.flatnonzero(sample.target_absorbing).tolist()
    np.testing.assert_array_equal(
        sample.future_proprio[absorbing_rows, 7:14].numpy(),
        np.zeros((5, 7), dtype=np.float32),
    )


def test_temporal_sample_rejects_wrong_boundary_disposition(tmp_path: Path) -> None:
    """Catches labeling a truncated terminal future as fully recorded."""

    from latency_meta_mdp.belief.action_conditioned_jepa.config import (
        load_jepa_temporal_sampling,
    )
    from latency_meta_mdp.belief.action_conditioned_jepa.temporal_view import (
        SharedJepaSampleIndex,
        materialize_temporal_jepa_sample,
    )

    record = _record(tmp_path, terminal_tick=30)
    sampling = load_jepa_temporal_sampling(
        Path("configs/belief/action_conditioned_jepa/stride5_100ms_history_200ms.yaml")
    )

    with pytest.raises(ValueError, match="boundary disposition"):
        materialize_temporal_jepa_sample(
            record=record,
            index=SharedJepaSampleIndex(
                level=3,
                split="train",
                episode_id=record.episode_id,
                source_tick=20,
                boundary_disposition="recorded_complete",
            ),
            sampling=sampling,
            normalization=_normalization(record),
        )


def test_temporal_corpus_filters_fold_episodes_and_collates_dynamic_batch(
    tmp_path: Path,
) -> None:
    """Catches loading development episodes into fit or returning fixed dense tensor shapes."""

    from latency_meta_mdp.belief.action_conditioned_jepa.config import (
        load_jepa_temporal_sampling,
    )
    from latency_meta_mdp.belief.action_conditioned_jepa.data_adapter import (
        compute_jepa_proprio_normalization,
    )
    from latency_meta_mdp.belief.action_conditioned_jepa.temporal_view import (
        TemporalJepaCorpus,
        build_shared_temporal_indices,
        collate_temporal_jepa_evaluation_samples,
        collate_temporal_jepa_samples,
    )

    first = _record(tmp_path, episode_id="fit", terminal_tick=30)
    second = _record(tmp_path, episode_id="development", terminal_tick=30)
    normalization = compute_jepa_proprio_normalization(
        records=(first,),
        source_manifest_sha256="a" * 64,
        split_manifest_sha256="b" * 64,
    )
    sampling = load_jepa_temporal_sampling(
        Path("configs/belief/action_conditioned_jepa/stride4_80ms_history_160ms.yaml")
    )
    indices = build_shared_temporal_indices(
        records=(first, second),
        samplings=(sampling,),
    )
    corpus = TemporalJepaCorpus(
        records=(first, second),
        indices=indices,
        episode_ids=(first.episode_id,),
        partition="fit",
        sampling=sampling,
        normalization=normalization,
    )
    development = TemporalJepaCorpus(
        records=(second,),
        indices=indices,
        episode_ids=(second.episode_id,),
        partition="development",
        sampling=sampling,
        normalization=normalization,
    )

    assert len(corpus) == 22
    assert len(development) == 22
    assert {index.episode_id for index in corpus.indices} == {"fit"}
    batch = collate_temporal_jepa_samples((corpus[0], corpus[1]))
    assert batch.context.vision_history.shape == (2, 3, 2, 196, 384)
    assert batch.context.executed_controls.shape == (2, 2, 4, 7)
    assert batch.context.executable_controls.shape == (2, 5, 4, 7)
    assert batch.target_visual_latents.shape == (2, 2, 2, 196, 384)
    assert batch.target_proprio_normalized.shape == (2, 2, 16)
    assert batch.target_proprio_physical.shape == (2, 2, 16)
    assert batch.target_absorbing.shape == (2, 2)
    singleton = collate_temporal_jepa_samples([corpus[0]])
    assert singleton.batch_size == 1
    moved = singleton.to(torch.device("cpu"))
    assert moved.indices == singleton.indices
    assert moved.context.device == torch.device("cpu")
    evaluation = collate_temporal_jepa_evaluation_samples([corpus[0], corpus[1]])
    assert evaluation.target_visual_latents.shape == (2, 5, 2, 196, 384)
    assert evaluation.target_proprio_physical.shape == (2, 5, 16)
    assert evaluation.target_absorbing.shape == (2, 5)
    assert evaluation.to(torch.device("cpu")).context.device == torch.device("cpu")


def test_deployed_evaluation_view_materializes_dense_d20_only_for_evaluation(
    tmp_path: Path,
) -> None:
    """Catches evaluating a coarse rollout only at native anchors and hiding quantization error."""

    from latency_meta_mdp.belief.action_conditioned_jepa.config import (
        load_jepa_temporal_sampling,
    )
    from latency_meta_mdp.belief.action_conditioned_jepa.temporal_view import (
        TemporalJepaCorpus,
        TemporalJepaDeployedEvaluationCorpus,
        build_shared_temporal_indices,
        collate_temporal_jepa_deployed_evaluation_samples,
    )

    record = _record(tmp_path, episode_id="development", terminal_tick=30)
    normalization_source = _record(tmp_path, episode_id="fit", terminal_tick=30)
    normalization = _normalization(normalization_source)
    sampling = load_jepa_temporal_sampling(
        Path("configs/belief/action_conditioned_jepa/stride4_80ms_history_160ms.yaml")
    )
    indices = build_shared_temporal_indices(records=(record,), samplings=(sampling,))
    native = TemporalJepaCorpus(
        records=(record,),
        indices=indices,
        episode_ids=(record.episode_id,),
        partition="development",
        sampling=sampling,
        normalization=normalization,
    )
    deployed = TemporalJepaDeployedEvaluationCorpus(native)

    sample = deployed[0]
    assert sample.native_sample.future_visual_latents.shape[0] == 5
    assert sample.dense_target_visual_latents.shape == (20, 2, 196, 384)
    assert sample.dense_target_proprio_physical.shape == (20, 16)
    np.testing.assert_array_equal(sample.dense_delay_ticks, np.arange(1, 21))
    batch = collate_temporal_jepa_deployed_evaluation_samples((sample,))
    assert batch.native_delay_ticks.tolist() == [4, 8, 12, 16, 20]
    assert batch.dense_target_visual_latents.shape == (1, 20, 2, 196, 384)
    assert batch.to(torch.device("cpu")).context.device == torch.device("cpu")


@pytest.mark.integration
def test_formal_source_cache_split_join_for_one_episode_per_level() -> None:
    from latency_meta_mdp.belief.action_conditioned_jepa.config import (
        load_action_conditioned_jepa_config,
    )
    from latency_meta_mdp.belief.action_conditioned_jepa.data_adapter import (
        compute_jepa_proprio_normalization,
        load_verified_jepa_inputs,
        load_verified_jepa_record,
        materialize_jepa_sample,
    )

    root = Path(__file__).resolve().parents[1]
    source_root = root / "outputs/source_corpus/panda-ball-structured-source-quota-formal-100x4-v1"
    cache_manifest = (
        root
        / "outputs/derived/vision_features/dinov3-vits16-structured-source-100x4-v1/manifest.json"
    )
    split_manifest = (
        root
        / "outputs/derived/source_splits/panda-ball-structured-source-quota-formal-100x4-v1"
        / "train80-validation20-seed20260903-v1.json"
    )
    if not all(path.exists() for path in (source_root, cache_manifest, split_manifest)):
        pytest.skip("formal structured source, split, or DINO cache is absent")
    config = load_action_conditioned_jepa_config(
        model_path=root / "configs/belief/action_conditioned_jepa/model.yaml",
        level_path=root / "configs/belief/action_conditioned_jepa/l3.yaml",
        temporal_sampling_path=(
            root
            / "configs/belief/action_conditioned_jepa/dense_20ms_history_100ms.yaml"
        ),
    )
    inputs = load_verified_jepa_inputs(
        source_root=source_root,
        cache_run_manifest=cache_manifest,
        split_manifest_path=split_manifest,
        config=config,
    )

    for level in (1, 2, 3):
        episode_id = next(
            value for value in inputs.split.train_episode_ids if f"source-L{level}-" in value
        )
        record = load_verified_jepa_record(
            inputs,
            episode_id=episode_id,
            level=level,
            split="train",
        )
        assert isinstance(record.cache.features, np.memmap)
        assert record.cache.features.shape == (
            record.terminal_tick + 1,
            2,
            196,
            384,
        )
        assert record.legal_source_ticks[0] == 5
        assert record.legal_source_ticks[-1] == record.terminal_tick - 1
        assert np.all(np.isfinite(record.proprio_physical))
        assert np.all(np.isfinite(record.controls))
        normalization = compute_jepa_proprio_normalization(
            records=(record,),
            source_manifest_sha256=inputs.source_manifest_sha256,
            split_manifest_sha256=inputs.split_manifest_sha256,
        )
        for source_tick in record.legal_source_ticks:
            sample = materialize_jepa_sample(
                record,
                source_tick=source_tick,
                normalization=normalization,
            )
            assert sample.future_visual_latents.shape == (20, 2, 196, 384)
            assert sample.future_proprio.shape == (20, 16)
            sample.launch_context.validate_finite()
            sample.future_rollout.validate_finite()

    assert set(inputs.split.train_episode_ids).isdisjoint(inputs.split.validation_episode_ids)
