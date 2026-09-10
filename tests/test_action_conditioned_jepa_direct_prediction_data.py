from dataclasses import replace

import numpy as np
import pytest
import torch
from test_action_conditioned_jepa_data import _normalization, _record


def test_direct_pair_keeps_native_actions_and_only_real_endpoint(tmp_path):
    from latency_meta_mdp.belief.action_conditioned_jepa.direct_prediction_data import (
        materialize_direct_sample,
    )

    record = _record(tmp_path, terminal_tick=35)
    sample = materialize_direct_sample(
        record, source_tick=30, query_ticks=5, normalization=_normalization(record)
    )
    assert sample.query.source_ticks.tolist() == [30]
    assert sample.query.query_ticks.tolist() == [5]
    np.testing.assert_array_equal(sample.query.vision_history[0, :, 0, 0, 0], [22, 26, 30])
    np.testing.assert_array_equal(
        sample.query.executed_controls[0].reshape(8, 7), record.controls[22:30]
    )
    np.testing.assert_array_equal(sample.query.executable_controls[0, :5], record.controls[30:35])
    assert torch.count_nonzero(sample.query.executable_controls[0, 5:]) == 0
    assert torch.all(sample.target_visual == 35)
    np.testing.assert_array_equal(sample.target_proprio[0], record.proprio_physical[35])
    with pytest.raises(ValueError, match="real"):
        materialize_direct_sample(
            record, source_tick=30, query_ticks=6, normalization=_normalization(record)
        )


def test_future_observations_cannot_leak_into_query(tmp_path):
    from latency_meta_mdp.belief.action_conditioned_jepa.direct_prediction_data import (
        materialize_direct_sample,
    )

    record = _record(tmp_path, terminal_tick=32)
    norm = _normalization(record)
    sample = materialize_direct_sample(record, source_tick=10, query_ticks=7, normalization=norm)
    altered = _record(tmp_path, terminal_tick=32, episode_id="altered")
    writable = np.load(altered.cache.features.filename, mmap_mode="r+")
    writable[11:] += 100
    writable.flush()
    changed_proprio = altered.proprio_physical.copy()
    changed_proprio[11:] += 100
    changed = materialize_direct_sample(
        replace(altered, proprio_physical=changed_proprio),
        source_tick=10,
        query_ticks=7,
        normalization=norm,
    )
    for name in vars(sample.query):
        torch.testing.assert_close(getattr(sample.query, name), getattr(changed.query, name))
    assert not torch.equal(sample.target_visual, changed.target_visual)
    assert not torch.equal(sample.target_proprio, changed.target_proprio)


def test_pair_inventory_balances_all_twenty_horizons_without_dropping_late_sources(tmp_path):
    from latency_meta_mdp.belief.action_conditioned_jepa.direct_prediction_data import (
        BalancedQuerySampler,
        DirectPredictionDataset,
        collate_direct_samples,
    )

    record = _record(tmp_path, terminal_tick=35)
    dataset = DirectPredictionDataset(records=(record,), normalization=_normalization(record))
    assert dataset.horizon_counts == tuple(26 - q for q in range(1, 21))
    assert dataset.pair_at(dataset.horizon_counts[0] - 1) == (0, 34, 1)
    sampler = BalancedQuerySampler(dataset, sample_count=100, seed=27)
    first = list(sampler)
    assert first == list(sampler)
    assert (
        np.bincount([dataset.pair_at(i)[2] for i in first], minlength=21).tolist() == [0] + [5] * 20
    )
    sampler.set_epoch(1)
    assert first != list(sampler)
    for index in first:
        _, source, q = dataset.pair_at(index)
        assert source >= 10 and source + q <= record.terminal_tick
    batch = collate_direct_samples([dataset[first[0]], dataset[first[1]]])
    assert batch.query.vision_history.shape == (2, 3, 2, 196, 384)
    assert batch.target_visual.shape == (2, 2, 196, 384)


def test_dataset_rejects_mixed_split_and_wrong_training_normalization(tmp_path):
    from latency_meta_mdp.belief.action_conditioned_jepa.direct_prediction_data import (
        DirectPredictionDataset,
    )

    train = _record(tmp_path, terminal_tick=35)
    validation = _record(tmp_path, terminal_tick=35, episode_id="validation", split="validation")
    with pytest.raises(ValueError, match="split"):
        DirectPredictionDataset(records=(train, validation), normalization=_normalization(train))
    with pytest.raises(ValueError, match="normalization"):
        DirectPredictionDataset(records=(train,), normalization=_normalization(validation))
