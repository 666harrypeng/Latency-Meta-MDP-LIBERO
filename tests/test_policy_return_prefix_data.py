from __future__ import annotations

import numpy as np
import pytest
from test_action_conditioned_jepa_data import _record
from test_policy_return_data import _NativeRows, _Predictions


def _prefix_dataset(tmp_path, mode="predicted_mixture"):
    from latency_meta_mdp.legacy.policy.policy_return_data import MatchedReturnPolicyDataset

    record = _record(tmp_path, terminal_tick=60)
    pmf = np.arange(1, 21, dtype=float)
    pmf /= pmf.sum()
    dataset = MatchedReturnPolicyDataset(
        native_dataset=_NativeRows(record),
        records=(record,),
        episode_rows=(
            {
                "episode_id": record.episode_id,
                "frame_count": 60,
                "level": 3,
                "logical_master_task_index": record.logical_master_task_index,
            },
        ),
        episode_probabilities={record.episode_id: pmf},
        mode=mode,
        predictions=_Predictions() if mode == "predicted_mixture" else None,
        conditioning="prefix",
    )
    return dataset, record, pmf


@pytest.mark.parametrize("mode", ["predicted_mixture", "gt_mixture", "no_future_control"])
def test_prefix_pairs_have_unit_weight_and_delay_independent_full_law(tmp_path, mode):
    dataset, record, pmf = _prefix_dataset(tmp_path, mode)
    reference = dataset[0]
    for delay in range(1, 21):
        sample = dataset[delay - 1]
        assert sample["action_loss_weight"] == 1
        assert dataset.sample_identity(delay - 1)["loss_weight"] == 1
        for key in reference["return_belief"]:
            np.testing.assert_array_equal(
                reference["return_belief"][key], sample["return_belief"][key]
            )
        np.testing.assert_allclose(sample["return_belief"]["latency_probabilities"], pmf)
        np.testing.assert_array_equal(sample["actions"][0], record.controls[10 + delay])
        assert "known_delay_oracle" not in sample


def test_prefix_oracle_adds_only_delay_to_the_complete_gt_packet(tmp_path):
    dataset, _, _ = _prefix_dataset(tmp_path, "gt_mixture")
    gt = dataset[7]
    dataset.mode = "known_delay_oracle"
    oracle = dataset[7]
    for key in gt["return_belief"]:
        np.testing.assert_array_equal(oracle["return_belief"][key], gt["return_belief"][key])
    assert oracle["known_delay_oracle"] == {"known_delay_ticks": 8}
    np.testing.assert_array_equal(oracle["actions"], gt["actions"])
    # The five time anchors remain distinct, rather than repeating one oracle frame.
    np.testing.assert_array_equal(
        oracle["return_belief"]["visual"][:, 0, 0, 0], [14, 18, 22, 26, 30]
    )


def test_uniform_sampler_balances_real_labels_without_dropping_partial_h50(tmp_path):
    dataset, _, _ = _prefix_dataset(tmp_path)
    sampler = dataset.training_sampler(seed=9)
    indices = list(sampler)
    assert len(indices) == len(sampler) == 20 * 49
    delays = np.array([dataset.sample_identity(i)["supervision_delay_ticks"] for i in indices])
    np.testing.assert_array_equal(np.bincount(delays, minlength=21)[1:], 49)
    for block in delays.reshape(-1, 20):
        np.testing.assert_array_equal(np.sort(block), np.arange(1, 21))
    for delay in range(1, 21):
        selected = [i for i, d in zip(indices, delays, strict=True) if d == delay]
        seen = {dataset.sample_identity(i)["source_tick"] for i in selected}
        assert seen == set(range(10, 60 - delay))
        assert all(not dataset[i]["actions_is_pad"][0] for i in selected)
    assert any(dataset[i]["actions_is_pad"][1:].any() for i in indices)
    # Direct access to unsupported terminal targets is still well defined, but not sampled.
    assert dataset[len(dataset) - 1]["actions_is_pad"].all()
    assert list(dataset.training_sampler(seed=9)) == indices
    assert list(sampler) != indices


def test_balanced_epoch_has_no_partial_batch_to_drop(tmp_path):
    dataset, _, _ = _prefix_dataset(tmp_path)
    sampler = dataset.training_sampler(seed=2, batch_size=128)
    indices = list(sampler)
    assert len(indices) % 128 == 0
    delays = [dataset.sample_identity(i)["supervision_delay_ticks"] for i in indices]
    counts = np.bincount(delays, minlength=21)[1:]
    assert counts.min() == counts.max()
    for delay in range(1, 21):
        ticks = {
            dataset.sample_identity(i)["source_tick"]
            for i, d in zip(indices, delays, strict=True)
            if d == delay
        }
        assert ticks == set(range(10, 60 - delay))


def test_legacy_view_keeps_original_episode_weighting(tmp_path):
    dataset, _, pmf = _prefix_dataset(tmp_path)
    dataset.conditioning = "late"
    assert dataset[0]["action_loss_weight"] == pytest.approx(20 * pmf[0])
    assert "latency_probabilities" not in dataset[0]["return_belief"]


def test_resume_sampler_skips_completed_batches_without_decoding_old_rows(tmp_path):
    dataset, _, _ = _prefix_dataset(tmp_path)
    original = dataset.training_sampler(seed=11, batch_size=128)
    first = list(original)
    second = list(original)
    third = list(original)
    batches_per_epoch = len(first) // 128
    resumed = dataset.training_sampler(seed=11, batch_size=128, start_batch=batches_per_epoch + 3)
    assert len(resumed) == len(second) - 3 * 128
    assert list(resumed) == second[3 * 128 :]
    assert list(resumed) == third
