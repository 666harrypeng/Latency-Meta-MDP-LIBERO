import numpy as np
import pytest

pytest_plugins = ("test_forecast_policy_cache",)


def test_only_view_replaces_values_shifts_labels_and_excludes_terminal(built_cache):
    from latency_meta_mdp.data.forecast.only_dataset import ForecastOnlyPolicyDataset

    cache, record, _ = built_cache

    class Native:
        def __len__(self):
            return record.terminal_tick

        def __getitem__(self, index):
            return {
                "state": record.proprio_physical[index],
                "image": np.zeros((224, 224, 3), np.uint8),
                "wrist_image": np.zeros((224, 224, 3), np.uint8),
                "prompt": "pick",
            }

    rows = [
        dict(
            episode_id=record.episode_id,
            frame_count=record.terminal_tick,
            level=record.level,
            logical_master_task_index=record.logical_master_task_index,
        )
    ]
    dataset = ForecastOnlyPolicyDataset(Native(), episode_rows=rows, cache=cache)
    assert len(dataset) == sum(max(0, record.terminal_tick - 10 - q) for q in range(1, 21))
    for index, native_index in enumerate(dataset.source_indices):
        h, q0 = divmod(int(native_index), 20)
        q, sample = q0 + 1, dataset[index]
        assert set(sample) == {
            "image",
            "wrist_image",
            "state",
            "prompt",
            "actions",
            "actions_is_pad",
        }
        np.testing.assert_array_equal(sample["image"], q)
        np.testing.assert_array_equal(sample["wrist_image"], q)
        np.testing.assert_array_equal(sample["state"], q)
        n = min(50, record.terminal_tick - h - q)
        assert n > 0 and not sample["actions_is_pad"][0]
        np.testing.assert_array_equal(sample["actions"][:n], record.controls[h + q : h + q + n])
        np.testing.assert_array_equal(sample["actions"][n:], 0)
    order = list(dataset.training_sampler(seed=7, batch_size=32))
    queries = dataset.source_indices[order] % 20
    assert len(set(np.bincount(queries))) == 1
    assert list(dataset.training_sampler(seed=7, batch_size=32, start_batch=2)) == order[64:]


def test_mode_binds_target_and_runtime_protocol():
    from latency_meta_mdp.policy.conditioning import conditioning_contract

    current = conditioning_contract("current_and_forecast")
    future = conditioning_contract("forecast_only")
    assert current["target_alignment"] == "observation_time"
    assert future["target_alignment"] == "forecast_time"
    assert future["plan_construction"] == "rtc_planned_handoff_h50_v1"
    with pytest.raises(ValueError):
        conditioning_contract("forecast")
