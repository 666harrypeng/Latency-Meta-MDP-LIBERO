import pickle

import numpy as np
import pytest


@pytest.fixture
def built_cache(tmp_path, monkeypatch):
    from test_action_conditioned_jepa_data import _normalization, _record

    import latency_meta_mdp.policy_forecast_cache as cache_module
    from latency_meta_mdp.policy_forecast_cache import ForecastCache, write_forecast_cache

    original_hash = cache_module.sha256_file

    def metadata_hash_only(path):
        assert path.name != "forecasts.sqlite", "do not hash the large prediction database"
        return original_hash(path)

    monkeypatch.setattr(cache_module, "sha256_file", metadata_hash_only)

    record = _record(tmp_path, terminal_tick=32)

    class Engine:
        def predict(self, query):
            q = query.query_ticks.numpy()
            return (
                np.broadcast_to(q[:, None, None, None, None], (len(q), 2, 224, 224, 3)).astype(
                    np.uint8
                ),
                np.broadcast_to(q[:, None], (len(q), 16)).astype(np.float32),
            )

    identity = {
        k: "a" * 64
        for k in (
            "predictor_sha256",
            "decoder_sha256",
            "jepa_normalization_sha256",
            "source_manifest_sha256",
            "split_manifest_sha256",
            "vision_cache_manifest_sha256",
        )
    }
    identity["predictor_architecture"] = "jepa_direct_q20_history_stride4_w3_v1"
    root = tmp_path / "cache"
    write_forecast_cache(
        records=(record,),
        normalization=_normalization(record),
        engine=Engine(),
        output_dir=root,
        bindings=identity,
        batch_size=32,
    )
    return ForecastCache(root, expected_bindings=identity), record, identity


def test_cache_roundtrips_lossless_and_unavailable_is_not_synthetic(built_cache):
    cache, record, identity = built_cache
    rgb, proprio = cache.read(record.episode_id, 10, 7)
    np.testing.assert_array_equal(rgb, 7)
    np.testing.assert_array_equal(proprio, 7)
    assert cache.read(record.episode_id, 31, 20) is None
    assert cache.read(record.episode_id, 0, 1) is None
    reopened = pickle.loads(pickle.dumps(cache))
    np.testing.assert_array_equal(reopened.read(record.episode_id, 10, 7)[0], rgb)
    from latency_meta_mdp.policy_forecast_cache import ForecastCache

    with pytest.raises(ValueError, match="binding"):
        ForecastCache(cache.root, expected_bindings={**identity, "decoder_sha256": "b" * 64})


def test_dataset_preserves_each_native_target_and_balances_q_with_resume(built_cache):
    cache, record, _ = built_cache
    from latency_meta_mdp.policy_forecast_dataset import ForecastPolicyDataset

    class Native:
        def __len__(self):
            return record.terminal_tick

        def __getitem__(self, index):
            count = min(50, record.terminal_tick - index)
            actions = np.zeros((50, 7), np.float32)
            actions[:count] = record.controls[index : index + count]
            return {
                "state": record.proprio_physical[index],
                "image": np.zeros((8, 8, 3), np.uint8),
                "wrist_image": np.zeros((8, 8, 3), np.uint8),
                "prompt": "pick",
                "actions": actions,
                "actions_is_pad": np.arange(50) >= count,
                "frame_index": index,
                "episode_index": 0,
            }

    rows = [
        {
            "episode_id": record.episode_id,
            "frame_count": record.terminal_tick,
            "level": record.level,
            "logical_master_task_index": record.logical_master_task_index,
        }
    ]
    data = ForecastPolicyDataset(Native(), episode_rows=rows, cache=cache)
    assert len(data) == record.terminal_tick * 20
    np.testing.assert_array_equal(data[10 * 20]["actions"], data[10 * 20 + 19]["actions"])
    assert data[31 * 20 + 19]["forecast"]["rgb"] is None
    assert (~data[31 * 20 + 19]["actions_is_pad"]).sum() == 1
    order = list(data.training_sampler(seed=1, batch_size=32))
    assert len(set(order)) == len(data)
    np.testing.assert_array_equal(np.bincount(np.array(order) % 20), len(order) // 20)
    resumed = list(data.training_sampler(seed=1, batch_size=32, start_batch=2))
    assert resumed == order[64:]
    assert data.coverage()["real_action_sources"] == record.terminal_tick


def test_forecast_generation_resumes_after_last_complete_episode(tmp_path):
    from test_action_conditioned_jepa_data import _normalization, _record

    from latency_meta_mdp.policy_forecast_cache import ForecastCache, write_forecast_cache

    records = tuple(_record(tmp_path, episode_id=f"record{i}", terminal_tick=12) for i in (1, 2))
    norm = _normalization(records[0])
    bindings = {
        key: "a" * 64
        for key in (
            "predictor_sha256",
            "decoder_sha256",
            "jepa_normalization_sha256",
            "source_manifest_sha256",
            "split_manifest_sha256",
            "vision_cache_manifest_sha256",
        )
    }
    bindings["predictor_architecture"] = "jepa_direct_q20_history_stride4_w3_v1"

    class Engine:
        def __init__(self, fail=False):
            self.calls = 0
            self.fail = fail

        def predict(self, q):
            self.calls += 1
            if self.fail and self.calls == 2:
                raise RuntimeError("interrupted")
            return np.zeros((len(q.query_ticks), 2, 224, 224, 3), np.uint8), np.zeros(
                (len(q.query_ticks), 16), np.float32
            )

    out = tmp_path / "forecasts"
    with pytest.raises(RuntimeError, match="interrupted"):
        write_forecast_cache(
            records=records,
            normalization=norm,
            engine=Engine(True),
            output_dir=out,
            bindings=bindings,
            batch_size=32,
        )
    engine = Engine()
    write_forecast_cache(
        records=records,
        normalization=norm,
        engine=engine,
        output_dir=out,
        bindings=bindings,
        batch_size=32,
        resume=True,
    )
    assert engine.calls == 1
    cache = ForecastCache(out, expected_bindings=bindings)
    assert cache.manifest["prediction_count"] == 6
    assert cache.read("record1", 10, 2)[0].shape == (2, 224, 224, 3)
    with pytest.raises(ValueError, match="identity"):
        write_forecast_cache(
            records=records,
            normalization=norm,
            engine=engine,
            output_dir=out,
            bindings={**bindings, "decoder_sha256": "b" * 64},
            resume=True,
        )
