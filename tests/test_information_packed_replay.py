import copy
from dataclasses import fields

import numpy as np
import pytest
import torch

from latency_meta_mdp.meta.information.config import sha256
from latency_meta_mdp.meta.information.dataset import load_episode, save_episode
from latency_meta_mdp.meta.information.records import StageRecorder
from latency_meta_mdp.meta.information.schema import ActorPacket


def source_episode(path):
    tokens = np.random.default_rng(12).normal(size=(2, 196, 384)).astype(np.float32)
    recorder = StageRecorder("sample")
    recorder.start(ActorPacket(10, "G"), 5, forced=False, probability=0.8)
    recorder.start(
        ActorPacket(10, "O", current_tokens=tokens, current_tick=10),
        6,
        forced=True,
        probability=1.0,
    )
    recorder.add_cost("a", "predict", 0.23)
    recorder.start(
        ActorPacket(
            10,
            "P",
            current_tokens=tokens,
            forecast_tokens=tokens,
            current_tick=10,
            forecast_tick=10,
            forecast_buffer_version=2,
            buffer_version=2,
            estimated_delay=7,
            plan_age=3,
            current_proprio=np.arange(16),
            observed_proprio=np.ones(16),
            forecast_proprio=np.full(16, 2),
            buffer=np.ones((50, 7)),
            buffer_mask=np.arange(50) < 35,
            completed_delays=(1, 2, 7),
            previous_action=6,
            elapsed_ticks=2,
        ),
        7,
        forced=False,
        probability=0.7,
    )
    recorder.add_reward(10, 1)
    recorder.finish(11, terminated=True)
    save_episode(path, recorder, report={}, binding={"sample": 1})
    return load_episode(path, binding={"sample": 1})


def prepared(source, cache):
    from latency_meta_mdp.meta.information.packed_replay import PackedReplay, prepare_packed_episode

    entry = prepare_packed_episode(
        source, cache, binding={"sample": 1}, expected_receipt=sha256(source / "complete.json")
    )
    return PackedReplay([entry], sources=["online"])


def test_shared_packed_replay_preserves_all_inputs_and_forced_folding(tmp_path):
    source = tmp_path / "source"
    original = source_episode(source)
    replay = prepared(source, tmp_path / "cache")
    actual = replay[0]
    assert len(actual) == len(original) == 2
    for a, b in zip(original, actual, strict=True):
        for key in a.keys() - {"packet", "next_packet"}:
            assert a[key] == b[key]
        for field in fields(ActorPacket):
            x, y = getattr(a["packet"], field.name), getattr(b["packet"], field.name)
            if isinstance(x, np.ndarray):
                np.testing.assert_array_equal(x, y)
                assert not y.flags.writeable
            else:
                assert x == y
    assert actual[0]["next_packet"] is actual[1]["packet"]
    assert actual[-1]["next_packet"] is None
    assert isinstance(actual[1]["packet"].current_tokens, np.memmap)
    assert np.shares_memory(actual[1]["packet"].current_tokens, actual[1]["packet"].forecast_tokens)
    assert replay[0] is actual
    assert replay.metrics()["hits"] == 1


def test_packed_replay_matches_fitted_targets_and_parameters(tmp_path):
    from latency_meta_mdp.meta.information.fqi import fit_iteration
    from latency_meta_mdp.meta.information.model import StageQ

    source = tmp_path / "source"
    original = source_episode(source)
    replay = prepared(source, tmp_path / "cache")
    torch.manual_seed(31)
    a = StageQ()
    b = copy.deepcopy(a)
    np.random.seed(20)
    first = fit_iteration(a, [original], mode="bound", epochs=1)
    np.random.seed(20)
    second = fit_iteration(b, replay, mode="bound", epochs=1)
    assert first["loss"] == second["loss"]
    assert first["target_min"] == second["target_min"]
    assert first["target_max"] == second["target_max"]
    for name, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize("which", ["source", "packed"])
def test_packed_replay_rejects_changes_after_admission(tmp_path, which):
    source = tmp_path / "source"
    source_episode(source)
    replay = prepared(source, tmp_path / "cache")
    replay[0]
    victim = (
        source / "records.json"
        if which == "source"
        else next((tmp_path / "cache").rglob("records.json"))
    )
    victim.write_text("{}")
    with pytest.raises(ValueError, match="changed|hash"):
        replay[0]
    with pytest.raises(ValueError, match="changed|hash"):
        prepared(source, tmp_path / "cache")


def test_spawn_serializes_only_verified_paths_not_open_mmaps(tmp_path):
    import pickle

    source = tmp_path / "source"
    source_episode(source)
    replay = prepared(source, tmp_path / "cache")
    expected = replay[0][1]["packet"].current_tokens
    serialized = pickle.dumps(replay)
    assert len(serialized) < 10000
    resumed = pickle.loads(serialized)
    np.testing.assert_array_equal(expected, resumed[0][1]["packet"].current_tokens)


def test_distributed_fitter_uses_mmap_without_an_archive_cache(tmp_path):
    from latency_meta_mdp.meta.information.distributed_fqi import fit_distributed
    from latency_meta_mdp.meta.information.model import StageQ

    source = tmp_path / "source"
    source_episode(source)
    replay = prepared(source, tmp_path / "cache")
    metrics = fit_distributed(StageQ(), replay, devices=["cpu"], mode="bound", epochs=1)
    assert metrics["rank_metrics"][0]["cache"]["storage"] == "mmap"
    assert metrics["rank_metrics"][0]["cache"]["hits"] == 1


def test_episode_without_visuals_remains_valid(tmp_path):
    source = tmp_path / "source"
    recorder = StageRecorder("sample")
    recorder.start(ActorPacket(10, "G"), 5, forced=False, probability=1.0)
    recorder.finish(11, terminated=True)
    save_episode(source, recorder, report={}, binding={"sample": 1})
    assert prepared(source, tmp_path / "cache")[0][0]["packet"].current_tokens is None


def test_source_change_during_conversion_is_not_published(tmp_path, monkeypatch):
    from latency_meta_mdp.meta.information import packed_replay as module

    source = tmp_path / "source"
    source_episode(source)
    load = module.load_episode

    def changed_source(*args, **kwargs):
        rows = load(*args, **kwargs)
        (source / "records.json").write_text("{}")
        return rows

    monkeypatch.setattr(module, "load_episode", changed_source)
    with pytest.raises(ValueError, match="changed"):
        prepared(source, tmp_path / "cache")
    assert not list((tmp_path / "cache").rglob("complete.json"))


def test_archive_and_mmap_formats_can_share_explicit_cache_root(tmp_path):
    from latency_meta_mdp.meta.information.replay_preparation import prepare_episode

    source = tmp_path / "source"
    source_episode(source)
    archive = prepare_episode(source, tmp_path / "cache")
    mmap = prepared(source, tmp_path / "cache")
    assert (
        mmap[0][1]["task_reward"] == load_episode(archive, binding={"sample": 1})[1]["task_reward"]
    )
