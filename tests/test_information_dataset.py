import json

import numpy as np
import pytest

from latency_meta_mdp.meta.information.dataset import load_episode, save_episode
from latency_meta_mdp.meta.information.records import StageRecorder
from latency_meta_mdp.meta.information.schema import ActorPacket


def test_roundtrip_failure_episode_and_binding_tamper_rejected(tmp_path):
    recorder = StageRecorder("failure")
    tokens = np.ones((2, 196, 384))
    recorder.start(
        ActorPacket(10, "G", current_tokens=tokens, current_tick=9),
        0,
        forced=False,
        probability=0.2,
    )
    recorder.add_reward(10, 0)
    recorder.start(
        ActorPacket(11, "G", current_tokens=tokens, current_tick=9),
        5,
        forced=False,
        probability=0.3,
    )
    recorder.add_reward(11, 0)
    recorder.finish(12, terminated=True)
    path = tmp_path / "episode"
    binding = {"mode": "bound", "asset": "x"}
    save_episode(path, recorder, report={"parcels": {"successes": 0}}, binding=binding)
    result = load_episode(path, binding=binding)
    assert len(result) == 2 and sum(t["task_reward"] for t in result) == 0
    np.testing.assert_array_equal(result[0]["packet"].current_tokens, tokens)
    assert not result[0]["packet"].current_tokens.flags.writeable
    with pytest.raises(ValueError, match="binding"):
        load_episode(path, binding={"mode": "split"})
    with pytest.raises(FileExistsError):
        save_episode(path, recorder, report={}, binding=binding)
    metadata = json.loads((path / "records.json").read_text())
    assert metadata["visual_arrays"] == 1
    (path / "records.json").write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        load_episode(path, binding=binding)


def test_repeated_visual_references_are_decompressed_once(tmp_path, monkeypatch):
    from numpy.lib.npyio import NpzFile

    r = StageRecorder("reuse")
    for tick in (10, 11):
        r.start(
            ActorPacket(tick, "G", current_tick=9, current_tokens=np.ones((2, 196, 384))),
            0,
            forced=False,
            probability=1.0,
        )
        r.add_reward(tick, 0)
    r.finish(12, terminated=True)
    save_episode(tmp_path / "episode", r, report={}, binding={})
    calls = []
    original = NpzFile.__getitem__

    def counted(self, name):
        calls.append(name)
        return original(self, name)

    monkeypatch.setattr(NpzFile, "__getitem__", counted)
    result = load_episode(tmp_path / "episode", binding={})
    assert len(result) == 2
    assert calls == ["visual_0"]


def test_visual_archive_preserves_exact_float_values_and_numpy_compatibility(tmp_path):
    from latency_meta_mdp.meta.information.dataset import save_visual_archive

    rng = np.random.default_rng(17)
    arrays = dict(
        visual_0=rng.standard_normal((2, 196, 384)).astype(np.float32),
        visual_1=np.array([0.0, -0.0, np.inf, -np.inf], dtype=np.float32),
    )
    path = tmp_path / "visual.npz"
    save_visual_archive(path, arrays)
    with np.load(path, allow_pickle=False) as store:
        assert set(store.files) == set(arrays)
        for key, value in arrays.items():
            assert store[key].dtype == value.dtype
            assert store[key].tobytes() == value.tobytes()
    with pytest.raises(FileExistsError):
        save_visual_archive(path, arrays)
    with pytest.raises(ValueError, match="Object arrays"):
        save_visual_archive(tmp_path / "object.npz", dict(unsafe=np.array([{}], dtype=object)))
