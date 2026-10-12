import numpy as np
import pytest

from latency_meta_mdp.meta.information.dataset import load_episode, save_episode
from latency_meta_mdp.meta.information.records import StageRecorder
from latency_meta_mdp.meta.information.schema import ActorPacket


def test_prepared_cache_preserves_values_and_rejects_source_and_cache_changes(tmp_path):
    from latency_meta_mdp.meta.information.replay_preparation import prepare_episode

    r = StageRecorder("sample")
    tokens = np.random.default_rng(12).normal(size=(2, 196, 384)).astype(np.float32)
    r.start(
        ActorPacket(10, "G", current_tokens=tokens, current_tick=10),
        5,
        forced=False,
        probability=1.0,
    )
    r.add_reward(10, 1)
    r.finish(11, terminated=True)
    source = tmp_path / "source"
    save_episode(source, r, report={}, binding={"x": 1})
    target = prepare_episode(source, tmp_path / "cache")
    a = load_episode(source, binding={"x": 1})
    b = load_episode(target, binding={"x": 1})
    assert b[0]["task_reward"] == a[0]["task_reward"] == 0.1
    assert b[0]["packet"].current_tokens.tobytes() == tokens.tobytes()
    assert prepare_episode(source, tmp_path / "cache") == target
    (target / "records.json").write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        prepare_episode(source, tmp_path / "cache")
    (source / "records.json").write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        prepare_episode(source, tmp_path / "other")


def test_preparation_rejects_changed_admitted_receipt(tmp_path):
    from latency_meta_mdp.meta.information.replay_preparation import prepare_replay

    class Admitted:
        entries = [(tmp_path / "source", {"mode": "bound"}, "stale-admission-hash")]
        cache = None

    p = tmp_path / "source"
    p.mkdir()
    import json

    from latency_meta_mdp.meta.information.config import sha256

    np.savez(p / "visual.npz")
    (p / "records.json").write_text("{}")
    (p / "audit.json").write_text("{}")
    (p / "complete.json").write_text(
        json.dumps(
            dict(
                binding={"mode": "bound"},
                files={n: sha256(p / n) for n in ("visual.npz", "records.json", "audit.json")},
            )
        )
    )
    with pytest.raises(ValueError, match="binding"):
        prepare_replay(Admitted(), tmp_path / "cache")


def test_preparation_rejects_source_changed_during_copy(tmp_path, monkeypatch):
    from latency_meta_mdp.meta.information import replay_preparation as module

    recorder = StageRecorder("sample")
    recorder.start(ActorPacket(10, "G"), 5, forced=False, probability=1.0)
    recorder.finish(11, terminated=True)
    source = tmp_path / "source"
    save_episode(source, recorder, report={}, binding={})
    copy = module.shutil.copy2

    def change_then_copy(src, dst):
        if src.name == "records.json":
            src.write_text("{}")
        return copy(src, dst)

    monkeypatch.setattr(module.shutil, "copy2", change_then_copy)
    with pytest.raises(ValueError, match="changed|hash"):
        module.prepare_episode(source, tmp_path / "cache")
    assert not list((tmp_path / "cache").glob("*/complete.json"))
