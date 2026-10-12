import pytest

from latency_meta_mdp.meta.information.records import StageRecorder, decision_view
from latency_meta_mdp.meta.information.schema import ActorPacket


def test_forced_stage_folds_cost_and_physical_reward_once():
    r = StageRecorder("episode")
    r.start(ActorPacket(10, "G"), 5, forced=False, probability=0.5)
    r.add_cost("encode", "dino", 0.002)
    r.start(ActorPacket(10, "O"), 6, forced=True, probability=1)
    r.add_cost("predict", "jepa", 0.003)
    r.start(ActorPacket(10, "P"), 7, forced=False, probability=0.5)
    r.add_reward(10, 2)
    r.add_reward(11, 1)
    r.finish(12, terminated=True)
    transitions = decision_view(r.rows)
    assert len(transitions) == 2
    assert transitions[0]["cost"] == pytest.approx(0.005)
    assert transitions[0]["duration"] == 0
    assert transitions[1]["task_reward"] == pytest.approx(0.3)
    assert transitions[1]["terminated"]
    assert r.summary()["task_reward"] == pytest.approx(0.3)


def test_stage_cycles_duplicate_cost_and_reward_are_rejected():
    r = StageRecorder("episode")
    r.start(ActorPacket(10, "G"), 5, forced=False, probability=1)
    r.add_cost("same", "dino", 0.01)
    with pytest.raises(ValueError, match="operation"):
        r.add_cost("same", "dino", 0.01)
    r.start(ActorPacket(10, "O"), 6, forced=True, probability=1)
    with pytest.raises(ValueError, match="stage"):
        r.start(ActorPacket(10, "G"), 5, forced=False, probability=1)
    r.add_reward(10, 0)
    with pytest.raises(ValueError, match="boundary"):
        r.add_reward(10, 0)


def test_external_truncation_is_not_a_failed_terminal_dataset():
    r = StageRecorder("episode")
    r.start(ActorPacket(10, "G"), 0, forced=False, probability=1)
    r.add_reward(10, 0)
    r.finish(11, terminated=False)
    with pytest.raises(ValueError, match="incomplete"):
        decision_view(r.rows)


def test_nested_operation_timers_are_exclusive_and_have_unique_ids(monkeypatch):
    import latency_meta_mdp.meta.information.records as module

    times = iter([0.0, 1.0, 4.0, 5.0])
    monkeypatch.setattr(module, "perf_counter", lambda: next(times))
    recorder = StageRecorder("nested")
    with recorder.measure("physics"):
        with recorder.measure("capture"):
            pass
    assert recorder.summary()["components"] == {"capture": 3.0, "physics": 2.0}
    assert recorder.summary()["measured_seconds"] == 5.0
    assert len({op["operation_id"] for op in recorder.operations}) == 2
