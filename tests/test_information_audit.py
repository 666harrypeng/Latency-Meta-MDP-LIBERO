import pytest

from latency_meta_mdp.envs.conveyor.meta_audit import audit_episode
from latency_meta_mdp.meta.information.records import StageRecorder
from latency_meta_mdp.meta.information.schema import ActorPacket


def test_reward_outcomes_and_cost_are_conserved_including_failures():
    recorder = StageRecorder("audit")
    recorder.start(ActorPacket(10, "G"), 0, forced=False, probability=1.0)
    recorder.add_cost("encode", "dino", 0.01)
    recorder.add_reward(10, 1)
    recorder.finish(11, terminated=True)
    report = dict(
        terminated=True,
        truncated=False,
        parcel_events=[
            {"parcel_id": 0, "event": "spawn", "tick": 0},
            {"parcel_id": 1, "event": "spawn", "tick": 1},
            {"parcel_id": 0, "event": "success", "tick": 10},
            {"parcel_id": 1, "event": "miss", "tick": 11},
        ],
    )
    audit = audit_episode(recorder, report)
    assert audit["task_reward"] == 0.1
    assert audit["failures"][0]["failure_mode"] == "unknown"
    assert audit["free_decision_cost"] == 0.01
    with pytest.raises(ValueError, match="outcome"):
        audit_episode(
            recorder,
            {**report, "parcel_events": report["parcel_events"] + [report["parcel_events"][-1]]},
        )
    recorder.rows[0]["task_reward"] = 0.2
    with pytest.raises(ValueError, match="reward"):
        audit_episode(recorder, report)


def test_progress_events_keep_regrasp_attempts_without_changing_reward(tmp_path):
    from latency_meta_mdp.envs.conveyor.meta_audit import ProgressAudit

    audit = ProgressAudit(
        tmp_path / "truth.jsonl",
        goal_center=[0, 0, 1],
        goal_radius=0.2,
        minimum_z=0.8,
        grasp_ticks=2,
    )

    def state(grasped, z):
        return [dict(parcel_id=0, position=[0, 0, z], grasped=grasped)]

    for tick, row in enumerate(
        [
            state(False, 0.1),
            state(True, 0.5),
            state(True, 0.9),
            state(False, 0.5),
            state(True, 0.9),
            state(True, 0.9),
        ]
    ):
        audit.record(tick, row, {0: "active"})
    events = [r["event"] for r in audit.events]
    assert events.count("grasp_started") == 2
    assert events.count("grasp_sustained") == 2
    assert events.count("grasp_lost") == 1
    assert events.count("lift_threshold_crossed") == 2
    assert len((tmp_path / "truth.jsonl").read_text().splitlines()) == 6
