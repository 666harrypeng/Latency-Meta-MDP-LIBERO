import pytest

from latency_meta_mdp.meta.information.config import sha256


def test_cycle_manifests_keep_online_eval_and_diagnostic_disjoint(tmp_path):
    from latency_meta_mdp.meta.information.cycle import build_cycle_manifest

    source = tmp_path / "audit-source.json"
    source.write_text("{}")
    config = dict(
        mode="bound",
        assets={"identity": {}},
        splits={"train": list(range(100, 164)), "validation": list(range(200, 220)), "test": [300]},
        excluded_scenes=[0],
    )
    audit = dict(
        local_inventory_pass=True,
        candidate_splits=config["splits"],
        excluded_scenes=[0],
        sources=[dict(path=source.name, sha256=sha256(source))],
    )
    policy = dict(checkpoint="pinned", receipt_sha256="receipt", model_hash="model")
    kwargs = dict(root=tmp_path, code_hash="code", policy=policy)
    online = build_cycle_manifest(config, audit, phase="online", **kwargs)
    assert len(online["episodes"]) == 32
    learned = [e for e in online["episodes"] if e["behavior"] == "fqi"]
    assert len(learned) == 24 and all(e["epsilon"] == 0.2 for e in learned)
    assert {e["scene"] for e in online["episodes"]} == set(range(116, 132))
    assert {e["split"] for e in online["episodes"]} == {"train"}
    evaluation = build_cycle_manifest(config, audit, phase="validation", **kwargs)
    assert len(evaluation["episodes"]) == 24
    assert all(e["epsilon"] == 0 and e["split"] == "validation" for e in evaluation["episodes"])
    smoke = build_cycle_manifest(config, audit, phase="smoke", **kwargs)
    assert len(smoke["episodes"]) == 4
    assert smoke["binding"]["scope"] == "diagnostic"
    assert online["binding"]["scope"] == "online_training"
    assert not {e["scene"] for e in evaluation["episodes"]} & {
        e["scene"] for e in online["episodes"]
    }
    with pytest.raises(ValueError):
        build_cycle_manifest(config, audit, phase="test", **kwargs)


def test_custom_scheduler_cannot_be_mislabeled_as_a_rule():
    from latency_meta_mdp.meta.information.collection import resolve_scheduler
    from latency_meta_mdp.meta.information.fqi import FqiScheduler
    from latency_meta_mdp.meta.information.model import StageQ

    scheduler = FqiScheduler(StageQ())
    assert resolve_scheduler("fqi", 3, scheduler) is scheduler
    with pytest.raises(ValueError):
        resolve_scheduler("periodic", 3, scheduler)
    with pytest.raises(ValueError):
        resolve_scheduler("fqi", 3, None)
    assert resolve_scheduler("periodic", 3, None).component == "rule_scheduler"


def test_rollout_metrics_separate_evaluation_and_exploration():
    from latency_meta_mdp.meta.cycle_tracking import rollout_metrics, summarize_rollouts

    episode = dict(scene=11000, regime="nominal", behavior="fqi", epsilon=0.0)
    report = dict(
        task_reward=0.3,
        parcels=dict(spawned=10, successes=3, misses=6, timeouts=1),
        executed_steps=2200,
        starvation_ticks=0,
        components={"vla": 2.0, "jepa": 1.0},
    )
    values = rollout_metrics(episode, report, phase="validation", completed=1)
    assert values["eval/fqi/nominal/task_return"] == 0.3
    assert values["eval/fqi/nominal/seconds/vla"] == 2.0
    assert not any(k.startswith("collection/") for k in values)
    online = rollout_metrics({**episode, "epsilon": 0.2}, report, phase="online", completed=1)
    assert online["collection/fqi/nominal/task_return"] == 0.3
    bad = {**episode, "epsilon": 0.2}
    with pytest.raises(ValueError, match="greedy"):
        rollout_metrics(bad, report, phase="validation", completed=1)
    rows = [dict(episode=episode, report=report, video="v.mp4")]
    summary = summarize_rollouts(rows)
    assert summary["fqi/nominal"]["successes"] == 3
    assert summary["fqi/nominal"]["episodes"] == 1
    assert summary["fqi/nominal"]["success_rate"] == 0.3


def test_paired_evaluation_keeps_scene_clusters_and_requires_complete_pairs():
    from latency_meta_mdp.meta.information.evaluation import paired_summary

    rows = []
    for scene, a, b in [(1, 0.2, 0.1), (2, 0.4, 0.2)]:
        for regime in ["nominal", "family"]:
            for policy, reward in [("fqi", a), ("periodic", b)]:
                rows.append(
                    dict(
                        episode=dict(scene=scene, regime=regime, behavior=policy),
                        report=dict(task_reward=reward),
                    )
                )
    result = paired_summary(rows, reference="periodic", repeats=100, seed=1)
    assert result["paired_scenes"] == 2
    assert result["mean_difference"] == pytest.approx(0.15)
    assert result["wins"] == 4 and result["losses"] == 0
    assert result["interval_95"][0] >= 0.1 - 1e-8
    with pytest.raises(ValueError, match="paired"):
        paired_summary(rows[:-1], reference="periodic")


def test_tracker_retains_complete_video_index_without_wandb(tmp_path):
    import json

    from latency_meta_mdp.meta.cycle_tracking import CycleTracker

    manifest = dict(binding=dict(phase="validation"), episodes=[])
    (tmp_path / "cohort.json").write_text(json.dumps(manifest))
    tracker = CycleTracker(tmp_path, name="test", enabled=False)
    episode = dict(scene=1, regime="nominal", behavior="fqi", epsilon=0.0)
    report = dict(
        task_reward=0.1,
        parcels=dict(spawned=10, successes=1, misses=9, timeouts=0),
        executed_steps=2000,
        starvation_ticks=0,
    )
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    (attempt / "smoke.mp4").write_bytes(b"video-placeholder")
    tracker(episode, report, attempt, {})
    contents = json.loads((tmp_path / "rollout-summary.json").read_text())
    assert contents["episodes"][0]["video"] == str(attempt / "smoke.mp4")
    assert str(attempt / "smoke.mp4") in (tmp_path / "videos.md").read_text()
    tracker.close(success=True)


def test_completed_eval_tracker_exports_paired_metrics(tmp_path):
    import json

    from latency_meta_mdp.meta.cycle_tracking import CycleTracker

    definitions = [
        dict(episode_id=r + b, scene=1, regime=r, behavior=b, epsilon=0.0)
        for r in ["nominal", "family"]
        for b in ["fqi", "periodic", "active"]
    ]
    (tmp_path / "cohort.json").write_text(
        json.dumps(dict(binding=dict(phase="validation"), episodes=definitions))
    )
    tracker = CycleTracker(tmp_path, name="test", enabled=False)
    for ep in definitions:
        p = tmp_path / (ep["regime"] + ep["behavior"])
        p.mkdir()
        (p / "smoke.mp4").write_bytes(b"video-placeholder")
        n = 2 if ep["behavior"] == "fqi" else 1
        tracker(
            ep,
            dict(
                task_reward=n / 10,
                parcels=dict(spawned=10, successes=n, misses=10 - n, timeouts=0),
                executed_steps=2000,
                starvation_ticks=0,
            ),
            p,
            {},
        )
    tracker.close(success=True)
    report = json.loads((tmp_path / "evaluation.json").read_text())
    assert report["paired"]["periodic"]["mean_difference"] == pytest.approx(0.1)
    assert report["metrics"]["fqi/mean_task_return"] == pytest.approx(0.2)
