import pytest


def test_online_metrics_have_explicit_axes_and_loss_sources_without_fake_reward():
    from latency_meta_mdp.meta.online_tracking import fit_metrics

    event = dict(
        phase="fit",
        iteration=2,
        epoch=1,
        episode=3,
        loss=0.02,
        huber_loss=0.01,
        conservative_loss=1.0,
        source="online",
    )
    values = fit_metrics(event, episodes=160, epochs=2)
    assert values["train/update"] == 323
    assert values["train/global_epoch"] == 3
    assert values["train/online/huber_loss"] == 0.01
    assert not any("reward" in k for k in values)
    with pytest.raises(ValueError):
        fit_metrics({**event, "source": "validation"}, episodes=160, epochs=2)


def test_incomplete_event_tail_is_preserved_and_replay_continues(tmp_path):
    import json

    from latency_meta_mdp.meta.online_tracking import recover_events

    path = tmp_path / "events.jsonl"
    path.write_bytes(b'{"sequence":1,"event":{}}\n{"sequence":2')
    assert recover_events(path) == [{"sequence": 1, "event": {}}]
    assert path.read_bytes() == b'{"sequence":1,"event":{}}\n'
    assert list(tmp_path.glob("events.jsonl.partial-*"))
    with path.open("a") as f:
        f.write(json.dumps({"sequence": 2, "event": {}}) + "\n")
    assert len(recover_events(path)) == 2


def test_wandb_failure_does_not_interrupt_local_work(tmp_path):
    from latency_meta_mdp.meta.tracking_transport import SafeRun

    class Broken:
        def log(self, *args, **kwargs):
            raise ConnectionError("test")

    run = SafeRun(Broken(), tmp_path)
    run.log({"loss": 1})
    assert (tmp_path / "wandb-errors.jsonl").exists()


def test_restarted_fit_uses_actual_work_axis_and_explicit_attempt():
    from latency_meta_mdp.meta.online_tracking import fit_metrics

    e = dict(
        phase="fit",
        iteration=2,
        epoch=1,
        episode=1,
        loss=0.1,
        huber_loss=0.1,
        conservative_loss=0.0,
        source="online",
        actual_update=999,
        attempt=2,
        resume_iteration=1,
    )
    metrics = fit_metrics(e, episodes=160, epochs=2)
    assert metrics["train/update"] == 999
    assert metrics["train/attempt"] == 2
    assert metrics["train/resume_iteration"] == 1


def test_iteration_logs_replay_cache_and_phase_timings():
    from latency_meta_mdp.meta.online_tracking import fit_metrics

    metrics = fit_metrics(
        dict(
            phase="iteration_complete",
            iteration=1,
            metrics=dict(
                target_seconds=3.0,
                regression_seconds=4.0,
                replay_cache=dict(hits=12, misses=4, bytes_used=1024),
            ),
        ),
        episodes=4,
        epochs=2,
    )
    assert metrics["fqi/target_seconds"] == 3.0
    assert metrics["fqi/replay_cache/hits"] == 12
    assert metrics["fqi/replay_cache/bytes_used"] == 1024


def test_distributed_minibatch_counts_one_update_and_reports_source_contributions():
    from latency_meta_mdp.meta.online_tracking import fit_metrics

    event = dict(
        phase="fit",
        iteration=1,
        epoch=2,
        episode=40,
        actual_update=80,
        loss=0.3,
        huber_loss=0.2,
        conservative_loss=10.0,
        source="mixed",
        global_batch=4,
        source_losses={"offline": {"combined_loss": 0.1}, "online": {"combined_loss": 0.2}},
    )
    values = fit_metrics(event, episodes=40, epochs=2)
    assert values["train/update"] == 80
    assert values["train/global_epoch"] == 2
    assert values["train/offline/combined_loss"] == 0.1
    assert values["train/online/combined_loss"] == 0.2


def test_interrupted_iteration_events_are_not_all_labelled_discarded(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from latency_meta_mdp.meta import tracking_transport
    from latency_meta_mdp.meta.online_tracking import OnlineFitTracker

    # The W&B boundary is replaced; event parsing and recovery bookkeeping are real.
    remote = SimpleNamespace(
        url="offline-test",
        step=0,
        summary={},
        define_metric=lambda *a, **k: None,
        log=lambda *a, **k: None,
    )
    monkeypatch.setattr(tracking_transport, "start_run", lambda *a, **k: remote)
    entries = [
        dict(
            sequence=i,
            event=dict(
                phase="fit",
                iteration=1,
                optimizer_update=i,
                loss=0.1,
                grad_norm=0.1,
                actual_update=i,
                attempt=1,
            ),
        )
        for i in (1, 2, 3)
    ]
    (tmp_path / "events.jsonl").write_text("".join(json.dumps(x) + "\n" for x in entries))
    tracker = OnlineFitTracker(tmp_path, binding={}, episodes=4, epochs=1)
    assert tracker.actual_update == 3
    # Some of these steps can be retained by periodic latest, so only interruption is known.
    assert "recovery/discarded_events" not in remote.summary
    assert len(remote.summary["recovery/interrupted_events"]) == 3
    assert (tmp_path / "interrupted-before-attempt-2.json").exists()
