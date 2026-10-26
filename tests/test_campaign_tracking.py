import json

import pytest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def event(path, sequence, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(dict(sequence=sequence, event=values)) + "\n")


def test_scratch_startup_precedes_online_on_one_update_axis(tmp_path):
    from latency_meta_mdp.meta.campaign_tracking import snapshot

    event(
        tmp_path / "startup/events.jsonl",
        1,
        phase="fit",
        iteration=1,
        actual_update=10,
        loss=2,
        grad_norm=0.3,
    )
    first, status = snapshot(tmp_path)
    assert status["total_updates"] == 10
    event(
        tmp_path / "cycle-0001/fit/events.jsonl",
        1,
        phase="fit",
        iteration=1,
        actual_update=2,
        loss=1,
        grad_norm=0.2,
    )
    records, status = snapshot(tmp_path)
    assert records[:1] == first
    assert [r["metrics"]["train/total_updates"] for r in records] == [10, 12]
    assert status["total_updates"] == 12


def test_campaign_axes_continue_across_cycles_and_keep_reward_types_separate(tmp_path):
    from latency_meta_mdp.meta.campaign_tracking import snapshot

    for cycle in (1, 2):
        folder = tmp_path / f"cycle-{cycle:04d}"
        write(
            folder / "collection/rollout-summary.json",
            dict(
                completed=1,
                expected=1,
                episodes=[
                    dict(
                        episode=dict(episode_id=f"e{cycle}", regime="zero", behavior="fqi"),
                        report=dict(
                            parcels=dict(successes=3, spawned=10),
                            budget=dict(
                                episode_reward=18,
                                compute_cost=400,
                                drops=2,
                                module_calls=dict(vla=200),
                            ),
                        ),
                    )
                ],
            ),
        )
        path = folder / "fit/events.jsonl"
        event(path, 1, phase="fit", iteration=1, actual_update=1, loss=2, grad_norm=0.2)
        event(
            path,
            2,
            phase="iteration_complete",
            iteration=1,
            actual_update=1,
            metrics=dict(td_loss=0.3, diagnostics={"startup/G": dict(taken_q_mean=10)}),
        )
        event(
            path,
            3,
            phase="validation_complete",
            iteration=1,
            actual_update=1,
            metrics={"zero/reward": 0.4, "zero/budget_return": 21, "zero/compute_cost": 300},
        )
    records, status = snapshot(tmp_path)
    updates = [
        r["metrics"]["train/total_updates"]
        for r in records
        if "train/total_updates" in r["metrics"]
    ]
    assert updates == [1, 2]
    vals = [r["metrics"] for r in records if "validation/cycle" in r["metrics"]]
    assert [v["validation/cycle"] for v in vals] == [1, 2]
    assert vals[0]["validation/zero/delivery_ratio"] == 0.4
    assert vals[0]["validation/zero/budget_return"] == 21
    assert records[0]["metrics"]["collection/zero/fqi/delivery_ratio"] == 0.3
    assert records[0]["metrics"]["collection/zero/fqi/calls/vla"] == 200
    assert status["total_updates"] == 2
    assert status["total_episodes"] == 2
    assert any("fqi/diagnostics/startup/G/taken_q_mean" in r["metrics"] for r in records)


def test_resume_backfill_is_append_only_and_never_truncates_writer_tail(tmp_path):
    from latency_meta_mdp.meta.campaign_tracking import pending_records, snapshot

    path = tmp_path / "cycle-0001/fit/events.jsonl"
    event(path, 1, phase="fit", iteration=1, actual_update=1, loss=2, grad_norm=0.2)
    first, _ = snapshot(tmp_path)
    event(path, 2, phase="fit", iteration=1, actual_update=2, loss=1, grad_norm=0.2)
    with path.open("ab") as stream:
        stream.write(b'{"sequence":3')
    records, _ = snapshot(tmp_path)
    assert [step for step, _ in pending_records(records, 1, first[-1]["digest"])] == [1]
    assert path.read_bytes().endswith(b'{"sequence":3')
    with pytest.raises(ValueError, match="prefix"):
        list(pending_records(records, 1, "changed"))
    with pytest.raises(ValueError, match="shorter"):
        list(pending_records(records, 3, "unused"))


def test_sampling_metrics_append_without_changing_old_records(tmp_path):
    from latency_meta_mdp.meta.campaign_tracking import console_line, snapshot

    path = tmp_path / "cycle-0001/fit/events.jsonl"
    event(path, 1, phase="iteration_complete", iteration=1, metrics={"loss": 1.0})
    before, _ = snapshot(tmp_path)
    event(
        path,
        2,
        phase="iteration_complete",
        iteration=2,
        metrics={
            "sampling": dict(
                target_online_fraction=0.2,
                effective_online_fraction=0.19,
                groups={"latest": dict(draws=300, unique=200, probability=0.09)},
            )
        },
    )
    after, _ = snapshot(tmp_path)
    assert before[0] == after[0]
    assert after[1]["metrics"]["fqi/sampling/latest/unique"] == 200
    write(
        tmp_path / "cycle-0002/collection/parallel-progress.json",
        dict(
            completed=6,
            expected=24,
            per_worker=[2, 1, 2, 1],
            exit_codes=[None] * 4,
            wall_seconds=120,
        ),
    )
    _, status = snapshot(tmp_path)
    assert status["rollout_workers"] == 4
    assert status["rollout_episodes_per_hour"] == 180
    assert "workers=4" in console_line(status)


def test_target_progress_is_visible_before_optimizer_updates(tmp_path):
    from latency_meta_mdp.meta.campaign_tracking import snapshot

    path = tmp_path / "cycle-0001/fit/events.jsonl"
    event(
        path,
        1,
        phase="target_progress",
        iteration=1,
        actual_update=0,
        completed_decisions=64,
        unique_decisions=2000,
        seconds=12,
    )
    _, status = snapshot(tmp_path)
    assert status["phase"] == "target_progress"
    assert status["target_completed"] == 64
    assert status["target_total"] == 2000
    assert status["total_updates"] == 0


def test_parallel_collection_resume_cannot_reorder_published_points(tmp_path):
    from latency_meta_mdp.meta.campaign_tracking import pending_records, snapshot

    folder = tmp_path / "cycle-0001/collection"
    episodes = [dict(episode_id=str(i), regime="zero", behavior="fqi") for i in range(3)]
    rows = [dict(episode=ep, report=dict(parcels=dict(successes=1, spawned=10))) for ep in episodes]
    write(folder / "cohort.json", dict(episodes=episodes))
    write(folder / "rollout-summary.json", dict(episodes=[rows[2], rows[0]]))
    first, status = snapshot(tmp_path)
    assert len(first) == 1 and status["total_episodes"] == 2
    write(folder / "rollout-summary.json", dict(episodes=rows))
    resumed, _ = snapshot(tmp_path)
    assert len(list(pending_records(resumed, 1, first[0]["digest"]))) == 2


def test_stage_wandb_can_be_disabled_without_losing_fit_events(tmp_path, monkeypatch):
    from latency_meta_mdp.meta import tracking_transport
    from latency_meta_mdp.meta.online_tracking import OnlineFitTracker

    monkeypatch.setenv("CONVEYOR_STAGE_WANDB", "0")
    monkeypatch.setattr(
        tracking_transport, "start_run", lambda *a, **k: pytest.fail("stage run created")
    )
    tracker = OnlineFitTracker(tmp_path, binding={}, episodes=1, epochs=1)
    tracker(dict(phase="fit", iteration=1, optimizer_update=1, loss=1, grad_norm=0.2))
    tracker.close(success=True)
    assert not (tmp_path / "wandb-run.json").exists()
    assert json.loads((tmp_path / "events.jsonl").read_text())["event"]["actual_update"] == 1


def test_stage_wandb_can_be_disabled_for_collection(tmp_path, monkeypatch):
    from latency_meta_mdp.meta import tracking_transport
    from latency_meta_mdp.meta.cycle_tracking import CycleTracker

    write(tmp_path / "cohort.json", dict(binding=dict(phase="online"), episodes=[]))
    monkeypatch.setenv("CONVEYOR_STAGE_WANDB", "0")
    monkeypatch.setattr(
        tracking_transport, "start_run", lambda *a, **k: pytest.fail("stage run created")
    )
    tracker = CycleTracker(tmp_path, name="collection")
    assert tracker.run is None
    assert not (tmp_path / "wandb-run.json").exists()


def test_optional_monitor_completion_allows_hold_and_uses_budget_selection(tmp_path):
    from latency_meta_mdp.meta.cycle_tracking import CycleTracker

    tracker = CycleTracker.__new__(CycleTracker)
    tracker.directory, tracker.phase, tracker.run = tmp_path, "monitor", None
    tracker.rows = [
        dict(
            episode=dict(scene=scene, regime=regime, behavior="fqi", epsilon=0,
                         split="validation"),
            report=dict(
                task_reward=0.0, terminated=True, truncated=False, starvation_ticks=100,
                parcels=dict(spawned=10, successes=0, misses=10, timeouts=0, active=0),
                budget=dict(reward_spec_hash="same", deliveries=0, drops=10, timeouts=0,
                            episode_reward=-50.1, compute_cost=10, module_calls={}),
            ),
        )
        for scene in (1, 2, 3) for regime in ("zero", "nominal", "family")
    ]
    tracker.manifest = dict(
        binding=dict(schema_version=3, actions=dict(starvation_mode="allow_hold")),
        episodes=[r["episode"] for r in tracker.rows],
    )
    tracker.close(success=True)
    result = json.loads((tmp_path / "evaluation.json").read_text())
    assert result["metrics"]["fqi/selection_score"] == pytest.approx(-50.1)
    assert result["metrics"]["fqi/zero/reward"] == 0


def test_previous_validation_does_not_hide_new_target_progress(tmp_path):
    from latency_meta_mdp.meta.campaign_tracking import snapshot

    folder = tmp_path / "cycle-0001/fit"
    write(
        folder / "validation/iteration-0001/parallel-progress.json", dict(completed=9, expected=9)
    )
    event(
        folder / "events.jsonl",
        1,
        phase="target_progress",
        iteration=2,
        actual_update=500,
        completed_decisions=0,
        unique_decisions=30000,
    )
    _, status = snapshot(tmp_path)
    assert status["phase"] == "target_progress"


@pytest.mark.parametrize("publish_error", [None, False, True])
def test_supervisor_drains_child_and_resumes_one_run_without_duplicate_points(
    tmp_path, monkeypatch, publish_error
):
    import sys
    from types import SimpleNamespace

    from latency_meta_mdp.meta.campaign_tracking import main

    root, ops = tmp_path / "run", tmp_path / "ops"
    ops.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text("budget: true\n")
    received, identities, syncs = [], [], []
    expected_exit = 75 if publish_error else 0

    class Run:
        url = "https://wandb.ai/test/project/runs/fixed"
        summary = {}
        step = 0

        def define_metric(self, *args, **kwargs):
            pass

        def log(self, values, *, step, commit):
            assert step == self.step
            received.append(values)
            self.summary.update(values)
            self.step += 1

        def finish(self, *, exit_code):
            assert exit_code == expected_exit

    run = Run()

    def init(**kwargs):
        assert kwargs["settings"]["console"] == "wrap"
        assert kwargs["settings"]["x_disable_stats"] is False
        assert kwargs["settings"]["x_stats_sampling_interval"] == 5
        identities.append((kwargs["id"], kwargs["resume"]))
        return run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(
        init=init, Settings=lambda **k: k, Table=lambda **k: k))
    command = tmp_path / "child.py"
    command.write_text(
        "import json, os, pathlib, time\n"
        "assert os.environ['CONVEYOR_STAGE_WANDB'] == '0'\n"
        "time.sleep(0.03)\n"
        f"p = pathlib.Path({str(root / 'cycle-0001/fit/events.jsonl')!r})\n"
        "p.parent.mkdir(parents=True, exist_ok=True)\n"
        "p.write_text(json.dumps(dict(sequence=1, event=dict(phase='fit', iteration=1, "
        "actual_update=1, loss=2, grad_norm=0.1))) + '\\n')\n"
    )
    args = [
        "supervisor",
        "--root",
        str(root),
        "--operations",
        str(ops),
        "--config",
        str(config),
        "--project",
        "test",
        "--name",
        "test",
        "--poll-seconds",
        "0.01",
    ]
    if publish_error is not None:
        from latency_meta_mdp.io import fqi_publication
        recipe = tmp_path / 'recipe.json'
        recipe.write_text('{}')
        args += ['--recipe', str(recipe), '--checkpoint-repo', 'owner/model']
        def sync(*a, **k):
            import time
            time.sleep(0.08)  # The periodic pass can outlive the training child.
            syncs.append((root / 'cycle-0001/fit/events.jsonl').exists())
            return dict(uploaded=0, pending=1 if publish_error else 0,
                        error='ConnectionError' if publish_error else None,
                        index_pending=publish_error, checkpoints=[])
        monkeypatch.setattr(fqi_publication, 'sync_checkpoints', sync)
    monkeypatch.setattr(sys, "argv", args + ["--command", sys.executable, str(command)])
    assert main() == expected_exit
    assert len(received) == 1
    monkeypatch.setattr(sys, "argv", args + ["--once"])
    assert main() == expected_exit
    assert len(received) == 1
    assert identities[0][0] == identities[1][0]
    assert identities[1][1] == "must"
    if publish_error is not None:
        assert len(syncs) >= 3 and syncs[-2:] == [True, True]


def test_preparation_status_is_current_and_distinct_from_prepack(tmp_path):
    import json

    from latency_meta_mdp.meta.campaign_tracking import console_line, snapshot

    fit = tmp_path / "cycle-0001/fit"
    fit.mkdir(parents=True)
    (fit / "preparation-progress.json").write_text(
        json.dumps(dict(phase="prepare_replay", episodes=149, expected=472, elapsed_seconds=2))
    )
    _, status = snapshot(tmp_path)
    assert status["phase"] == "prepare_replay"
    assert status["replay_completed"] == 149
    assert "replay=149/472" in console_line(status)
    (fit / "events.jsonl").write_text(
        json.dumps(
            dict(
                sequence=1,
                event=dict(phase="fit", iteration=1, actual_update=5, loss=1, grad_norm=0.1),
            )
        )
        + "\n"
    )
    _, status = snapshot(tmp_path)
    assert status["phase"] == "fit"


def test_training_console_contains_iteration_loss_breakdown_and_throughput(tmp_path):
    from latency_meta_mdp.meta.campaign_tracking import console_line, snapshot

    path = tmp_path / "cycle-0001/fit/events.jsonl"
    event(
        path,
        1,
        phase="fit",
        iteration=2,
        actual_update=510,
        optimizer_update=10,
        optimizer_updates=500,
        loss=1.2,
        td_loss=0.2,
        weighted_conservative_loss=1.0,
        grad_norm=0.4,
        learning_rate=1e-4,
        input_seconds=0.01,
        update_seconds=0.1,
        updates_per_second=10.0,
    )
    records, status = snapshot(tmp_path)
    line = console_line(status)
    for expected in (
        "iteration=2",
        "step=10/500",
        "td=0.20000",
        "cql=1.00000",
        "lr=0.0001",
        "updates/s=10.00",
        "input=10.0ms",
    ):
        assert expected in line
    assert records[0]["metrics"]["train/input_seconds"] == 0.01


def test_console_distinguishes_active_workers_from_configured_slots():
    from latency_meta_mdp.meta.campaign_tracking import console_line, rollout_status

    status = rollout_status(
        dict(
            per_worker=[3, 2, 1, 3],
            exit_codes=[0, None, None, 0],
            completed=9,
            wall_seconds=60,
        )
    )
    assert status["rollout_workers"] == 4
    assert status["rollout_active_workers"] == 2
    line = console_line(dict(status, phase="validation", cycle=2))
    assert "workers=2/4" in line
