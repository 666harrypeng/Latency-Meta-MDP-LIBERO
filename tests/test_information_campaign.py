import json

import pytest


def test_journal_reuses_only_verified_completed_phases_and_retries_failures(tmp_path):
    from latency_meta_mdp.meta.information.campaign import CampaignJournal
    from latency_meta_mdp.meta.information.config import sha256

    journal = CampaignJournal(tmp_path, binding={"recipe": "fixed"})
    calls = []
    artifact = tmp_path / "checkpoint.txt"

    def operation():
        calls.append(1)
        artifact.write_text("verified result")
        if len(calls) == 1:
            raise RuntimeError("interrupted after save")
        return dict(artifacts={str(artifact): sha256(artifact)}, value=42)

    with pytest.raises(RuntimeError, match="interrupted"):
        journal.run("cycle-1", operation)
    resumed = CampaignJournal(tmp_path, binding={"recipe": "fixed"})
    assert resumed.run("cycle-1", operation)["value"] == 42
    assert resumed.run("cycle-1", operation)["value"] == 42
    assert len(calls) == 2
    state = json.loads((tmp_path / "campaign-state.json").read_text())
    assert [a["status"] for a in state["phases"]["cycle-1"]["attempts"]] == ["failed", "complete"]
    artifact.write_text("changed")
    with pytest.raises(ValueError, match="hash"):
        resumed.run("cycle-1", operation)
    with pytest.raises(ValueError, match="binding"):
        CampaignJournal(tmp_path, binding={"recipe": "changed"})


def test_stop_rule_requires_a_full_repeat_and_preserves_incumbent_ties():
    from latency_meta_mdp.meta.information.campaign import selection_decision

    def metrics(successes):
        return {"selection_score": successes / 60}

    assert selection_decision(metrics(13), metrics(13), metrics(13), drop_successes=3) == "keep"
    # Equal discrete deliveries can sum to adjacent floating-point representations.
    assert (
        selection_decision(
            metrics(13),
            {"selection_score": 0.21666666666666665},
            {"selection_score": 0.2166666666666667},
            drop_successes=3,
        )
        == "keep"
    )
    assert selection_decision(metrics(13), metrics(13), metrics(14), drop_successes=3) == "best"
    assert selection_decision(metrics(13), metrics(13), metrics(10), drop_successes=3) == "repeat"
    assert (
        selection_decision(
            metrics(13), metrics(13), metrics(10), drop_successes=3, repeat=metrics(10)
        )
        == "stop"
    )
    assert (
        selection_decision(
            metrics(13), metrics(13), metrics(10), drop_successes=3, repeat=metrics(12)
        )
        == "keep"
    )


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from latency_meta_mdp.meta.information import campaign, deployment, online_training, parallel
    from latency_meta_mdp.meta.information.cohort import build_manifest, complete_attempt
    from latency_meta_mdp.meta.information.config import identity_hash, sha256
    from latency_meta_mdp.meta.information.cycle import build_cycle_manifest
    from latency_meta_mdp.meta.information.validation import atomic_json

    config = dict(
        mode="bound",
        assets={"identity": {}},
        excluded_scenes=[],
        splits=dict(train=list(range(100, 164)), validation=[200, 201, 202], test=[300]),
    )
    config_path = tmp_path / "config.json"
    atomic_json(config_path, config)
    audit = dict(
        local_inventory_pass=True, candidate_splits=config["splits"], excluded_scenes=[], sources=[]
    )
    audit_path = tmp_path / "audit.json"
    atomic_json(audit_path, audit)
    monkeypatch.setattr(campaign, "load_config", lambda _: config)

    def identity(path, *args, **kwargs):
        return dict(
            checkpoint=str(path),
            receipt_sha256=sha256(path / "complete.json"),
            model_hash=path.name,
        )

    monkeypatch.setattr(deployment, "load_scheduler", identity)
    monkeypatch.setattr(deployment, "scheduler_identity", lambda value: value)
    for name in ("initial", "behavior"):
        path = tmp_path / name
        path.mkdir()
        (path / "state.pt").write_text(name)
        atomic_json(path / "complete.json", dict(files={"state.pt": sha256(path / "state.pt")}))
    for name, manifest in (
        ("offline", build_manifest(config, audit, root=tmp_path, code_hash="legacy")),
        (
            "online",
            build_cycle_manifest(
                config,
                audit,
                root=tmp_path,
                code_hash="legacy",
                policy=identity(tmp_path / "behavior"),
                phase="online",
            ),
        ),
    ):
        (tmp_path / name).mkdir()
        atomic_json(tmp_path / name / "cohort.json", manifest)
    compatibility = tmp_path / "compatibility.json"
    atomic_json(
        compatibility, dict(config_hash=identity_hash(config), collection_code_hashes=["legacy"])
    )
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    for s in config["splits"]["validation"]:
        (bootstrap / f"seed-{s}.npz").write_text("bootstrap")

    def evaluation(directory, successes):
        directory.mkdir(parents=True, exist_ok=True)
        manifest = build_cycle_manifest(
            config,
            audit,
            root=tmp_path,
            code_hash=identity_hash({}),
            policy=identity(tmp_path / "initial"),
            phase="monitor",
            baselines=False,
        )
        atomic_json(directory / "cohort.json", manifest)
        rows = []
        for ep in manifest["episodes"]:
            report = dict(
                task_reward=successes / 10,
                terminated=True,
                truncated=False,
                starvation_ticks=0,
                parcels=dict(
                    spawned=10, successes=successes, misses=10 - successes, timeouts=0, active=0
                ),
            )
            rows.append(dict(episode=ep, report=report))
            attempt = directory / "episodes" / ep["episode_id"] / "attempt-000"
            attempt.mkdir(parents=True, exist_ok=True)
            atomic_json(attempt / "summary.json", report)
            complete_attempt(attempt, binding=dict(cohort=manifest["binding"], episode=ep))
        atomic_json(directory / "rollout-summary.json", dict(episodes=rows))

    evaluation(tmp_path / "reference", 3)
    recipe = dict(
        initial_checkpoint="initial",
        behavior_checkpoint="behavior",
        offline_dir="offline",
        online_dirs=["online"],
        validation_bootstrap="bootstrap",
        compatibility="compatibility.json",
        initial_validation="reference/rollout-summary.json",
        global_batch=4,
        seed=1,
        drop_successes=3,
        cycles=[
            dict(
                generation=dict(
                    name=f"batch-{i}",
                    scenes=[132 + i],
                    behaviors=["fqi"],
                    regimes=["nominal", "family", "zero"],
                    epsilon=0.2,
                ),
                optimizer_updates=3 + i,
            )
            for i in range(3)
        ],
    )
    recipe_path = tmp_path / "recipe.json"
    atomic_json(recipe_path, recipe)
    state = SimpleNamespace(
        collections=[], fit_calls=[], repeats=[], fail=True, score=4, fitted=set()
    )

    def bootstrap_run(command, **kwargs):
        directory = kwargs["directory"] / "collection/bootstrap"
        directory.mkdir(parents=True)
        definition = json.loads((directory.parent.parent / "generation.json").read_text())
        for s in definition["scenes"]:
            (directory / f"seed-{s}.npz").write_text("generated")

    monkeypatch.setattr(campaign, "_bootstrap", bootstrap_run)

    def collect(config_path, directory, **kwargs):
        if kwargs["phase"] == "monitor":
            state.repeats.append(directory)
            evaluation(directory, state.score)
            return dict(completed=9)
        state.collections.append((directory, kwargs["checkpoint"], kwargs["gpus"]))
        manifest = build_cycle_manifest(
            config,
            audit,
            root=tmp_path,
            code_hash=identity_hash({}),
            policy=identity(kwargs["checkpoint"]),
            phase="online",
            baselines=False,
            generation=kwargs["generation"],
        )
        atomic_json(directory / "cohort.json", manifest)
        for ep in manifest["episodes"]:
            attempt = directory / "episodes" / ep["episode_id"] / "attempt-000"
            attempt.mkdir(parents=True)
            (attempt / "summary.json").write_text("{}")
            complete_attempt(attempt, binding=dict(cohort=manifest["binding"], episode=ep))
        return dict(completed=len(manifest["episodes"]))

    monkeypatch.setattr(parallel, "run_parallel", collect)

    def fit(config_path, offline, online, directory, **kwargs):
        state.fit_calls.append((directory, kwargs))
        checkpoint = directory / "iteration-0001"
        checkpoint.mkdir(parents=True, exist_ok=True)
        if directory not in state.fitted:
            state.fitted.add(directory)
            (checkpoint / "state.pt").write_text(str(kwargs["warm_start_checkpoint"]))
            atomic_json(
                checkpoint / "complete.json",
                dict(files={"state.pt": sha256(checkpoint / "state.pt")}),
            )
        if state.fail:
            state.fail = False
            raise RuntimeError("validation interrupted")
        evaluation(directory / "validation/iteration-0001", state.score)
        return dict(completed_iterations=1)

    monkeypatch.setattr(online_training, "run_online_training", fit)
    return SimpleNamespace(
        state=state,
        recipe=recipe,
        recipe_path=recipe_path,
        root=tmp_path,
        arguments=dict(
            config_path=config_path,
            output_dir=tmp_path / "run",
            root=tmp_path,
            recipe_path=recipe_path,
            audit_path=audit_path,
            gpus="7",
            training_gpus="7",
        ),
    )


def test_campaign_resumes_collection_and_fit_boundary_preserving_parent_and_budget(campaign):
    from latency_meta_mdp.meta.information.campaign import run_campaign

    with pytest.raises(RuntimeError, match="validation interrupted"):
        run_campaign(**campaign.arguments)
    first = campaign.root / "run/cycle-0001/fit/iteration-0001/state.pt"
    saved = first.read_bytes()
    result = run_campaign(**campaign.arguments)
    assert result["status"] == "complete"
    assert len(campaign.state.collections) == 3
    assert len(campaign.state.fitted) == 3
    assert first.read_bytes() == saved
    calls = campaign.state.fit_calls
    assert [args["optimizer_updates"] for _, args in calls] == [3, 3, 4, 5]
    assert [len(args["additional_online_dirs"]) for _, args in calls] == [1, 1, 2, 3]
    assert all(args["validate_initial"] is False for _, args in calls)
    assert all(args["global_batch"] == 4 and args["training_gpus"] == "7" for _, args in calls)
    assert campaign.state.collections[1][1] == first.parent
    assert result["best"] == str(first.parent)  # Equal scores retain the earliest winner.
    assert run_campaign(**campaign.arguments) == result
    assert len(campaign.state.fit_calls) == 4


def test_confirmed_regression_stops_admissions_and_keeps_initial_best(campaign):
    from latency_meta_mdp.meta.information.campaign import run_campaign

    campaign.state.fail, campaign.state.score = False, 1
    result = run_campaign(**campaign.arguments)
    assert result["status"] == "stopped_regression"
    assert len(campaign.state.collections) == 1
    assert len(campaign.state.repeats) == 1
    assert result["best"] == str(campaign.root / "initial")
    assert result["latest"].endswith("cycle-0001/fit/iteration-0001")


def test_overlapping_future_generation_is_rejected_before_any_collection(campaign):
    from latency_meta_mdp.meta.information.campaign import run_campaign

    campaign.recipe["cycles"][2]["generation"] = campaign.recipe["cycles"][0]["generation"]
    campaign.recipe_path.write_text(json.dumps(campaign.recipe))
    with pytest.raises(ValueError, match="overlap"):
        run_campaign(**campaign.arguments)
    assert not campaign.state.collections


def test_reusing_a_starter_scene_with_a_different_behavior_is_rejected(campaign):
    from latency_meta_mdp.meta.information.campaign import run_campaign

    campaign.recipe["cycles"][0]["generation"]["scenes"] = [100]
    campaign.recipe_path.write_text(json.dumps(campaign.recipe))
    with pytest.raises(ValueError, match="scene"):
        run_campaign(**campaign.arguments)
    assert not campaign.state.collections


def test_reference_summary_must_match_verified_episode_outcomes(campaign):
    from latency_meta_mdp.meta.information.campaign import run_campaign

    path = campaign.root / "reference/rollout-summary.json"
    data = json.loads(path.read_text())
    report = data["episodes"][0]["report"]
    report["task_reward"] = 0.9
    report["parcels"].update(successes=9, misses=1)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="binding"):
        run_campaign(**campaign.arguments)
    assert not campaign.state.collections


def test_bootstrap_termination_cleans_up_only_the_owned_process(tmp_path, monkeypatch):
    from latency_meta_mdp.meta.information import campaign
    from latency_meta_mdp.meta.information.resources import worker_lock

    calls = []

    class Process:
        def wait(self, timeout=None):
            calls.append(("wait", timeout))
            if timeout is None:
                raise InterruptedError("terminated")
            return 0

        def poll(self):
            return None

        def terminate(self):
            calls.append("terminate")

    monkeypatch.setattr(campaign.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(campaign.Path, "home", lambda: tmp_path)
    with pytest.raises(InterruptedError):
        campaign._bootstrap(
            ["python"], root=tmp_path, directory=tmp_path, device="3", cpu_threads=2
        )
    assert calls == [("wait", None), "terminate", ("wait", 30)]
    with worker_lock(tmp_path / ".cache/conveyor-meta/gpu-locks/3"):
        pass  # Cancellation released our GPU admission lock.
