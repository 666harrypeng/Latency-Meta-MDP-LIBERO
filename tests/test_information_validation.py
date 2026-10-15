import copy
import json
import random

import numpy as np
import pytest
import torch

from latency_meta_mdp.meta.information.model import StageQ
from latency_meta_mdp.meta.information.schema import ActorPacket
from latency_meta_mdp.meta.information.training import train_iterations


def fit(model, path, **kwargs):
    data = [
        [
            dict(
                packet=ActorPacket(10, "G"),
                action=5,
                task_reward=0.1,
                cost=0.0,
                duration=1,
                terminated=True,
                next_packet=None,
            )
        ]
    ]
    return train_iterations(
        model,
        data,
        path,
        binding={"mode": "bound"},
        total_iterations=3,
        epochs=1,
        learning_rate=0.001,
        conservative_weight=0.01,
        **kwargs,
    )


def test_validation_runs_at_initial_and_each_boundary_without_changing_learning(tmp_path):
    torch.manual_seed(20)
    initial = StageQ()
    baseline, monitored = copy.deepcopy(initial), copy.deepcopy(initial)
    np.random.seed(9)
    fit(baseline, tmp_path / "baseline")
    seen = []

    def validate(path, receipt):
        assert (path / "state.pt").exists()
        seen.append(receipt["iteration"])
        torch.rand(3)
        np.random.rand(3)
        random.random()

    np.random.seed(9)
    fit(monitored, tmp_path / "monitored", validate_checkpoint=validate)
    assert seen == [0, 1, 2, 3]
    for k, value in baseline.state_dict().items():
        torch.testing.assert_close(value, monitored.state_dict()[k], rtol=0, atol=0)


def test_failed_validation_resumes_before_next_fqi_and_pruning_keeps_recovery(tmp_path):
    from latency_meta_mdp.meta.information.validation import retain_checkpoints

    def fail(path, receipt):
        if receipt["iteration"] == 1:
            raise RuntimeError("evaluation interrupted")

    with pytest.raises(RuntimeError, match="evaluation interrupted"):
        fit(StageQ(), tmp_path, validate_checkpoint=fail)
    assert (tmp_path / "iteration-0001/state.pt").exists()
    assert not (tmp_path / "iteration-0002").exists()
    assert (tmp_path / "latest").resolve().name == "iteration-0001"
    seen = []

    def validate(path, receipt):
        seen.append(receipt["iteration"])
        retain_checkpoints(tmp_path, best_iteration=1)

    fit(StageQ(), tmp_path, validate_checkpoint=validate)
    assert seen == [1, 2, 3]
    assert sorted(p.parent.name for p in tmp_path.glob("iteration-*/state.pt")) == [
        "iteration-0001",
        "iteration-0003",
    ]
    assert (tmp_path / "latest").resolve().name == "iteration-0003"
    assert (tmp_path / "best").resolve().name == "iteration-0001"
    assert fit(StageQ(), tmp_path)["completed_iterations"] == 3
    # Missing weights without a hash-bound pruning receipt remain corruption.
    (tmp_path / "retention.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        fit(StageQ(), tmp_path)


def test_monitor_manifest_is_paired_greedy_three_by_three(tmp_path):
    from latency_meta_mdp.meta.information.config import sha256
    from latency_meta_mdp.meta.information.cycle import build_cycle_manifest

    source = tmp_path / "source.json"
    source.write_text("{}")
    config = dict(
        mode="bound",
        assets={"identity": {}},
        excluded_scenes=[],
        splits={"train": list(range(100, 164)), "validation": list(range(200, 220)), "test": [300]},
    )
    audit = dict(
        local_inventory_pass=True,
        candidate_splits=config["splits"],
        excluded_scenes=[],
        sources=[dict(path=source.name, sha256=sha256(source))],
    )
    manifest = build_cycle_manifest(
        config, audit, root=tmp_path, code_hash="code", policy={}, phase="monitor", baselines=False
    )
    eps = manifest["episodes"]
    assert len(eps) == 9
    assert {(e["scene"], e["regime"]) for e in eps} == {
        (s, r) for s in (200, 201, 202) for r in ("zero", "nominal", "family")
    }
    assert all(e["epsilon"] == 0 and e["split"] == "validation" for e in eps)
    assert manifest["binding"]["scope"] == "evaluation"


def test_three_rewards_stay_separate_and_selection_uses_delayed_modes():
    from latency_meta_mdp.meta.information.validation import validation_metrics
    from latency_meta_mdp.meta.online_tracking import fit_metrics

    rows = []
    for scene in (1, 2, 3):
        for regime, delivered in [("zero", 9), ("nominal", 2), ("family", 4)]:
            rows.append(
                dict(
                    episode=dict(
                        scene=scene, regime=regime, behavior="fqi", epsilon=0, split="validation"
                    ),
                    report=dict(
                        task_reward=delivered / 10,
                        terminated=True,
                        truncated=False,
                        starvation_ticks=0,
                        parcels=dict(
                            spawned=10,
                            successes=delivered,
                            misses=10 - delivered,
                            timeouts=0,
                            active=0,
                        ),
                    ),
                )
            )
    metrics = validation_metrics(rows, scenes=[1, 2, 3])
    assert metrics["zero/reward"] == pytest.approx(0.9)
    assert metrics["nominal/reward"] == pytest.approx(0.2)
    assert metrics["family/reward"] == pytest.approx(0.4)
    assert metrics["selection_score"] == pytest.approx(0.3)
    values = fit_metrics(
        dict(phase="validation_complete", iteration=2, actual_update=640, metrics=metrics),
        episodes=160,
        epochs=2,
    )
    assert values["validation/zero/reward"] == pytest.approx(0.9)
    assert values["validation/nominal/reward"] == pytest.approx(0.2)
    assert values["validation/family/reward"] == pytest.approx(0.4)
    assert values["validation/iteration"] == 2
    for bad in (rows[:-1], rows + [rows[0]]):
        with pytest.raises(ValueError):
            validation_metrics(bad, scenes=[1, 2, 3])
    rows[0]["episode"]["split"] = "train"
    with pytest.raises(ValueError):
        validation_metrics(rows, scenes=[1, 2, 3])


def test_gpu_handoff_only_exempts_direct_waiting_parent(monkeypatch):
    from latency_meta_mdp.meta.information.preflight import blocking_gpu_processes

    monkeypatch.setattr("os.getppid", lambda: 123)
    monkeypatch.setenv("CONVEYOR_IDLE_LEARNER_PID", "123")
    assert blocking_gpu_processes("123, python\n456, other") == "456, other"
    assert blocking_gpu_processes("123, python") == ""
    monkeypatch.setenv("CONVEYOR_IDLE_LEARNER_PID", "456")
    with pytest.raises(ValueError, match="parent"):
        blocking_gpu_processes("123, python\n456, other")


def test_best_selection_keeps_incumbent_on_ties_and_rejects_cohort_change(tmp_path):
    from latency_meta_mdp.meta.information.validation import record_validation

    model = StageQ()
    fit(model, tmp_path)
    metrics = {
        "zero/reward": 0.8,
        "nominal/reward": 0.2,
        "family/reward": 0.4,
        "selection_score": 0.3,
    }
    protocol = {"scenes": [1, 2, 3], "regimes": ["zero", "nominal", "family"]}
    for iteration in (0, 1):
        record_validation(
            tmp_path,
            iteration=iteration,
            metrics=dict(metrics, selection_score=(0.2 + 0.4) / 2) if iteration == 1 else metrics,
            protocol=protocol,
            evaluation_sha256="evidence",
        )
    state = json.loads((tmp_path / "validation-state.json").read_text())
    assert state["best_iteration"] == 0
    assert state["completed"]["1"]["metrics"]["zero/reward"] == 0.8
    with pytest.raises(ValueError):
        record_validation(
            tmp_path,
            iteration=2,
            metrics=metrics,
            protocol={"scenes": [9, 8, 7]},
            evaluation_sha256="evidence",
        )


@pytest.mark.parametrize("behavior", ["fqi", "actor"])
def test_monitor_tracker_exports_three_separate_rewards(tmp_path, behavior):
    from latency_meta_mdp.meta.cycle_tracking import CycleTracker

    definitions = [
        dict(
            episode_id=f"{s}-{r}",
            scene=s,
            regime=r,
            behavior=behavior,
            deterministic=True,
            epsilon=0.0,
            split="validation",
        )
        for s in (1, 2, 3)
        for r in ("zero", "nominal", "family")
    ]
    (tmp_path / "cohort.json").write_text(
        json.dumps(dict(binding=dict(phase="monitor"), episodes=definitions))
    )
    tracker = CycleTracker(tmp_path, name="test", enabled=False)
    for ep in definitions:
        n = {"zero": 9, "nominal": 2, "family": 4}[ep["regime"]]
        tracker(
            ep,
            dict(
                task_reward=n / 10,
                terminated=True,
                truncated=False,
                starvation_ticks=0,
                executed_steps=2000,
                parcels=dict(spawned=10, successes=n, misses=10 - n, timeouts=0, active=0),
            ),
            tmp_path,
            {},
        )
    tracker.close(success=True)
    result = json.loads((tmp_path / "evaluation.json").read_text())["metrics"]
    assert result[f"{behavior}/zero/reward"] == pytest.approx(0.9)
    assert result[f"{behavior}/nominal/reward"] == pytest.approx(0.2)
    assert result[f"{behavior}/family/reward"] == pytest.approx(0.4)
    assert result[f"{behavior}/selection_score"] == pytest.approx(0.3)
