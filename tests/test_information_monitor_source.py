import json
from pathlib import Path

import pytest


def test_offline_monitor_requires_source_manifest_before_launch(tmp_path, monkeypatch):
    from latency_meta_mdp.meta.information.monitor import run_monitor_series

    checkpoint = tmp_path / "iteration-0000"
    checkpoint.mkdir()
    (checkpoint / "complete.json").write_text(
        json.dumps({"binding": {"cohort_sha256": "original"}})
    )
    with pytest.raises(ValueError, match="offline-dir"):
        run_monitor_series(
            "unused",
            tmp_path,
            tmp_path / "evaluation",
            root=tmp_path,
            audit_path="unused",
            bootstrap_dir="unused",
        )


@pytest.mark.parametrize("parallel", [False, True])
def test_monitor_binds_source_manifest_and_forwards_to_evaluator(tmp_path, monkeypatch, parallel):
    from latency_meta_mdp.meta.information import monitor
    from latency_meta_mdp.meta.information.config import sha256

    config, audit = tmp_path / "config.yaml", tmp_path / "audit.json"
    config.write_text("config")
    audit.write_text("{}")
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    for seed in [1, 2, 3]:
        (bootstrap / f"seed-{seed}.npz").write_bytes(b"bootstrap")
    source = tmp_path / "source/cohort.json"
    source.parent.mkdir()
    source.write_text("{}")
    checkpoint = tmp_path / "fit/iteration-0000"
    checkpoint.mkdir(parents=True)
    monkeypatch.setattr(monitor, "load_config", lambda _: {"splits": {"validation": [1, 2, 3]}})
    monkeypatch.setenv("PYTHONPATH", "/runtime/compat:/runtime/libs")
    validator = monitor.IterationValidator(
        config,
        tmp_path / "evaluation",
        root=tmp_path,
        audit_path=audit,
        bootstrap_dir=bootstrap,
        source_manifest=source,
        worker_python="/custom/bin/python" if parallel else None,
        gpus="4,5,6,7" if parallel else None,
        cpu_threads=7,
    )
    assert validator.protocol["source_manifest_sha256"] == sha256(source)

    def fake_launch(command, **kwargs):
        import sys

        assert kwargs["env"]["PYTHONPATH"] == f"/runtime/compat:/runtime/libs:{tmp_path}/src"
        assert kwargs["env"]["OMP_NUM_THREADS"] == "7"
        if parallel:
            assert command[0] == "/custom/bin/python"
            assert "parallel-collect" in command
            assert command[command.index("--gpus") + 1] == "4,5,6,7"
            assert command[command.index("--bootstrap-dir") + 1] == str(bootstrap)
        else:
            assert command[0] == sys.executable
            assert "cycle-collect" in command
        i = command.index("--offline-dir")
        assert Path(command[i + 1]) == source.parent
        raise RuntimeError("verified launch command")

    monkeypatch.setattr(monitor.subprocess, "Popen", fake_launch)
    with pytest.raises(RuntimeError, match="verified launch command"):
        validator(checkpoint, {"iteration": 0})
    source.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="binding"):
        monitor.IterationValidator(
            config,
            tmp_path / "evaluation",
            root=tmp_path,
            audit_path=audit,
            bootstrap_dir=bootstrap,
            source_manifest=source,
        )


def test_monitor_cli_forwards_offline_manifest(tmp_path, monkeypatch):
    import runpy
    import sys

    from latency_meta_mdp.meta.information import monitor

    seen = {}
    monkeypatch.setattr(monitor, "run_monitor_series", lambda *a, **kw: seen.update(kw))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_conveyor_meta.py",
            "monitor-series",
            "--config",
            "config.yaml",
            "--output-dir",
            "evaluation",
            "--checkpoint",
            "fit",
            "--scene-audit",
            "audit.json",
            "--bootstrap-dir",
            "bootstrap",
            "--offline-dir",
            str(tmp_path),
        ],
    )
    runpy.run_path("scripts/run_conveyor_meta.py", run_name="__main__")
    assert seen["source_manifest"] == tmp_path / "cohort.json"


@pytest.mark.skipif(__import__("torch").cuda.is_available() is False, reason="CUDA required")
@pytest.mark.parametrize("retain_tensor", [False, True])
def test_validation_releases_blas_cache_but_rejects_live_tensors(
    tmp_path, monkeypatch, retain_tensor
):
    import gc

    import torch

    from latency_meta_mdp.meta.information import monitor
    from latency_meta_mdp.meta.information.fqi import fit_iteration
    from latency_meta_mdp.meta.information.model import StageQ
    from latency_meta_mdp.meta.information.schema import ActorPacket

    config, audit = tmp_path / "config.yaml", tmp_path / "audit.json"
    config.write_text("config")
    audit.write_text("{}")
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    for seed in [1, 2, 3]:
        (bootstrap / f"seed-{seed}.npz").write_bytes(b"bootstrap")
    monkeypatch.setattr(monitor, "load_config", lambda _: {"splits": {"validation": [1, 2, 3]}})
    model = StageQ().cuda()
    fit_iteration(
        model,
        [[dict(packet=ActorPacket(10, "G"), action=5, task_reward=0.1, cost=0.0, terminated=True)]],
        mode="bound",
    )
    extra = torch.ones(1, device="cuda") if retain_tensor else None
    validator = monitor.IterationValidator(
        config,
        tmp_path / "evaluation",
        root=tmp_path,
        audit_path=audit,
        bootstrap_dir=bootstrap,
        model=model,
    )

    def launch(*args, **kwargs):
        assert torch.cuda.memory_allocated() == 0
        raise RuntimeError("reached validation launch")

    monkeypatch.setattr(monitor.subprocess, "Popen", launch)
    expected = "live GPU tensors" if retain_tensor else "reached validation launch"
    try:
        with pytest.raises(RuntimeError, match=expected):
            validator(tmp_path / "fit/iteration-0001", {"iteration": 1})
        assert next(model.parameters()).is_cuda
    finally:
        del extra
        model.cpu()
        gc.collect()
        torch._C._cuda_clearCublasWorkspaces()
        torch.cuda.empty_cache()


def test_cached_validation_recomputes_metrics_before_reusing_selection(tmp_path):
    from latency_meta_mdp.meta.information.config import sha256
    from latency_meta_mdp.meta.information.monitor import IterationValidator
    from latency_meta_mdp.meta.information.validation import validation_metrics

    checkpoint = tmp_path / "fit/iteration-0001"
    checkpoint.mkdir(parents=True)
    (checkpoint / "complete.json").write_text("{}")
    destination = tmp_path / "evaluation/iteration-0001"
    destination.mkdir(parents=True)
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    rows = [
        dict(
            episode=dict(scene=s, regime=r, behavior="fqi", epsilon=0, split="validation"),
            report=dict(
                task_reward=0.1,
                terminated=True,
                truncated=False,
                starvation_ticks=0,
                parcels=dict(spawned=10, successes=1, misses=9, timeouts=0, active=0),
            ),
        )
        for s in (1, 2, 3)
        for r in ("zero", "nominal", "family")
    ]
    summary = destination / "rollout-summary.json"
    summary.write_text(json.dumps(dict(episodes=rows)))
    metrics = validation_metrics(rows, scenes=[1, 2, 3])
    item = dict(
        metrics=metrics,
        checkpoint_receipt_sha256=sha256(checkpoint / "complete.json"),
        evaluation_sha256=sha256(summary),
    )
    validator = object.__new__(IterationValidator)
    validator.output_dir = destination.parent
    validator.bootstrap_dir = bootstrap
    validator.protocol = {"scenes": [1, 2, 3]}
    validator.progress = None
    validator.prune = False
    state = dict(protocol=validator.protocol, completed={"1": item}, best_iteration=1)
    state_path = checkpoint.parent / "validation-state.json"
    state_path.write_text(json.dumps(state))
    assert validator(checkpoint, {"iteration": 1}) == metrics
    item["metrics"] = {key: 0.9 for key in metrics}
    state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="binding"):
        validator(checkpoint, {"iteration": 1})
