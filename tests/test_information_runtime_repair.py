import json

import pytest

from latency_meta_mdp.meta.information.config import identity_hash, sha256


def test_sharing_records_occupants_and_admits_new_pids(monkeypatch, capsys):
    from latency_meta_mdp.meta.information import resources

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    monkeypatch.delenv("CONVEYOR_SHARED_GPU_PIDS", raising=False)
    monkeypatch.delenv("CONVEYOR_ROLLOUT_PARENT_PID", raising=False)
    monkeypatch.delenv("CONVEYOR_IDLE_LEARNER_PID", raising=False)
    active = ["gpu4, 1234, another-job"]
    monkeypatch.setattr(
        resources.subprocess,
        "check_output",
        lambda args, **kw: (
            "4, gpu4\n5, gpu5\n6, gpu6\n7, gpu7" if "index,uuid" in args[1] else active[0]
        ),
    )
    monkeypatch.delenv("CONVEYOR_GPU_SHARING", raising=False)
    assert "1234" in resources.gpu_processes()[0]
    monkeypatch.setenv("CONVEYOR_GPU_SHARING", "allow")
    for pid in ("1234", "5678"):
        active[0] = f"gpu4, {pid}, another-job"
        blockers, inventory = resources.gpu_processes()
        assert blockers == "" and pid in inventory
        assert pid in capsys.readouterr().out
    monkeypatch.setenv("CONVEYOR_GPU_SHARING", "typo")
    with pytest.raises(ValueError, match="sharing"):
        resources.gpu_processes()


def test_runtime_patch_pins_whole_execution_tree_without_changing_binding(tmp_path, monkeypatch):
    from latency_meta_mdp.meta.information.runtime_identity import source_identity

    folder = tmp_path / "src/latency_meta_mdp/meta/information"
    folder.mkdir(parents=True)
    resource = folder / "resources.py"
    resource.write_text("old resource admission")
    model = folder / "model.py"
    model.write_text("unchanged scientific model")

    def snapshot():
        return {str(p.relative_to(tmp_path)): sha256(p) for p in sorted(tmp_path.rglob("*.py"))}

    before = snapshot()
    resource.write_text("reviewed resource admission")
    after = snapshot()
    receipt = tmp_path / "runtime-patch.json"
    receipt.write_text(
        json.dumps(
            dict(
                root=str(tmp_path), before=before, after=after, reason="GPU sharing admission only"
            )
        )
    )
    monkeypatch.setenv("CONVEYOR_RUNTIME_PATCH", str(receipt))
    monkeypatch.setenv("CONVEYOR_RUNTIME_PATCH_SHA256", sha256(receipt))
    result = source_identity(tmp_path)
    assert result["code_hash"] == identity_hash(before)
    assert result["execution_code_hash"] == identity_hash(after)
    model.write_text("unexpected science change")
    with pytest.raises(ValueError, match="execution source"):
        source_identity(tmp_path)
    model.write_text("unchanged scientific model")
    receipt.write_text(receipt.read_text() + " ")
    with pytest.raises(ValueError, match="receipt"):
        source_identity(tmp_path)


def test_runtime_patch_cannot_approve_scientific_module(tmp_path, monkeypatch):
    from latency_meta_mdp.meta.information.runtime_identity import source_identity

    path = tmp_path / "src/latency_meta_mdp/meta/information/model.py"
    path.parent.mkdir(parents=True)
    path.write_text("changed model")
    after = {str(path.relative_to(tmp_path)): sha256(path)}
    before = dict.fromkeys(after, "0" * 64)
    receipt = tmp_path / "runtime-patch.json"
    receipt.write_text(json.dumps(dict(root=str(tmp_path), before=before, after=after)))
    monkeypatch.setenv("CONVEYOR_RUNTIME_PATCH", str(receipt))
    monkeypatch.setenv("CONVEYOR_RUNTIME_PATCH_SHA256", sha256(receipt))
    with pytest.raises(ValueError, match="operational"):
        source_identity(tmp_path)


def test_preflight_reports_sharing_and_execution_provenance(tmp_path, monkeypatch):
    from latency_meta_mdp.meta.information import preflight, resources

    config_path = tmp_path / "config.yaml"
    config_path.write_text("bound config")
    monkeypatch.setenv("CONVEYOR_GPU_SHARING", "allow")
    monkeypatch.delenv("CONVEYOR_RUNTIME_PATCH", raising=False)
    monkeypatch.setattr(preflight, "load_config", lambda _: {"assets": {}, "splits": {}})
    monkeypatch.setattr(preflight, "verify_runtime_dependencies", lambda *a: None)
    monkeypatch.setattr(preflight, "verify_assets", lambda *a: {})
    monkeypatch.setattr(preflight.subprocess, "check_output", lambda *a, **k: "inventory")
    monkeypatch.setattr(resources, "gpu_processes", lambda: ("", "other PID recorded"))
    result = preflight.preflight(config_path, tmp_path / "report", root=tmp_path)
    assert result["gpu_sharing"] == "allow"
    assert result["all_compute_processes"] == "other PID recorded"
    assert result["execution_code_hash"] == result["code_hash"]


@pytest.mark.parametrize("existing_protocol", [False, True])
def test_validation_resumes_audited_repair_without_rebinding_checkpoint(
    tmp_path, monkeypatch, existing_protocol
):
    from latency_meta_mdp.meta.information import monitor

    folder = tmp_path / "src/latency_meta_mdp/meta/information"
    folder.mkdir(parents=True)
    resource = folder / "resources.py"
    resource.write_text("original admission")
    before = {str(resource.relative_to(tmp_path)): sha256(resource)}
    resource.write_text("approved sharing admission")
    previous = {str(resource.relative_to(tmp_path)): sha256(resource)}
    config, audit = tmp_path / "config.yaml", tmp_path / "audit.json"
    config.write_text("config")
    audit.write_text("{}")
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    for scene in (1, 2, 3):
        (bootstrap / f"seed-{scene}.npz").write_bytes(b"bootstrap")
    monkeypatch.setattr(monitor, "load_config", lambda _: {"splits": {"validation": [1, 2, 3]}})
    monkeypatch.delenv("CONVEYOR_RUNTIME_PATCH", raising=False)
    kwargs = dict(root=tmp_path, audit_path=audit, bootstrap_dir=bootstrap, prune=False)
    evaluation = tmp_path / "fit/validation"
    if existing_protocol:
        old = monitor.IterationValidator(config, evaluation, **kwargs).protocol
        assert old["code_hash"] == identity_hash(previous)
        protocol_bytes = (evaluation / "protocol.json").read_bytes()
    # The validator plumbing is the only additional source change.
    (folder / "monitor.py").write_text("audited source identity plumbing")
    after = {str(p.relative_to(tmp_path)): sha256(p) for p in sorted(folder.glob("*.py"))}
    patch = tmp_path / "repair.json"
    patch.write_text(
        json.dumps(
            dict(root=str(tmp_path), before=before, after=after, previous_execution=previous)
        )
    )
    monkeypatch.setenv("CONVEYOR_RUNTIME_PATCH", str(patch))
    monkeypatch.setenv("CONVEYOR_RUNTIME_PATCH_SHA256", sha256(patch))
    validator = monitor.IterationValidator(config, evaluation, **kwargs)
    expected = identity_hash(previous if existing_protocol else before)
    assert validator.protocol["code_hash"] == expected
    if existing_protocol:
        assert (evaluation / "protocol.json").read_bytes() == protocol_bytes
    checkpoint = tmp_path / "fit/iteration-0002"
    checkpoint.mkdir()
    (checkpoint / "complete.json").write_text(json.dumps({"binding": validator.protocol}))
    receipt_hash = sha256(checkpoint / "complete.json")
    destination = evaluation / "iteration-0002"
    destination.mkdir()
    (destination / "cohort.json").write_text(
        json.dumps(
            dict(
                binding=dict(
                    code_hash=identity_hash(before), policy=dict(receipt_sha256=receipt_hash)
                )
            )
        )
    )
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
    (destination / "rollout-summary.json").write_text(
        json.dumps(dict(completed=9, expected=9, episodes=rows))
    )

    class CompletedCollection:
        stdout = []

        def wait(self):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(monitor.subprocess, "Popen", lambda *a, **k: CompletedCollection())
    result = validator(checkpoint, {"iteration": 2})
    assert result["nominal/reward"] == pytest.approx(0.1)
    assert sha256(checkpoint / "complete.json") == receipt_hash
    assert validator(checkpoint, {"iteration": 2}) == result
    # An unknown protocol or a changed scientific field must still fail.
    path = evaluation / "protocol.json"
    saved = json.loads(path.read_text())
    for key, value in (("code_hash", "unknown"), ("epsilon", 0.5)):
        path.write_text(json.dumps({**saved, key: value}))
        with pytest.raises(ValueError, match="binding"):
            monitor.IterationValidator(config, evaluation, **kwargs)
        path.write_text(json.dumps(saved))


def test_previous_execution_cannot_alias_scientific_changes(tmp_path, monkeypatch):
    from latency_meta_mdp.meta.information.runtime_identity import source_identity

    folder = tmp_path / "src/latency_meta_mdp/meta/information"
    folder.mkdir(parents=True)
    p = folder / "resources.py"
    p.write_text("audited repair")
    after = {str(p.relative_to(tmp_path)): sha256(p)}
    before = dict.fromkeys(after, "old resource hash")
    previous = {**before, "src/latency_meta_mdp/meta/information/model.py": "changed science"}
    patch = tmp_path / "repair.json"
    patch.write_text(
        json.dumps(
            dict(root=str(tmp_path), before=before, after=after, previous_execution=previous)
        )
    )
    monkeypatch.setenv("CONVEYOR_RUNTIME_PATCH", str(patch))
    monkeypatch.setenv("CONVEYOR_RUNTIME_PATCH_SHA256", sha256(patch))
    with pytest.raises(ValueError, match="operational"):
        source_identity(tmp_path)
