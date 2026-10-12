import json
from pathlib import Path

import pytest

from latency_meta_mdp.meta.information.parallel import merge_shards, partition_manifest
from latency_meta_mdp.meta.information.resources import selected_gpu_processes


def test_partition_covers_each_episode_once_and_preserves_binding():
    manifest = dict(
        binding={"policy": "fixed"}, episodes=[{"episode_id": str(i)} for i in range(9)]
    )
    shards = [partition_manifest(manifest, i, 4) for i in range(4)]
    assert [len(s["episodes"]) for s in shards] == [3, 2, 2, 2]
    assert sorted(e["episode_id"] for s in shards for e in s["episodes"]) == list(
        map(str, range(9))
    )
    assert all(s["binding"] == manifest["binding"] for s in shards)
    assert "shard" not in manifest
    with pytest.raises(ValueError):
        partition_manifest(manifest, 4, 4)


def test_admission_ignores_other_devices_and_only_explicit_shared_pid():
    inventory = "0, GPU-a\n4, GPU-b\n5, GPU-c"
    active = "GPU-a, 12, python\nGPU-b, 12, python\nGPU-c, 24, python"
    assert selected_gpu_processes(inventory, active, visible="4", allowed="12") == ""
    assert selected_gpu_processes(inventory, active, visible="5", allowed="12") == "24, python"
    assert selected_gpu_processes(inventory, active, visible="GPU-b", allowed="") == "12, python"
    with pytest.raises(ValueError):
        selected_gpu_processes(inventory, active, visible="8", allowed="")
    with pytest.raises(ValueError):
        selected_gpu_processes(inventory, active, visible="", allowed="")


def test_merge_requires_all_shards_and_verified_episode_receipts(tmp_path):
    from latency_meta_mdp.meta.information.cohort import complete_attempt

    manifest = dict(
        binding={"policy": "fixed"}, episodes=[{"episode_id": str(i)} for i in range(4)]
    )
    (tmp_path / "cohort.json").write_text(json.dumps(manifest))
    for i in range(2):
        d = tmp_path / "workers" / str(i)
        d.mkdir(parents=True)
        shard = partition_manifest(manifest, i, 2)
        (d / "cohort.json").write_text(json.dumps(shard))
        for ep in shard["episodes"]:
            attempt = d / "episodes" / ep["episode_id"] / "attempt-000"
            attempt.mkdir(parents=True)
            (attempt / "summary.json").write_text("{}")
            complete_attempt(attempt, binding=dict(cohort=manifest["binding"], episode=ep))
    assert merge_shards(tmp_path, 2) == 4
    assert merge_shards(tmp_path, 2) == 4
    victim = tmp_path / "workers/1/episodes/1/attempt-000/summary.json"
    victim.write_text("tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        merge_shards(tmp_path, 2)


def test_writer_lock_rejects_concurrent_writer_and_releases(tmp_path):
    from latency_meta_mdp.meta.information.resources import worker_lock

    with worker_lock(tmp_path):
        with pytest.raises(RuntimeError, match="already owns"):
            with worker_lock(tmp_path):
                pass
    with worker_lock(tmp_path):
        pass


def test_admission_accepts_csv_whitespace():
    assert (
        selected_gpu_processes("0,GPU-a\n4,  GPU-b", "GPU-b, 12, python", visible="4")
        == "12, python"
    )


def test_worker_environment_maps_cuda_and_egl_without_parent_exemption(monkeypatch):
    import os

    from latency_meta_mdp.meta.information.parallel import worker_environment

    monkeypatch.setenv("CONVEYOR_IDLE_LEARNER_PID", str(os.getppid()))
    env = worker_environment("4")
    assert env["CUDA_VISIBLE_DEVICES"] == env["MUJOCO_EGL_DEVICE_ID"] == "4"
    assert env["OMP_NUM_THREADS"] == "4"
    assert "CONVEYOR_IDLE_LEARNER_PID" not in env


def test_worker_environment_honors_runtime_cpu_allocation():
    from latency_meta_mdp.meta.information.parallel import worker_environment

    env = worker_environment("2", cpu_threads=7)
    assert {env[k] for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")} == {"7"}
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    with pytest.raises(ValueError, match="threads"):
        worker_environment("2", cpu_threads=0)


def test_partial_merge_publishes_only_complete_verified_episodes(tmp_path):
    manifest = dict(
        binding={"policy": "fixed"}, episodes=[{"episode_id": "a"}, {"episode_id": "b"}]
    )
    (tmp_path / "cohort.json").write_text(json.dumps(manifest))
    assert merge_shards(tmp_path, 2, require_all=False) == 0
    assert not list((tmp_path / "episodes").iterdir())


def test_worker_tracker_does_not_treat_partial_matrix_as_complete_evaluation(tmp_path):
    from latency_meta_mdp.meta.cycle_tracking import CycleTracker

    tracker = object.__new__(CycleTracker)
    tracker.directory = tmp_path
    tracker.manifest = dict(episodes=[{}], shard=dict(index=0, count=4))
    tracker.phase = "monitor"
    tracker.run = None
    tracker.rows = [
        dict(
            episode=dict(behavior="fqi", regime="nominal"),
            report=dict(task_reward=0.0, parcels=dict(successes=0, spawned=10)),
        )
    ]
    tracker.close(success=True)
    assert not (tmp_path / "evaluation.json").exists()


@pytest.mark.parametrize("failed", [False, True])
def test_parallel_orchestrator_merges_or_stops_only_its_children(tmp_path, monkeypatch, failed):
    from latency_meta_mdp.meta import cycle_tracking
    from latency_meta_mdp.meta.information import cycle, parallel
    from latency_meta_mdp.meta.information.cohort import complete_attempt

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    manifest = dict(
        binding={"policy": "fixed"}, episodes=[{"episode_id": str(i)} for i in range(2)]
    )

    def prepare(config, directory, **kwargs):
        (directory / "cohort.json").write_text(json.dumps(manifest))
        return {}, manifest

    monkeypatch.setattr(cycle, "prepare_cycle", prepare)

    class Tracker:
        rows = []

        def __init__(self, *args, **kwargs):
            self.rows = []

        def __call__(self, ep, report, attempt, progress):
            self.rows.append(dict(episode=ep))

        def close(self, *, success):
            assert success == (not failed)

    monkeypatch.setattr(cycle_tracking, "CycleTracker", Tracker)
    children = []

    class Child:
        def __init__(self, code):
            self.code, self.terminated = code, False

        def poll(self):
            return self.code

        def terminate(self):
            self.terminated, self.code = True, -15

        def wait(self, **kwargs):
            return self.code

    def launch(command, *, env, **kwargs):
        index = int(command[command.index("--shard-index") + 1])
        assert env["CUDA_VISIBLE_DEVICES"] == str(4 + index)
        directory = Path(command[command.index("--output-dir") + 1])
        shard = partition_manifest(manifest, index, 2)
        (directory / "cohort.json").write_text(json.dumps(shard))
        if not failed:
            ep = shard["episodes"][0]
            attempt = directory / "episodes" / ep["episode_id"] / "attempt-000"
            attempt.mkdir(parents=True)
            (attempt / "summary.json").write_text("{}")
            complete_attempt(attempt, binding=dict(cohort=manifest["binding"], episode=ep))
            (directory / "progress.json").write_text('{"completed": 1}')
        child = Child((1 if index == 0 else None) if failed else 0)
        children.append(child)
        return child

    monkeypatch.setattr(parallel.subprocess, "Popen", launch)
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    kwargs = dict(
        root=tmp_path,
        audit_path=tmp_path / "audit.json",
        checkpoint=tmp_path / "checkpoint",
        phase="monitor",
        bootstrap_dir=bootstrap,
        gpus="4,5",
        enabled=False,
    )
    if failed:
        with pytest.raises(RuntimeError, match="worker failed"):
            parallel.run_parallel(tmp_path / "config", tmp_path / "result", **kwargs)
        assert children[1].terminated and not children[0].terminated
    else:
        assert (
            parallel.run_parallel(tmp_path / "config", tmp_path / "result", **kwargs)["completed"]
            == 2
        )
        assert len(list((tmp_path / "result/episodes").iterdir())) == 2
    import os

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_termination_interrupts_coordinator_and_restores_handler():
    import signal

    from latency_meta_mdp.meta.information.resources import interrupt_on_termination

    previous = signal.getsignal(signal.SIGTERM)
    with pytest.raises(InterruptedError, match="terminated"):
        with interrupt_on_termination():
            signal.raise_signal(signal.SIGTERM)
    assert signal.getsignal(signal.SIGTERM) == previous
