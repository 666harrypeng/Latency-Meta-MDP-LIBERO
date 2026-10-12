"""Independent resident rollout workers; deterministic partition and verified merge."""

import json
import os
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

from latency_meta_mdp.meta.information.config import validate_binding


def partition_manifest(manifest, index, count):
    if type(count) is not int or type(index) is not int or not 0 <= index < count:
        raise ValueError("invalid shard index/count")
    if count > len(manifest["episodes"]):
        raise ValueError("more workers than episodes")
    return dict(
        manifest, episodes=manifest["episodes"][index::count], shard=dict(index=index, count=count)
    )


def merge_shards(directory, count, *, require_all=True):
    from latency_meta_mdp.meta.information.cohort import verify_complete

    directory = Path(directory).resolve()
    manifest = json.loads((directory / "cohort.json").read_text())
    admitted = []
    for i in range(count):
        worker = directory / "workers" / str(i)
        if not (worker / "cohort.json").exists() and not require_all:
            continue
        shard = json.loads((worker / "cohort.json").read_text())
        validate_binding(partition_manifest(manifest, i, count), shard)
        for ep in shard["episodes"]:
            attempts = list((worker / "episodes" / ep["episode_id"]).glob("attempt-*"))
            valid = [
                a
                for a in attempts
                if verify_complete(a, binding=dict(cohort=manifest["binding"], episode=ep))
            ]
            if not valid and not require_all:
                continue
            if len(valid) != 1:
                raise ValueError("merge requires exactly one completed attempt per episode")
            admitted.append((ep["episode_id"], valid[0].parent))
    # Validate everything before publishing any canonical links.
    destination = directory / "episodes"
    destination.mkdir(exist_ok=True)
    for episode_id, source in admitted:
        link = destination / episode_id
        if link.exists():
            if not link.is_symlink() or link.resolve() != source:
                raise ValueError("merge destination differs")
        else:
            link.symlink_to(os.path.relpath(source, destination), target_is_directory=True)
    return len(admitted)


def worker_environment(device, *, cpu_threads=4):
    if type(cpu_threads) is not int or cpu_threads < 1:
        raise ValueError("worker CPU threads must be positive")
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=device,
        MUJOCO_EGL_DEVICE_ID=device,
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        OMP_NUM_THREADS=str(cpu_threads),
        MKL_NUM_THREADS=str(cpu_threads),
        OPENBLAS_NUM_THREADS=str(cpu_threads),
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
    )
    # The coordinator checked its parked direct parent. A worker has a different parent.
    parent = env.pop("CONVEYOR_IDLE_LEARNER_PID", None)
    if parent is not None:
        if parent != str(os.getppid()):
            raise ValueError("idle learner must be the coordinator's direct parent")
        env["CONVEYOR_SHARED_GPU_PIDS"] = ",".join(
            p for p in (env.get("CONVEYOR_SHARED_GPU_PIDS", ""), parent) if p
        )
    return env


def run_parallel(
    config_path,
    output_dir,
    *,
    root,
    audit_path,
    checkpoint,
    phase,
    bootstrap_dir,
    gpus,
    worker_python=None,
    cpu_threads=4,
    baselines=False,
    source_manifest=None,
    enabled=True,
    generation=None,
):
    from latency_meta_mdp.meta.cycle_tracking import CycleTracker
    from latency_meta_mdp.meta.information.cycle import prepare_cycle
    from latency_meta_mdp.meta.information.resources import interrupt_on_termination, worker_lock
    from latency_meta_mdp.meta.information.validation import atomic_json

    root, output_dir = Path(root).resolve(), Path(output_dir).resolve()
    devices = gpus.split(",")
    if not devices or len(set(devices)) != len(devices) or any(not d.isdigit() for d in devices):
        raise ValueError("gpus must be unique physical GPU indices")
    if not Path(bootstrap_dir).is_dir():
        raise ValueError("verified bootstrap directory is required")
    with ExitStack() as stack:
        stack.enter_context(interrupt_on_termination())
        stack.enter_context(worker_lock(output_dir))
        for device in sorted(devices, key=int):
            stack.enter_context(
                worker_lock(Path.home() / ".cache/conveyor-meta/gpu-locks" / device)
            )
        previous = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        try:
            _, manifest = prepare_cycle(
                config_path,
                output_dir,
                root=root,
                audit_path=audit_path,
                checkpoint=checkpoint,
                phase=phase,
                baselines=baselines,
                source_manifest=source_manifest,
                generation=generation,
            )
        finally:
            if previous is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous
        for i in range(len(devices)):
            partition_manifest(manifest, i, len(devices))
        if generation is not None:
            atomic_json(output_dir / "generation.json", generation)
        execution = dict(
            gpus=devices,
            worker_python=str(worker_python or sys.executable),
            cpu_threads_per_worker=cpu_threads,
            host=os.uname().nodename,
            partition="manifest_index_mod_workers",
        )
        path = output_dir / "execution.json"
        if path.exists():
            validate_binding(execution, json.loads(path.read_text()))
        else:
            atomic_json(path, execution)
        atomic_json(
            output_dir / "admission.json",
            dict(shared_pids=os.environ.get("CONVEYOR_SHARED_GPU_PIDS", "")),
        )
        tracker = CycleTracker(output_dir, name=output_dir.name, enabled=enabled)
        reported = {row["episode"]["episode_id"] for row in tracker.rows}
        success = False
        processes, logs = [], []
        started = time.perf_counter()
        try:
            for i, device in enumerate(devices):
                worker = output_dir / "workers" / str(i)
                worker.mkdir(parents=True, exist_ok=True)
                link = worker / "bootstrap"
                if not link.exists():
                    link.symlink_to(Path(bootstrap_dir).resolve(), target_is_directory=True)
                elif link.resolve() != Path(bootstrap_dir).resolve():
                    raise ValueError("worker bootstrap differs")
                command = [
                    str(worker_python or sys.executable),
                    "-u",
                    str(root / "scripts/run_conveyor_meta.py"),
                    "cycle-collect",
                    "--config",
                    str(Path(config_path).resolve()),
                    "--scene-audit",
                    str(Path(audit_path).resolve()),
                    "--checkpoint",
                    str(Path(checkpoint).resolve()),
                    "--phase",
                    phase,
                    "--output-dir",
                    str(worker),
                    "--shard-index",
                    str(i),
                    "--num-shards",
                    str(len(devices)),
                    "--no-wandb",
                ]
                if not baselines:
                    command.append("--meta-only")
                if generation is not None:
                    command.extend(["--generation", str(output_dir / "generation.json")])
                if source_manifest is not None:
                    command.extend(["--offline-dir", str(Path(source_manifest).resolve().parent)])
                env = worker_environment(device, cpu_threads=cpu_threads)
                log = (worker / "execution.log").open("a")
                logs.append(log)
                processes.append(
                    subprocess.Popen(
                        command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT
                    )
                )
            last = None
            while True:
                codes = [p.poll() for p in processes]
                counts = []
                for i in range(len(devices)):
                    progress = output_dir / "workers" / str(i) / "progress.json"
                    counts.append(
                        json.loads(progress.read_text())["completed"] if progress.exists() else 0
                    )
                status = dict(
                    completed=sum(counts),
                    expected=len(manifest["episodes"]),
                    per_worker=counts,
                    exit_codes=codes,
                    wall_seconds=time.perf_counter() - started,
                )
                atomic_json(output_dir / "parallel-progress.json", status)
                if (counts, codes) != last:
                    print("PARALLEL_PROGRESS " + json.dumps(status), flush=True)
                    last = (counts, codes)
                    merge_shards(output_dir, len(devices), require_all=False)
                    for ep in manifest["episodes"]:
                        if ep["episode_id"] in reported:
                            continue
                        for attempt in sorted(
                            (output_dir / "episodes" / ep["episode_id"]).glob("attempt-*")
                        ):
                            from latency_meta_mdp.meta.information.cohort import verify_complete

                            if verify_complete(
                                attempt, binding=dict(cohort=manifest["binding"], episode=ep)
                            ):
                                tracker(
                                    ep,
                                    json.loads((attempt / "summary.json").read_text()),
                                    attempt,
                                    status,
                                )
                                reported.add(ep["episode_id"])
                                break
                if any(c not in (None, 0) for c in codes):
                    raise RuntimeError("rollout worker failed; inspect workers/*/execution.log")
                if all(c == 0 for c in codes):
                    break
                time.sleep(5)
            completed = merge_shards(output_dir, len(devices))
            success = True
            return dict(
                completed=completed, remaining=0, wall_seconds=time.perf_counter() - started
            )
        finally:
            for p in processes:
                if p.poll() is None:
                    p.terminate()
            for p in processes:
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
            for log in logs:
                log.close()
            tracker.close(success=success)
