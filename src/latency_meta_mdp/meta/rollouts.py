"""Local GPU workers for the existing RTC collection/evaluation implementation."""

import json
import os
import signal
import subprocess
import time
from pathlib import Path

from latency_meta_mdp.meta.cost import episode_cost_components


def worker_environment(gpu):
    return {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "MUJOCO_GL": "egl",
        "MUJOCO_EGL_DEVICE_ID": str(gpu),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(Path.cwd() / "src"),
        "OMP_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "1",
    }


def stop_workers(processes):
    for process in processes:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def rollout_command(
    cfg,
    *,
    cohort,
    regime,
    output,
    worker,
    workers,
    scheduler,
    checkpoint,
    collect,
    epsilon,
    replica,
):
    command = [
        cfg["evaluation_python"],
        "-u",
        "-m",
        "latency_meta_mdp.runtime.evaluate",
        "--protocol",
        "rtc",
        "--prepare-forecast-before-decision",
        "--discount-per-tick",
        "1",
        "--task-horizon-terminal",
        "--cohort",
        str(cohort),
        "--regime",
        regime,
        "--output-root",
        str(output),
        "--worker-count",
        str(workers),
        "--worker-index",
        str(worker),
        "--scheduler",
        scheduler,
    ]
    for key, value in cfg["evaluation"].items():
        command += ["--" + key.replace("_", "-"), str(value)]
    if checkpoint is not None:
        command += ["--meta-q-checkpoint", str(checkpoint)]
    if collect:
        command += [
            "--collect-meta-transitions",
            "--meta-exploration-epsilon",
            str(epsilon),
            "--exploration-replica",
            str(replica),
        ]
    if cfg["record_video"]:
        command.append("--record-video")
    return command


def collect_results(cohort, output, regime, *, collect, video):
    cases = json.loads(Path(cohort).read_text())["cases"]
    expected = {
        f"master-{c['master_index']:03d}-seed-{c['policy_seed']}-{regime}.json": c for c in cases
    }
    paths = {p.name: p for p in Path(output).glob("master*.json")}
    if set(paths) != set(expected):
        raise ValueError("rollout result inventory differs from cohort")
    entries = []
    for name in sorted(paths):
        path = paths[name]
        row = json.loads(path.read_text())
        if (
            row["case"] != expected[name]
            or row["identity"]["regime"] != regime
            or not row["terminated"]
            or row["truncated"]
        ):
            raise ValueError("incomplete or mismatched rollout result")
        entry = {"result": str(path.resolve())}
        if collect:
            replay = path.parent / row["meta_replay"]["path"]
            if not replay.is_file():
                raise FileNotFoundError(replay)
            entry["replay"] = str(replay.resolve())
        if video:
            recording = row["video"]
            file = path.parent / recording["path"]
            if (
                not file.is_file()
                or file.stat().st_size == 0
                or recording["frames"] != row["recorded_frames"]
            ):
                raise ValueError("missing or incomplete rollout video")
        entries.append(entry)
    return entries


def run_rollouts(cfg, partition, output, *, checkpoint=None, collect=True, epsilon=0.2, replica=0):
    """One worker per configured GPU; each worker keeps models loaded across episodes."""
    entries = []
    for regime, cohort in cfg["cohorts"][partition].items():
        dest = Path(output) / regime
        dest.mkdir(parents=True, exist_ok=True)
        cases = json.loads(Path(cohort).read_text())["cases"]
        workers = min(len(cases), len(cfg["gpu_ids"]))
        processes, logs = [], []
        scheduler = "probabilistic" if collect else "learned"
        try:
            for worker, gpu in enumerate(cfg["gpu_ids"][:workers]):
                log = (dest / f"worker-{worker}.log").open("a")
                logs.append(log)
                command = rollout_command(
                    cfg,
                    cohort=cohort,
                    regime=regime,
                    output=dest,
                    worker=worker,
                    workers=workers,
                    scheduler=scheduler,
                    checkpoint=checkpoint,
                    collect=collect,
                    epsilon=epsilon,
                    replica=replica,
                )
                processes.append(
                    subprocess.Popen(
                        command,
                        env=worker_environment(gpu),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                )
            last_report = 0
            while True:
                codes = [p.poll() for p in processes]
                if any(code not in (0, None) for code in codes):
                    raise RuntimeError(f"{partition}/{regime} worker failed; logs: {dest}")
                now = time.monotonic()
                if now - last_report >= 30 or all(code == 0 for code in codes):
                    print(
                        json.dumps(
                            {
                                "stage": partition,
                                "regime": regime,
                                "episodes": len(list(dest.glob("master*.json"))),
                                "expected": len(cases),
                            }
                        ),
                        flush=True,
                    )
                    last_report = now
                if all(code == 0 for code in codes):
                    break
                time.sleep(1)
        finally:
            stop_workers(processes)
            for log in logs:
                log.close()
        entries.extend(
            collect_results(cohort, dest, regime, collect=collect, video=cfg["record_video"])
        )
    return entries


def summarize_results(entries, profile):
    groups = {}
    for entry in entries:
        result = json.loads(Path(entry["result"]).read_text())
        regime = result["identity"]["regime"]
        group = groups.setdefault(
            regime,
            {
                "episodes": 0,
                "successes": 0,
                "total_cost": 0,
                "policy_calls": 0,
                "forecast_calls": 0,
            },
        )
        group["episodes"] += 1
        group["successes"] += int(result["success"])
        group["total_cost"] += episode_cost_components(result, profile)["episode_cost"]
        group["policy_calls"] += result["policy_calls"]
        group["forecast_calls"] += result["forecast_calls"]
    for group in groups.values():
        group["mean_cost"] = group["total_cost"] / group["episodes"]
    return groups
