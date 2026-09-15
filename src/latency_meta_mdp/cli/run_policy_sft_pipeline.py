"""Execute clean SFT, node-local forecast generation and conditioned SFT in sequence."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from latency_meta_mdp.artifacts import sha256_file
from latency_meta_mdp.forecast_delivery import load_forecast_job


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--preparation-python", type=Path, default=Path("/opt/belief/bin/python"))
    p.add_argument("--check-access-only", action="store_true")
    a = p.parse_args()
    a.config = a.config.resolve()
    work = a.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    job = load_forecast_job(a.config, project_root=Path.cwd())
    identity = {
        "job_sha256": sha256_file(a.config),
        "clean_job_sha256": sha256_file(job["clean_job"]),
    }
    state_path = work / "pipeline.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["identity"] != identity:
            raise ValueError("Pipeline config changed; use a separate output directory")
    else:
        state = {"identity": identity, "completed": []}

    def save():
        temp = state_path.with_suffix(".tmp")
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(state_path)

    def run(stage, command):
        if stage in state["completed"]:
            return
        state.update(phase=stage, status="running")
        save()
        print(f"Starting {stage}", flush=True)
        result = subprocess.run(command)
        if result.returncode:
            state.update(status="failed", exit_code=result.returncode)
            save()
            raise SystemExit(result.returncode)
        state["completed"].append(stage)
        state.update(status="completed")
        save()

    prep = [
        str(a.preparation_python),
        "-u",
        "-m",
        "latency_meta_mdp.cli.prepare_forecast_sft",
        "--config",
        str(a.config),
        "--work-dir",
        str(work / "preparation"),
    ]
    # Recheck gated encoder access before allocating a long training job.
    subprocess.run(prep + ["--check-access"], check=True)
    if a.check_access_only:
        return
    from latency_meta_mdp.sft_launch import SFTLaunchRequest, resolve_sft_schedule
    from latency_meta_mdp.sft_profile import load_sft_profile

    c = job["clean"]
    profile = load_sft_profile(c["profile"])
    schedule = resolve_sft_schedule(
        profile=profile,
        request=SFTLaunchRequest(
            c["level"], c["run_name"], "formal", False, c["device_count"], c["batch_size"]
        ),
    )
    clean = work / "clean/checkpoints" / profile.levels[c["level"]].config_name / c["run_name"]
    command = [
        sys.executable,
        "-u",
        "-m",
        "latency_meta_mdp.cli.train_structured_pi05",
        "--config",
        str(job["clean_job"]),
        "--work-dir",
        str(work / "clean"),
    ]
    if (clean / f"verified_run-step{schedule.num_train_steps}.json").exists():
        command += ["--publish-only"]
    elif clean.exists():
        command += ["--resume"]
    run("clean_sft", command)
    run("forecast_generation", prep)
    command = [
        sys.executable,
        "-u",
        "-m",
        "latency_meta_mdp.cli.train_forecast_pi05",
        "--config",
        str(a.config),
        "--work-dir",
        str(work),
    ]
    if (work / "conditioned/completion.json").exists():
        command += ["--publish-only"]
    elif (work / "conditioned/checkpoints").exists():
        command += ["--resume"]
    run("conditioned_sft", command)
    state.update(phase="complete")
    save()
    print("Clean and conditioned SFT milestones published", flush=True)


if __name__ == "__main__":
    main()
