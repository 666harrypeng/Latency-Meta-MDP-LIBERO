"""Execute clean SFT, node-local forecast generation and conditioned SFT in sequence."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from latency_meta_mdp.data.forecast.assets import load_forecast_job
from latency_meta_mdp.io.artifacts import sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", "--work-dir", dest="work_dir", type=Path, required=True)
    p.add_argument("--preparation-python", type=Path, default=Path("/opt/belief/bin/python"))
    p.add_argument("--check-access-only", action="store_true")
    a = p.parse_args()
    a.config = a.config.resolve()
    work = a.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    job = load_forecast_job(a.config, project_root=Path.cwd())
    if job.get("input_mode", "current_and_forecast") != "current_and_forecast":
        raise ValueError("Use train_conditioned_policy.py for a forecast-only job; reuse clean SFT")
    identity = {
        "job_sha256": sha256_file(a.config),
        "clean_job_sha256": sha256_file(job["clean_job"]),
        "training_recipe_sha256": sha256_file(job["clean"]["training"])
        if job["clean"].get("training")
        else None,
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
        "latency_meta_mdp.data.forecast.prepare",
        "--config",
        str(a.config),
        "--work-dir",
        str(work / "preparation"),
    ]
    # Recheck gated encoder access before allocating a long training job.
    subprocess.run(prep + ["--check-access"], check=True)
    if a.check_access_only:
        return
    if job.get("initialization") is None:
        from latency_meta_mdp.policy.config import resolve_policy_profile
        from latency_meta_mdp.policy.schedule import SFTLaunchRequest, resolve_sft_schedule

        c = job["clean"]
        task_inputs = None
        if c["schema_version"] == 2:
            from latency_meta_mdp.policy.clean_data import resolve_clean_inputs

            task_inputs = resolve_clean_inputs(c, work / "clean")
        profile = task_inputs.profile if task_inputs else resolve_policy_profile(c)
        schedule = resolve_sft_schedule(
            profile=profile,
            request=SFTLaunchRequest(
                c.get("level"),
                c["run_name"],
                "formal",
                False,
                c["device_count"],
                c["batch_size"],
                task_id=c.get("task_id"),
            ),
        )
        config_name = (
            task_inputs.task_parameters["config_name"]
            if task_inputs
            else profile.levels[c["level"]].config_name
        )
        clean = work / "clean/checkpoints" / config_name / c["run_name"]
        command = [
            sys.executable,
            "-u",
            "-m",
            "latency_meta_mdp.policy.train_clean",
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
        "latency_meta_mdp.policy.train_conditioned",
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
    print("Policy pipeline completed", flush=True)


if __name__ == "__main__":
    main()
