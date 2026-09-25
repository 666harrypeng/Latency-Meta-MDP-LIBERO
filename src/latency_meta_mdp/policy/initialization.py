"""Resolve a matched local or published clean checkpoint for conditioned training."""

import json
from pathlib import Path

from latency_meta_mdp.data.bundles import verify_bundle_files
from latency_meta_mdp.io.artifacts import sha256_file


def conditioned_initialization(job, profile, schedule, *, work, clean_work, clean_inputs=None):
    clean = job["clean"]
    task_based = clean.get("schema_version") == 2
    if task_based:
        if clean_inputs is None:
            from latency_meta_mdp.policy.clean_data import resolve_clean_inputs

            clean_inputs = resolve_clean_inputs(clean, clean_work)
        bundle = clean_inputs.bundle_root
    else:
        bundle = clean_work / "bundles" / clean["dataset_revision"]
    initialization = job.get("initialization")
    if initialization is not None:
        from huggingface_hub import snapshot_download

        if not task_based:
            snapshot_download(
                repo_id=clean["dataset_repo"],
                repo_type="dataset",
                revision=clean["dataset_revision"],
                local_dir=bundle,
            )
        step = initialization["step"]
        if type(step) is not int or step <= 0:
            raise ValueError("initialization step must be a positive integer")
        if initialization.get("local_path"):
            checkpoint = Path(initialization["local_path"]).expanduser().resolve()
            if checkpoint.name != str(step):
                raise ValueError("local initialization directory must match the declared step")
        else:
            root = work / "initialization" / initialization["revision"]
            snapshot_download(
                repo_id=initialization["repo_id"],
                revision=initialization["revision"],
                local_dir=root,
                allow_patterns=[
                    f"{step}/params/**",
                    f"{step}/assets/**",
                    f"{step}/_CHECKPOINT_METADATA",
                ],
            )
            checkpoint = root / str(step)
        receipt = dict(initialization)
        if initialization.get("local_path"):
            receipt["local_path"] = str(checkpoint)
    else:
        step = schedule.num_train_steps
        checkpoint = (
            clean_work
            / "checkpoints"
            / (
                clean_inputs.task_parameters["config_name"]
                if task_based
                else profile.levels[clean["level"]].config_name
            )
            / clean["run_name"]
            / str(step)
        )
        receipt = {
            "step": step,
            "clean_report_sha256": sha256_file(checkpoint.parent / f"verified_run-step{step}.json"),
        }
    if (not task_based and verify_bundle_files(bundle)["level"] != clean["level"]) or not (
        checkpoint / "params"
    ).is_dir():
        raise ValueError("conditioned initialization requires matching clean data and checkpoint")
    prep = (
        clean_inputs.preparation
        if task_based
        else json.loads((bundle / "preparation/preparation.json").read_text())
    )
    norms = list((checkpoint / "assets").rglob("norm_stats.json"))
    if len(norms) != 1 or sha256_file(norms[0]) != prep["norm_stats_sha256"]:
        raise ValueError("clean checkpoint normalization differs from the training bundle")
    return bundle, checkpoint, receipt
