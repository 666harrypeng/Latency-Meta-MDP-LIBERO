"""Resolve a matched local or published clean checkpoint for conditioned training."""

import json

from latency_meta_mdp.data.bundles import verify_bundle_files
from latency_meta_mdp.io.artifacts import sha256_file


def conditioned_initialization(job, profile, schedule, *, work, clean_work):
    clean = job["clean"]
    bundle = clean_work / "bundles" / clean["dataset_revision"]
    initialization = job.get("initialization")
    if initialization is not None:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=clean["dataset_repo"],
            repo_type="dataset",
            revision=clean["dataset_revision"],
            local_dir=bundle,
            token=False,
        )
        step = initialization["step"]
        if type(step) is not int or step <= 0:
            raise ValueError("initialization step must be a positive integer")
        root = work / "initialization" / initialization["revision"]
        snapshot_download(
            repo_id=initialization["repo_id"],
            revision=initialization["revision"],
            local_dir=root,
            token=False,
            allow_patterns=[
                f"{step}/params/**",
                f"{step}/assets/**",
                f"{step}/_CHECKPOINT_METADATA",
            ],
        )
        checkpoint = root / str(step)
        receipt = dict(initialization)
    else:
        step = schedule.num_train_steps
        checkpoint = (
            clean_work
            / "checkpoints"
            / profile.levels[clean["level"]].config_name
            / clean["run_name"]
            / str(step)
        )
        receipt = {
            "step": step,
            "clean_report_sha256": sha256_file(checkpoint.parent / f"verified_run-step{step}.json"),
        }
    if (
        verify_bundle_files(bundle)["level"] != clean["level"]
        or not (checkpoint / "params").is_dir()
    ):
        raise ValueError("conditioned initialization requires matching clean data and checkpoint")
    prep = json.loads((bundle / "preparation/preparation.json").read_text())
    norms = list((checkpoint / "assets").rglob("norm_stats.json"))
    if len(norms) != 1 or sha256_file(norms[0]) != prep["norm_stats_sha256"]:
        raise ValueError("clean checkpoint normalization differs from the training bundle")
    return bundle, checkpoint, receipt
