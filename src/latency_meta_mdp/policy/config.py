"""Experiment jobs and training-only overrides for policy learning."""

from __future__ import annotations

import re
from pathlib import Path

import yaml


def load_training_job(path: Path) -> dict:
    job = yaml.safe_load(path.read_text())
    if job["schema_version"] != 1 or job["level"] not in (1, 2, 3):
        raise ValueError("Unsupported training job")
    devices, batch = job["device_count"], job["batch_size"]
    if (
        type(devices) is not int
        or type(batch) is not int
        or devices < 1
        or batch < 1
        or batch % devices
    ):
        raise ValueError("Global batch must divide evenly over devices")
    if not re.fullmatch(r"[0-9a-f]{40}", job["dataset_revision"]):
        raise ValueError("Pin the dataset to a commit revision")
    job["profile"] = (path.resolve().parent / job["profile"]).resolve()
    if job.get("training") is not None:
        job["training"] = (path.resolve().parent / job["training"]).resolve()
    return job


def resolve_policy_profile(job: dict):
    """Keep published data-contract bytes independent of optimization settings."""
    import dataclasses

    from latency_meta_mdp.policy.profile import load_sft_profile

    profile = load_sft_profile(job["profile"])
    if job.get("training") is None:
        return profile
    recipe = yaml.safe_load(job["training"].read_text())
    allowed = {
        "batch_size",
        "num_workers",
        "num_train_steps",
        "warmup_steps",
        "peak_learning_rate",
        "decay_learning_rate",
        "save_interval",
        "keep_period",
        "log_interval",
        "fsdp_devices",
        "ema_decay",
    }
    if not isinstance(recipe, dict) or not set(recipe) <= allowed:
        raise ValueError("Training recipe must not change the data/model contract")
    return dataclasses.replace(profile, **recipe)
