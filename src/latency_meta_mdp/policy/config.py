"""Experiment jobs and training-only overrides for policy learning."""

from __future__ import annotations

import math
import re
from pathlib import Path

import yaml

from latency_meta_mdp.io.paths import repository_root


def load_training_job(path: Path) -> dict:
    job = yaml.safe_load(path.read_text())
    version = job["schema_version"]
    if version == 1:
        if job["level"] not in (1, 2, 3):
            raise ValueError("Unsupported training job")
        if not re.fullmatch(r"[0-9a-f]{40}", job["dataset_revision"]):
            raise ValueError("Pin the dataset to a commit revision")
        config_root = path.resolve().parent
    elif version == 2:
        if not job.get("training"):
            raise ValueError("Task jobs require an explicit source-epoch training recipe")
        if (
            job.get("task_id") != "conveyor_sort"
            or job.get("variant") != "surface"
            or job.get("action_contract") != "panda_osc_pose_delta_conveyor_v2"
            or "level" in job
        ):
            raise ValueError("Unsupported task training identity")
        data = job["dataset"]
        if not re.fullmatch(r"[0-9a-f]{64}", data.get("manifest_sha256", "")):
            raise ValueError("Pin the training bundle manifest")
        if data.get("kind") == "local":
            data["path"] = (path.resolve().parent / data["path"]).resolve()
        elif data.get("kind") == "huggingface":
            if not data.get("repo_id") or not re.fullmatch(
                r"[0-9a-f]{40}", data.get("revision", "")
            ):
                raise ValueError("Pin the public dataset repo and revision")
        else:
            raise ValueError("Unsupported dataset source")
        config_root = repository_root()
    else:
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
    job["profile"] = (config_root / job["profile"]).resolve()
    if job.get("training") is not None:
        job["training"] = (config_root / job["training"]).resolve()
    return job


def resolve_policy_profile(job: dict, *, source_count: int | None = None):
    """Keep published data-contract bytes independent of optimization settings."""
    import dataclasses

    from latency_meta_mdp.policy.profile import load_sft_profile

    profile = load_sft_profile(job["profile"])
    if job.get("training") is None:
        return profile
    recipe = yaml.safe_load(job["training"].read_text())
    if job.get("schema_version", 1) == 2:
        if type(source_count) is not int or source_count < 1:
            raise ValueError("task training requires a verified source_count")
        recipe = dict(recipe)
        epochs = recipe.pop("source_epochs")
        warmup_fraction = recipe.pop("warmup_fraction")
        if not math.isfinite(epochs) or epochs <= 0 or not 0 < warmup_fraction < 1:
            raise ValueError("invalid source epoch or warmup budget")
        if {"batch_size", "num_train_steps", "warmup_steps", "keep_period"} & set(recipe):
            raise ValueError("task epoch recipe must not override derived batch/step counts")
        milestone = math.ceil(source_count * epochs / (3 * job["batch_size"]))
        if milestone < 2:
            raise ValueError("too few sources for rolling saves and three milestones")
        recipe.update(
            batch_size=job["batch_size"],
            num_train_steps=3 * milestone,
            keep_period=milestone,
            warmup_steps=min(3 * milestone - 1, max(1, math.ceil(3 * milestone * warmup_fraction))),
            save_interval=min(
                recipe.get("save_interval", profile.save_interval), max(1, milestone // 4)
            ),
        )
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
