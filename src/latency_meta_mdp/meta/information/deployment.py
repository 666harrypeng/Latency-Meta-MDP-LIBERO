"""Read-only deployment restore; deliberately independent of training resume."""

import json
from pathlib import Path

import torch

from latency_meta_mdp.meta.information.config import identity_hash, sha256, validate_binding
from latency_meta_mdp.meta.information.fqi import FqiScheduler, model_hash
from latency_meta_mdp.meta.information.model import StageQ


def deployment_binding(config):
    return dict(
        mode=config["mode"],
        algorithm="fqi",
        schema_version=1,
        assets=config["assets"]["identity"],
        config_hash=identity_hash(config),
    )


def scheduler_identity(scheduler):
    identity = dict(
        checkpoint=scheduler.checkpoint,
        receipt_sha256=scheduler.receipt_sha256,
        model_hash=scheduler.version,
    )
    if scheduler.source_manifest is not None:
        identity["source_manifest"] = scheduler.source_manifest
    return identity


def load_scheduler(
    checkpoint, expected, *, device="cpu", epsilon=0.0, seed=0, source_manifest=None
):
    checkpoint = Path(checkpoint)
    receipt = json.loads((checkpoint / "complete.json").read_text())
    for key in ("mode", "algorithm", "schema_version", "assets"):
        validate_binding(expected[key], receipt["binding"][key])
    if receipt["binding"].get("normalization") != "fixed-physical-scales-and-learned-LayerNorm":
        raise ValueError("deployment normalization differs")
    if "config_hash" in expected:
        if "config_hash" in receipt["binding"]:
            validate_binding(expected["config_hash"], receipt["binding"]["config_hash"])
        else:
            if source_manifest is None or sha256(source_manifest) != receipt["binding"].get(
                "cohort_sha256"
            ):
                raise ValueError("deployment source cohort hash differs")
            original = json.loads(Path(source_manifest).read_text())
            validate_binding(expected["config_hash"], original["binding"]["config_hash"])
    if receipt["files"] != {"state.pt": sha256(checkpoint / "state.pt")}:
        raise ValueError("deployment checkpoint hash differs")
    state = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=True)
    model = StageQ()
    model.load_state_dict(state["model"], strict=True)
    if model_hash(model) != receipt["model_hash"]:
        raise ValueError("deployment model hash differs")
    model.to(device).eval().requires_grad_(False)
    scheduler = FqiScheduler(model, epsilon=epsilon, seed=seed)
    scheduler.source_manifest = (
        str(Path(source_manifest).resolve()) if source_manifest is not None else None
    )
    scheduler.checkpoint = str(checkpoint.resolve())
    scheduler.receipt_sha256 = sha256(checkpoint / "complete.json")
    return scheduler
