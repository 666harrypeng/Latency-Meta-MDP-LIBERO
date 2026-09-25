"""Public dependencies and local materialization for clean-to-forecast SFT."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import yaml

from latency_meta_mdp.io.artifacts import sha256_file
from latency_meta_mdp.policy.config import load_training_job


def load_forecast_job(path: Path, *, project_root: Path) -> dict:
    job = yaml.safe_load(path.read_text())
    if job["schema_version"] not in (1, 2):
        raise ValueError("Unsupported forecast job")
    from latency_meta_mdp.policy.conditioning import conditioning_contract

    conditioning_contract(job.get("input_mode", "current_and_forecast"))
    job["clean_job"] = (path.resolve().parent / job["clean_job"]).resolve()
    job["clean"] = load_training_job(job["clean_job"])
    if job["schema_version"] == 1:
        job["split_manifest"] = (project_root / job["split_manifest"]).resolve()
    elif (
        job.get("task_id") != "conveyor_sort"
        or job["clean"].get("task_id") != "conveyor_sort"
        or "conditioned_source_epochs" not in job
    ):
        raise ValueError("conveyor forecast job requires its task identity and source-epoch budget")
    for key in ("predictor_dir", "decoder_dir", "terminal_dir"):
        if key in job:
            job[key] = (project_root / job[key]).resolve()
    return job


def stage_source(job: dict, target: Path) -> Path:
    """Materialize only declared source files; HF's cache stays outside the corpus."""
    from huggingface_hub import hf_hub_download, snapshot_download

    manifest_path = Path(
        hf_hub_download(
            job["source_repo"],
            "manifest.json",
            repo_type="dataset",
            revision=job["source_revision"],
        )
    )
    manifest = json.loads(manifest_path.read_text())
    if (target / "manifest.json").exists():
        if sha256_file(target / "manifest.json") != sha256_file(manifest_path):
            raise ValueError("Existing source revision mismatch")
        for name, meta in manifest["artifacts"].items():
            p = target / name
            if not p.is_file() or p.stat().st_size != meta["bytes"]:
                raise ValueError("Source files missing/truncated")
        return target
    snapshot = Path(
        snapshot_download(
            job["source_repo"],
            repo_type="dataset",
            revision=job["source_revision"],
            allow_patterns=["manifest.json", *manifest["artifacts"]],
        )
    )
    target.mkdir(parents=True, exist_ok=True)
    for name in [*manifest["artifacts"], "manifest.json"]:
        path = target / name
        if not path.resolve().is_relative_to(target.resolve()):
            raise ValueError("Invalid source path")
        path.parent.mkdir(parents=True, exist_ok=True)
        source = (snapshot / name).resolve()
        if path.exists() and path.stat().st_size == source.stat().st_size:
            continue
        temporary = path.with_name(path.name + ".copying")
        temporary.unlink(missing_ok=True)
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
        temporary.replace(path)
    return target


def download_forecast_models(job: dict, work_dir: Path) -> tuple[Path, Path]:
    from huggingface_hub import snapshot_download

    paths = []
    for kind in ("predictor", "decoder"):
        if kind + "_dir" in job:
            path = Path(job[kind + "_dir"])
            if not path.is_dir():
                raise FileNotFoundError(path)
            paths.append(path)
            continue
        path = work_dir / "models" / job[kind + "_revision"]
        snapshot_download(job[kind + "_repo"], revision=job[kind + "_revision"], local_dir=path)
        paths.append(path)
    return tuple(paths)


def forecast_bindings(predictor: Path, decoder: Path) -> dict:
    return {
        "direct_checkpoint_sha256": sha256_file(
            predictor / "checkpoints/epoch-075/model.safetensors"
        ),
        "normalization_sha256": sha256_file(predictor / "proprio_normalization.json"),
        "decoder_sha256": sha256_file(decoder / "model.safetensors"),
    }


def verify_forecast_models(job, bindings, work_dir):
    """Do not let a cache define its own expected model identity."""
    predictor, decoder = download_forecast_models(job, work_dir)
    expected = forecast_bindings(predictor, decoder)
    pairs = {
        "predictor_sha256": "direct_checkpoint_sha256",
        "jepa_normalization_sha256": "normalization_sha256",
        "decoder_sha256": "decoder_sha256",
    }
    if any(bindings.get(key) != expected[value] for key, value in pairs.items()):
        raise ValueError("forecast cache differs from the job's pinned model assets")
