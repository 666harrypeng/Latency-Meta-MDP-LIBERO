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
    if job["schema_version"] != 1:
        raise ValueError("Unsupported forecast job")
    job["clean_job"] = (path.resolve().parent / job["clean_job"]).resolve()
    job["split_manifest"] = (project_root / job["split_manifest"]).resolve()
    job["clean"] = load_training_job(job["clean_job"])
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
            token=False,
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
            token=False,
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
        path = work_dir / "models" / job[kind + "_revision"]
        snapshot_download(
            job[kind + "_repo"], revision=job[kind + "_revision"], local_dir=path, token=False
        )
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
