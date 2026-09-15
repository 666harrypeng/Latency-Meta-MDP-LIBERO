"""Public training bundles and inference-only checkpoint delivery."""

from __future__ import annotations

import json
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
    return job


def verify_bundle_files(root: Path) -> dict:
    manifest = json.loads((root / "bundle.json").read_text())
    if (
        manifest["schema_version"] != 1
        or manifest["format_id"] != "structured_pi05_training_bundle_v1"
    ):
        raise ValueError("Unsupported training bundle")
    for name, size in manifest["files"].items():
        path = (root / name).resolve()
        if (
            not path.is_relative_to(root.resolve())
            or not path.is_file()
            or path.stat().st_size != size
        ):
            raise ValueError(f"Missing or truncated bundle file: {name}")
    for key in ("dataset_root", "preparation_root"):
        if not (root / manifest[key]).resolve().is_relative_to(root.resolve()):
            raise ValueError("Bundle path escapes its root")
    return manifest


def inference_checkpoint_files(root: Path, steps: tuple[int, ...]) -> dict[str, Path]:
    files = {}
    for step in steps:
        checkpoint = root / str(step)
        params = list((checkpoint / "params").rglob("*"))
        assets = list((checkpoint / "assets").rglob("*"))
        if not any(p.is_file() for p in params) or not any(
            p.name == "norm_stats.json" for p in assets
        ):
            raise ValueError(f"Incomplete inference checkpoint: {step}")
        for path in params + assets + [checkpoint / "_CHECKPOINT_METADATA"]:
            if path.is_file():
                files[path.relative_to(root).as_posix()] = path
    return files


def publish_checkpoints(*, root: Path, steps: tuple[int, ...], repo_id: str, notices: Path) -> str:
    from huggingface_hub import CommitOperationAdd, HfApi

    files = inference_checkpoint_files(root, steps)
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    operations = [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path))
        for name, path in files.items()
    ]
    operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=b""))
    for name in ("LICENSE", "NOTICE", "GEMMA_TERMS.txt"):
        operations.append(
            CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(notices / name))
        )
    return api.create_commit(
        repo_id, operations=operations, commit_message="Archive clean SFT inference milestones"
    ).oid
