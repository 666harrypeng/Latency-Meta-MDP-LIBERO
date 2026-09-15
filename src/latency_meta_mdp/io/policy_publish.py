"""Publish policy inference checkpoints with normalization and upstream notices."""

from __future__ import annotations

from pathlib import Path


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
