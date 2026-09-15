"""Publish only sanitized, completed SFT milestone checkpoints to a public HF repo."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
from typing import Any

from latency_meta_mdp.legacy.policy.sft_publish import (
    LICENSE_BYTES,
    README_BYTES,
    MilestonePublishPlan,
    validate_remote_checkpoint_files,
    write_publish_status,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--milestones", type=int, nargs="+", required=True)
    parser.add_argument("--status-path", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser


def main(argv: list[str] | None = None, *, api: Any | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = MilestonePublishPlan(
        checkpoint_root=args.checkpoint_root,
        repo_id=args.repo_id,
        milestones=tuple(args.milestones),
        status_path=args.status_path,
    )
    if args.num_workers <= 0:
        raise ValueError("HF upload worker count must be positive")
    try:
        plan.validate_local()
        if api is None:
            token = os.environ.get("HF_TOKEN")
            if not token:
                raise RuntimeError("HF_TOKEN is not set")
            from huggingface_hub import HfApi

            api = HfApi(token=token)
        api.create_repo(
            repo_id=plan.repo_id,
            repo_type="model",
            private=False,
            exist_ok=True,
        )
        info = api.model_info(plan.repo_id)
        if info.private or getattr(info, "gated", False) not in (False, None):
            raise RuntimeError("HF checkpoint repo must be public and non-gated")
        write_publish_status(plan.status_path, "repo_ready", repo_id=plan.repo_id)
        for path, payload in (("README.md", README_BYTES), ("LICENSE", LICENSE_BYTES)):
            api.upload_file(
                path_or_fileobj=io.BytesIO(payload),
                path_in_repo=path,
                repo_id=plan.repo_id,
                repo_type="model",
                commit_message=f"chore: add {path}",
            )
        write_publish_status(plan.status_path, "uploading", repo_id=plan.repo_id)
        api.upload_large_folder(
            repo_id=plan.repo_id,
            repo_type="model",
            folder_path=plan.checkpoint_root,
            allow_patterns=list(plan.allow_patterns),
            num_workers=args.num_workers,
        )
        info = api.model_info(plan.repo_id)
        if info.private or getattr(info, "gated", False) not in (False, None):
            raise RuntimeError("HF checkpoint repo visibility changed during upload")
        files = set(api.list_repo_files(plan.repo_id, repo_type="model"))
        validate_remote_checkpoint_files(files, milestones=plan.milestones)
        write_publish_status(
            plan.status_path,
            "complete",
            repo_id=plan.repo_id,
            repo_sha=info.sha,
            file_count=len(files),
            milestones=list(plan.milestones),
        )
    except BaseException as error:
        write_publish_status(
            plan.status_path,
            "failed",
            repo_id=plan.repo_id,
            error=f"{type(error).__name__}: {error}",
        )
        raise
    print(json.dumps({"repo_id": plan.repo_id, "status": str(plan.status_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
