from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from latency_meta_mdp.cli.publish_sft_milestones import main as publish_main
from latency_meta_mdp.sft_publish import (
    LICENSE_BYTES,
    README_BYTES,
    MilestonePublishPlan,
    validate_remote_checkpoint_files,
)


def _checkpoint(root: Path, step: int) -> None:
    directory = root / str(step)
    (directory / "params").mkdir(parents=True)
    (directory / "train_state").mkdir()
    assets = directory / "assets" / "private_dataset_identity"
    assets.mkdir(parents=True)
    (directory / "_CHECKPOINT_METADATA").write_text("{}\n", encoding="utf-8")
    (directory / "params" / "value").write_bytes(b"params")
    (directory / "train_state" / "value").write_bytes(b"state")
    (assets / "norm_stats.json").write_text("{}\n", encoding="utf-8")


def test_publish_plan_selects_only_complete_milestone_subtrees(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    for step in (1333, 2666, 3999):
        _checkpoint(root, step)
    (root / ".cache").mkdir()
    (root / "wandb_id.txt").write_text("not-public\n", encoding="utf-8")

    plan = MilestonePublishPlan(
        checkpoint_root=root,
        repo_id="owner/repository",
        milestones=(1333, 2666, 3999),
        status_path=tmp_path / "status.json",
    )
    plan.validate_local()

    assert plan.allow_patterns == ("1333/**", "2666/**", "3999/**")
    assert README_BYTES == b""
    assert b"All Rights Reserved" in LICENSE_BYTES
    assert b"moving_ball" not in LICENSE_BYTES


def test_publish_plan_rejects_incomplete_or_temporary_checkpoints(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    for step in (1333, 2666, 3999):
        _checkpoint(root, step)
    (root / "2666" / "train_state" / "value").unlink()

    plan = MilestonePublishPlan(root, "owner/repository", (1333, 2666, 3999), tmp_path / "s.json")
    with pytest.raises(ValueError, match="train_state"):
        plan.validate_local()

    (root / "2666" / "train_state" / "value").write_bytes(b"state")
    (root / "3999.orbax-checkpoint-tmp-0").mkdir()
    with pytest.raises(ValueError, match="temporary"):
        plan.validate_local()


def test_remote_checkpoint_allowlist_rejects_metadata_and_logs() -> None:
    valid = {
        ".gitattributes",
        "README.md",
        "LICENSE",
        "1333/_CHECKPOINT_METADATA",
        "1333/params/value",
        "1333/train_state/value",
        "1333/assets/dataset/norm_stats.json",
        "2666/_CHECKPOINT_METADATA",
        "2666/params/value",
        "2666/train_state/value",
        "2666/assets/dataset/norm_stats.json",
        "3999/_CHECKPOINT_METADATA",
        "3999/params/value",
        "3999/train_state/value",
        "3999/assets/dataset/norm_stats.json",
    }
    validate_remote_checkpoint_files(valid, milestones=(1333, 2666, 3999))

    with pytest.raises(ValueError, match="unexpected"):
        validate_remote_checkpoint_files(
            valid | {"metamdp_metadata/run_manifest.json"},
            milestones=(1333, 2666, 3999),
        )
    with pytest.raises(ValueError, match="unexpected"):
        validate_remote_checkpoint_files(
            valid | {"wandb_id.txt"},
            milestones=(1333, 2666, 3999),
        )


class _FakeHubApi:
    def __init__(self, files: set[str]) -> None:
        self.files = files
        self.created: dict | None = None
        self.uploaded_small: dict[str, bytes] = {}
        self.large_upload: dict | None = None

    def create_repo(self, **kwargs):
        self.created = kwargs

    def model_info(self, repo_id: str):
        del repo_id
        return SimpleNamespace(private=False, gated=False, sha="fake-sha")

    def upload_file(self, *, path_or_fileobj, path_in_repo: str, **kwargs):
        del kwargs
        self.uploaded_small[path_in_repo] = path_or_fileobj.getvalue()

    def upload_large_folder(self, **kwargs):
        self.large_upload = kwargs

    def list_repo_files(self, repo_id: str, *, repo_type: str):
        del repo_id, repo_type
        return sorted(self.files)


def test_publish_cli_uses_sanitized_public_allowlist(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    for step in (1333, 2666, 3999):
        _checkpoint(root, step)
    (root / ".cache").mkdir()
    (root / "wandb_id.txt").write_text("not-public\n", encoding="utf-8")
    files = {
        ".gitattributes",
        "README.md",
        "LICENSE",
        *{
            path
            for step in (1333, 2666, 3999)
            for path in (
                f"{step}/_CHECKPOINT_METADATA",
                f"{step}/params/value",
                f"{step}/train_state/value",
                f"{step}/assets/private_dataset_identity/norm_stats.json",
            )
        },
    }
    api = _FakeHubApi(files)

    assert (
        publish_main(
            [
                "--checkpoint-root",
                str(root),
                "--repo-id",
                "owner/repository",
                "--milestones",
                "1333",
                "2666",
                "3999",
                "--status-path",
                str(tmp_path / "status.json"),
            ],
            api=api,
        )
        == 0
    )
    assert api.created == {
        "repo_id": "owner/repository",
        "repo_type": "model",
        "private": False,
        "exist_ok": True,
    }
    assert api.uploaded_small == {"README.md": README_BYTES, "LICENSE": LICENSE_BYTES}
    assert api.large_upload is not None
    assert api.large_upload["allow_patterns"] == ["1333/**", "2666/**", "3999/**"]
    assert "wandb_id.txt" not in api.large_upload["allow_patterns"]
    assert "metamdp_metadata/**" not in api.large_upload["allow_patterns"]
    assert '"phase": "complete"' in (tmp_path / "status.json").read_text()
