from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from latency_meta_mdp.cli.run_pi05_sft import main as run_sft_main
from latency_meta_mdp.openpi_runtime import temporary_patched_openpi_worktree
from latency_meta_mdp.sft_launch import SFTLaunchRequest
from latency_meta_mdp.sft_profile import load_sft_profile


def test_level_train_config_preserves_formal_schedule_and_smoke_override(
    tmp_path: Path,
) -> None:
    profile = load_sft_profile(Path("configs/policy/pi05_panda_ball_full_sft_h50_v2.yaml"))
    with temporary_patched_openpi_worktree(
        openpi_root=Path("third_party/openpi"),
        patch_path=Path("patches/openpi/0001-filter-incomplete-action-chunks.patch"),
        expected_revision=profile.openpi_revision,
        additional_patch_paths=(Path("patches/openpi/0002-save-completed-step-checkpoints.patch"),),
    ) as worktree:
        sys.path.insert(0, str(worktree / "src"))
        try:
            from latency_meta_mdp.openpi_sft import build_level_train_config

            smoke = build_level_train_config(
                profile=profile,
                request=SFTLaunchRequest(
                    1,
                    "l1-smoke",
                    "smoke",
                    False,
                    1,
                    batch_size_override=256,
                ),
                assets_root=tmp_path / "assets",
                checkpoint_root=tmp_path / "checkpoints",
                wandb_enabled=False,
            )
            smoke_resume = build_level_train_config(
                profile=profile,
                request=SFTLaunchRequest(1, "l1-smoke", "smoke", True, 1),
                assets_root=tmp_path / "assets",
                checkpoint_root=tmp_path / "checkpoints",
                wandb_enabled=False,
            )
            formal = build_level_train_config(
                profile=profile,
                request=SFTLaunchRequest(2, "l2-formal", "formal", True, 1),
                assets_root=tmp_path / "assets",
                checkpoint_root=tmp_path / "checkpoints",
                wandb_enabled=True,
            )
        finally:
            sys.path.remove(str(worktree / "src"))

    assert smoke.name == "pi05_panda_ball_l1_full_h50"
    assert smoke.exp_name == "l1-smoke"
    assert smoke.model.pi05 is True
    assert smoke.model.action_horizon == 50
    assert smoke.data.repo_id == profile.levels[1].repo_id
    assert smoke.batch_size == 256
    assert smoke.num_train_steps == 100
    assert smoke.save_interval == 100
    assert smoke.keep_period == 100
    assert smoke.fsdp_devices == 1
    assert smoke.resume is False
    assert smoke.wandb_enabled is False

    assert smoke_resume.num_train_steps == 120
    assert smoke_resume.save_interval == 20
    assert smoke_resume.keep_period == 20
    assert smoke_resume.resume is True
    assert smoke_resume.wandb_enabled is False

    assert formal.name == "pi05_panda_ball_l2_full_h50"
    assert formal.exp_name == "l2-formal"
    assert formal.num_train_steps == 12000
    assert formal.save_interval == 4000
    assert formal.keep_period == 4000
    assert formal.resume is True
    assert formal.wandb_enabled is True
    assert formal.assets_base_dir == str((tmp_path / "assets").resolve())
    assert formal.checkpoint_base_dir == str((tmp_path / "checkpoints").resolve())


def test_sft_launcher_stages_assets_and_publishes_terminal_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    asset_source = Path("outputs/sft/norm_stats/franka-moving-ball-h50-32cea2f/L1")
    if not asset_source.is_dir():
        pytest.skip("launcher integration requires local formal L1 norm assets")

    def download(*, repo_id: str, revision: str, path_in_repo: str) -> Path:
        del repo_id, revision
        return asset_source / Path(path_in_repo).name

    def download_dataset(*, repo_id: str, revision: str, local_dir: Path) -> Path:
        del revision
        source = (
            Path("outputs/derived/lerobot/panda-ball-formal-franka-h50-0ac4f2b")
            / repo_id
            / "metamdp_dataset.json"
        )
        local_dir.mkdir(parents=True)
        shutil.copyfile(source, local_dir / "metamdp_dataset.json")
        return local_dir

    def train(*, config, openpi_root: Path) -> None:
        assert openpi_root.is_dir()
        (Path(config.checkpoint_dir) / "100").mkdir(parents=True)

    run_dir = tmp_path / "run"
    assert (
        run_sft_main(
            [
                "--project-root",
                str(Path.cwd()),
                "--profile",
                "configs/policy/pi05_panda_ball_full_sft_h50_v2.yaml",
                "--asset-lock",
                "configs/data/franka_moving_ball_sft_assets_v1.yaml",
                "--data-patch",
                "patches/openpi/0001-filter-incomplete-action-chunks.patch",
                "--training-patch",
                "patches/openpi/0002-save-completed-step-checkpoints.patch",
                "--openpi-root",
                "third_party/openpi",
                "--level",
                "1",
                "--experiment-name",
                "l1-smoke-test",
                "--mode",
                "smoke",
                "--batch-size",
                "256",
                "--assets-root",
                str(tmp_path / "assets"),
                "--dataset-root",
                str(tmp_path / "datasets"),
                "--checkpoint-root",
                str(tmp_path / "checkpoints"),
                "--run-dir",
                str(run_dir),
                "--wandb-disabled",
            ],
            asset_downloader=download,
            dataset_downloader=download_dataset,
            train_backend=train,
            device_count=1,
        )
        == 0
    )

    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"manifest": str(run_dir / "manifest.json")}
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format_id"] == "metamdp_pi05_sft_run_v1"
    assert manifest["level"] == 1
    assert manifest["mode"] == "smoke"
    assert manifest["num_train_steps"] == 100
    assert manifest["batch_size"] == 256
    assert manifest["checkpoint_steps"] == [100]
    assert "[sft][L1] start mode=smoke" in captured.err
    assert "[sft][L1] done checkpoints=100" in captured.err
