from __future__ import annotations

import dataclasses
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from latency_meta_mdp.openpi_runtime import temporary_patched_openpi_worktree


def _load_train_script(path: Path):
    spec = importlib.util.spec_from_file_location("_metamdp_patched_train", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_training_patch_saves_exact_completed_update_counts(tmp_path: Path) -> None:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    root = Path("third_party/openpi").resolve()
    before = subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    data_patch = Path(
        "patches/openpi/0001-filter-incomplete-action-chunks.patch"
    ).resolve()
    training_patch = Path(
        "patches/openpi/0002-save-completed-step-checkpoints.patch"
    ).resolve()

    with temporary_patched_openpi_worktree(
        openpi_root=root,
        patch_path=data_patch,
        expected_revision="15a9616a00943ada6c20a0f158e3adb39df2ccac",
        additional_patch_paths=(training_patch,),
    ) as worktree:
        sys.path.insert(0, str(worktree / "src"))
        try:
            train = _load_train_script(worktree / "scripts/train.py")
        finally:
            sys.path.remove(str(worktree / "src"))

        assert train._checkpoint_save_step(  # noqa: SLF001
            completed_step=3999,
            num_train_steps=12000,
            save_interval=4000,
        ) is None
        assert train._checkpoint_save_step(  # noqa: SLF001
            completed_step=4000,
            num_train_steps=12000,
            save_interval=4000,
        ) == 4000
        assert train._checkpoint_save_step(  # noqa: SLF001
            completed_step=8000,
            num_train_steps=12000,
            save_interval=4000,
        ) == 8000
        assert train._checkpoint_save_step(  # noqa: SLF001
            completed_step=11999,
            num_train_steps=12000,
            save_interval=4000,
        ) is None
        assert train._checkpoint_save_step(  # noqa: SLF001
            completed_step=12000,
            num_train_steps=12000,
            save_interval=4000,
        ) == 12000

        from openpi.training import config as openpi_config

        config = dataclasses.replace(
            openpi_config.get_config("debug"),
            checkpoint_base_dir=str(tmp_path / "checkpoints"),
            exp_name="completed-step-smoke",
            overwrite=False,
            resume=False,
            num_train_steps=4,
            save_interval=2,
            keep_period=2,
            wandb_enabled=False,
        )
        train.main(config)
        checkpoint_root = Path(config.checkpoint_dir)
        assert {path.name for path in checkpoint_root.iterdir() if path.name.isdigit()} == {
            "2",
            "4",
        }

        train.main(dataclasses.replace(config, resume=True, num_train_steps=6))
        assert {path.name for path in checkpoint_root.iterdir() if path.name.isdigit()} == {
            "2",
            "4",
            "6",
        }

    assert subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout == before
