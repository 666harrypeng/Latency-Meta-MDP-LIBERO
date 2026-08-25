from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

from latency_meta_mdp.openpi_runtime import temporary_patched_openpi_worktree


def _run_debug_training(
    *,
    worktree: Path,
    checkpoint_base: Path,
    resume: bool,
    num_train_steps: int,
) -> None:
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    environment["OPENPI_WORKTREE"] = str(worktree)
    environment["CHECKPOINT_BASE"] = str(checkpoint_base)
    environment["RESUME"] = "1" if resume else "0"
    environment["NUM_TRAIN_STEPS"] = str(num_train_steps)
    environment["PYTHONPATH"] = os.pathsep.join((str(worktree / "src"), str(Path.cwd() / "src")))
    subprocess.run(
        (
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import dataclasses
                import importlib.util
                import os
                from pathlib import Path

                from openpi.training import config as openpi_config

                worktree = Path(os.environ["OPENPI_WORKTREE"])
                spec = importlib.util.spec_from_file_location(
                    "_metamdp_patched_train",
                    worktree / "scripts/train.py",
                )
                assert spec is not None and spec.loader is not None
                train = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(train)

                assert train._checkpoint_save_step(
                    completed_step=99,
                    num_train_steps=3999,
                    save_interval=100,
                    keep_period=1333,
                ) is None
                assert train._checkpoint_save_step(
                    completed_step=100,
                    num_train_steps=3999,
                    save_interval=100,
                    keep_period=1333,
                ) == 100
                assert train._checkpoint_save_step(
                    completed_step=1333,
                    num_train_steps=3999,
                    save_interval=100,
                    keep_period=1333,
                ) == 1333
                assert train._checkpoint_save_step(
                    completed_step=1334,
                    num_train_steps=3999,
                    save_interval=100,
                    keep_period=1333,
                ) is None
                assert train._checkpoint_save_step(
                    completed_step=3999,
                    num_train_steps=3999,
                    save_interval=100,
                    keep_period=1333,
                ) == 3999

                config = dataclasses.replace(
                    openpi_config.get_config("debug"),
                    checkpoint_base_dir=os.environ["CHECKPOINT_BASE"],
                    exp_name="completed-step-smoke",
                    overwrite=False,
                    resume=os.environ["RESUME"] == "1",
                    num_train_steps=int(os.environ["NUM_TRAIN_STEPS"]),
                    save_interval=2,
                    keep_period=3,
                    wandb_enabled=False,
                )
                train.main(config)
                """
            ),
        ),
        check=True,
        env=environment,
    )


def test_training_patch_saves_exact_completed_update_counts_on_initial_run(
    tmp_path: Path,
) -> None:
    root = Path("third_party/openpi").resolve()
    before = subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    checkpoint_base = tmp_path / "checkpoints"

    with temporary_patched_openpi_worktree(
        openpi_root=root,
        patch_path=Path("patches/openpi/0001-filter-incomplete-action-chunks.patch"),
        expected_revision="15a9616a00943ada6c20a0f158e3adb39df2ccac",
        additional_patch_paths=(Path("patches/openpi/0002-save-completed-step-checkpoints.patch"),),
    ) as worktree:
        _run_debug_training(
            worktree=worktree,
            checkpoint_base=checkpoint_base,
            resume=False,
            num_train_steps=6,
        )
    checkpoint_root = checkpoint_base / "debug" / "completed-step-smoke"
    assert {path.name for path in checkpoint_root.iterdir() if path.name.isdigit()} == {
        "3",
        "6",
    }
    assert (
        subprocess.run(
            ("git", "status", "--porcelain"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        == before
    )
