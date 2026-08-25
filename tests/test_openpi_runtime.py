from __future__ import annotations

import subprocess
from pathlib import Path

from latency_meta_mdp.openpi_runtime import temporary_patched_openpi_worktree


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def test_temporary_patched_openpi_worktree_preserves_canonical_checkout() -> None:
    root = Path("third_party/openpi").resolve()
    patch = Path("patches/openpi/0001-filter-incomplete-action-chunks.patch").resolve()
    revision = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
    before = _git(root, "status", "--porcelain")
    temporary = None

    with temporary_patched_openpi_worktree(
        openpi_root=root,
        patch_path=patch,
        expected_revision=revision,
    ) as worktree:
        temporary = worktree
        assert worktree != root
        assert _git(worktree, "rev-parse", "HEAD").strip() == revision
        assert _git(worktree, "apply", "--reverse", "--check", str(patch)) == ""
        assert set(
            line[3:] for line in _git(worktree, "status", "--porcelain").splitlines()
        ) == {
            "scripts/compute_norm_stats.py",
            "src/openpi/training/config.py",
            "src/openpi/training/data_loader.py",
            "src/openpi/training/data_loader_test.py",
        }

    assert temporary is not None
    assert not temporary.exists()
    assert _git(root, "status", "--porcelain") == before
