"""Isolated patched OpenPI source runtime without dirtying the pinned submodule."""

from __future__ import annotations

import contextlib
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

from latency_meta_mdp.openpi_patch import apply_openpi_patch


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *args),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


@contextlib.contextmanager
def temporary_patched_openpi_worktree(
    *,
    openpi_root: Path,
    patch_path: Path,
    expected_revision: str,
    additional_patch_paths: tuple[Path, ...] = (),
) -> Iterator[Path]:
    """Yield an exact-revision patched worktree and remove it on every exit path."""

    root = openpi_root.resolve()
    patch = patch_path.resolve()
    additional_patches = tuple(path.resolve() for path in additional_patch_paths)
    if not root.is_dir() or not (root / ".git").exists():
        raise ValueError("openpi_root must be a Git checkout")
    if not patch.is_file():
        raise FileNotFoundError(f"OpenPI patch does not exist: {patch}")
    for additional_patch in additional_patches:
        if not additional_patch.is_file():
            raise FileNotFoundError(
                f"additional OpenPI patch does not exist: {additional_patch}"
            )
    revision = _git(root, "rev-parse", "HEAD").stdout.strip()
    if revision != expected_revision:
        raise ValueError(
            f"OpenPI revision mismatch: expected {expected_revision}, found {revision}"
        )
    if _git(root, "status", "--porcelain").stdout.strip():
        raise ValueError("canonical OpenPI checkout must be clean")

    temporary_root = Path(tempfile.mkdtemp(prefix="metamdp-openpi-"))
    worktree = temporary_root / "openpi"
    added = False
    try:
        _git(root, "worktree", "add", str(worktree), "--detach", expected_revision)
        added = True
        apply_openpi_patch(
            openpi_root=worktree,
            patch_path=patch,
            expected_revision=expected_revision,
        )
        for additional_patch in additional_patches:
            _git(worktree, "apply", "--check", str(additional_patch))
            _git(worktree, "apply", str(additional_patch))
        yield worktree
    finally:
        if added:
            _git(root, "worktree", "remove", "--force", str(worktree))
        shutil.rmtree(temporary_root, ignore_errors=True)
