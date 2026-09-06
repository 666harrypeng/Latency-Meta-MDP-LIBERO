"""Isolated patched OpenPI source runtime without dirtying the pinned submodule."""

from __future__ import annotations

import contextlib
import io
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Iterator
from pathlib import Path

from latency_meta_mdp.openpi_patch import apply_openpi_patch


@contextlib.contextmanager
def temporary_patched_openpi_copy(
    *,
    openpi_root: Path,
    patch_paths: tuple[Path, ...],
    expected_revision: str,
) -> Iterator[Path]:
    """Patch a temporary archive of the pinned source, without creating a worktree."""
    root = openpi_root.resolve()
    if _git(root, "rev-parse", "HEAD").stdout.strip() != expected_revision:
        raise ValueError("OpenPI revision mismatch")
    if _git(root, "status", "--porcelain").stdout.strip():
        raise ValueError("canonical OpenPI checkout must be clean")
    archive = subprocess.run(
        ("git", "archive", expected_revision),
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    modules_before = set(sys.modules)
    with tempfile.TemporaryDirectory(prefix="metamdp-openpi-copy-") as directory:
        copied = Path(directory)
        try:
            with tarfile.open(fileobj=io.BytesIO(archive)) as source:
                source.extractall(copied, filter="data")
            for patch in patch_paths:
                _git(copied, "apply", "--check", str(patch.resolve()))
                _git(copied, "apply", str(patch.resolve()))
            yield copied
        finally:
            _purge_worktree_modules(copied, modules_before)


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *args),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


def _purge_worktree_modules(worktree: Path, modules_before: set[str]) -> None:
    for name, module in tuple(sys.modules.items()):
        if name in modules_before:
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        try:
            belongs_to_worktree = Path(module_file).resolve().is_relative_to(worktree)
        except (OSError, RuntimeError):
            belongs_to_worktree = False
        # Project transforms cache upstream classes/enums too. Keeping them
        # across source contexts mixes incompatible ModelType identities.
        if belongs_to_worktree or name.startswith("latency_meta_mdp.openpi_"):
            sys.modules.pop(name, None)
            parent_name, _, child_name = name.rpartition(".")
            parent = sys.modules.get(parent_name)
            if parent is not None and getattr(parent, child_name, None) is module:
                delattr(parent, child_name)


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
            raise FileNotFoundError(f"additional OpenPI patch does not exist: {additional_patch}")
    revision = _git(root, "rev-parse", "HEAD").stdout.strip()
    if revision != expected_revision:
        raise ValueError(
            f"OpenPI revision mismatch: expected {expected_revision}, found {revision}"
        )
    if _git(root, "status", "--porcelain").stdout.strip():
        raise ValueError("canonical OpenPI checkout must be clean")

    temporary_root = Path(tempfile.mkdtemp(prefix="metamdp-openpi-"))
    worktree = temporary_root / "openpi"
    modules_before = set(sys.modules)
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
        _purge_worktree_modules(worktree, modules_before)
        if added:
            _git(root, "worktree", "remove", "--force", str(worktree))
        shutil.rmtree(temporary_root, ignore_errors=True)
