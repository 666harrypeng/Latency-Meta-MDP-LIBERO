from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from latency_meta_mdp.policy.openpi.patches import PatchState, apply_openpi_patch


def _run(*args: str, cwd: Path) -> str:
    result = subprocess.run(
        args,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "openpi"
    repository.mkdir()
    _run("git", "init", "-q", cwd=repository)
    _run("git", "config", "user.name", "Test User", cwd=repository)
    _run("git", "config", "user.email", "test@example.com", cwd=repository)
    (repository / "value.txt").write_text("before\n", encoding="utf-8")
    _run("git", "add", "value.txt", cwd=repository)
    _run("git", "commit", "-q", "-m", "initial", cwd=repository)
    return repository, _run("git", "rev-parse", "HEAD", cwd=repository)


def _patch(path: Path) -> Path:
    patch = path / "change.patch"
    patch.write_text(
        """diff --git a/value.txt b/value.txt
--- a/value.txt
+++ b/value.txt
@@ -1 +1 @@
-before
+after
""",
        encoding="utf-8",
    )
    return patch


def test_apply_openpi_patch_is_revision_checked_and_idempotent(tmp_path: Path) -> None:
    repository, revision = _repository(tmp_path)
    patch = _patch(tmp_path)

    first = apply_openpi_patch(
        openpi_root=repository,
        patch_path=patch,
        expected_revision=revision,
    )
    second = apply_openpi_patch(
        openpi_root=repository,
        patch_path=patch,
        expected_revision=revision,
    )

    assert first.state is PatchState.APPLIED
    assert second.state is PatchState.ALREADY_APPLIED
    assert first.patch_sha256 == second.patch_sha256
    assert (repository / "value.txt").read_text(encoding="utf-8") == "after\n"


def test_apply_openpi_patch_rejects_an_unrelated_dirty_checkout(tmp_path: Path) -> None:
    repository, revision = _repository(tmp_path)
    patch = _patch(tmp_path)
    (repository / "unrelated.txt").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(ValueError, match="clean OpenPI checkout"):
        apply_openpi_patch(
            openpi_root=repository,
            patch_path=patch,
            expected_revision=revision,
        )


def test_apply_openpi_patch_rejects_unrelated_changes_after_application(
    tmp_path: Path,
) -> None:
    repository, revision = _repository(tmp_path)
    patch = _patch(tmp_path)
    apply_openpi_patch(
        openpi_root=repository,
        patch_path=patch,
        expected_revision=revision,
    )
    (repository / "unrelated.txt").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(ValueError, match="only the integration patch"):
        apply_openpi_patch(
            openpi_root=repository,
            patch_path=patch,
            expected_revision=revision,
        )
