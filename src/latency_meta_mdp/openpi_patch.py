"""Revision-checked application of the project OpenPI integration patch."""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from latency_meta_mdp.artifacts import sha256_file
from latency_meta_mdp.lerobot_conversion import OPENPI_REVISION


class PatchState(str, Enum):
    APPLIED = "applied"
    ALREADY_APPLIED = "already_applied"


@dataclass(frozen=True)
class PatchApplication:
    state: PatchState
    openpi_revision: str
    patch_sha256: str


def _git(
    openpi_root: Path,
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *args),
        cwd=openpi_root,
        check=check,
        capture_output=True,
        text=True,
    )


def _patch_paths(openpi_root: Path, patch: Path) -> frozenset[str]:
    output = _git(openpi_root, "apply", "--numstat", str(patch)).stdout
    return frozenset(line.split("\t", maxsplit=2)[-1] for line in output.splitlines())


def _changed_paths(openpi_root: Path) -> frozenset[str]:
    output = _git(openpi_root, "status", "--porcelain", "--untracked-files=normal").stdout
    return frozenset(line[3:].split(" -> ")[-1] for line in output.splitlines())


def apply_openpi_patch(
    *,
    openpi_root: Path,
    patch_path: Path,
    expected_revision: str = OPENPI_REVISION,
) -> PatchApplication:
    """Apply one patch only to its exact clean upstream revision."""

    root = openpi_root.resolve()
    patch = patch_path.resolve()
    if not root.is_dir() or not (root / ".git").exists():
        raise ValueError("openpi_root must be a Git checkout")
    if not patch.is_file():
        raise ValueError("patch_path must be a file")

    revision = _git(root, "rev-parse", "HEAD").stdout.strip()
    if revision != expected_revision:
        raise ValueError(
            f"OpenPI revision mismatch: expected {expected_revision}, found {revision}"
        )
    patch_hash = sha256_file(patch)

    reverse_check = _git(root, "apply", "--reverse", "--check", str(patch), check=False)
    if reverse_check.returncode == 0:
        if _changed_paths(root) != _patch_paths(root, patch):
            raise ValueError(
                "an already-patched checkout may contain only the integration patch"
            )
        return PatchApplication(PatchState.ALREADY_APPLIED, revision, patch_hash)

    if _git(root, "status", "--porcelain").stdout.strip():
        raise ValueError("patch application requires a clean OpenPI checkout")

    forward_check = _git(root, "apply", "--check", str(patch), check=False)
    if forward_check.returncode != 0:
        message = forward_check.stderr.strip() or forward_check.stdout.strip()
        raise ValueError(f"OpenPI patch does not apply cleanly: {message}")
    _git(root, "apply", str(patch))
    return PatchApplication(PatchState.APPLIED, revision, patch_hash)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    args = parser.parse_args()
    application = apply_openpi_patch(
        openpi_root=args.openpi_root,
        patch_path=args.patch,
    )
    print(
        f"state={application.state.value} "
        f"openpi_revision={application.openpi_revision} "
        f"patch_sha256={application.patch_sha256}"
    )


if __name__ == "__main__":
    main()
