"""Sanitized public publishing contract for completed SFT milestones."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

README_BYTES = b""
LICENSE_BYTES = (
    b"Copyright (c) 2026. All Rights Reserved.\n\n"
    b"The model weights, checkpoints, and all other contents of this repository are "
    b"proprietary. No license or permission is granted to any person or entity to use, "
    b"copy, reproduce, modify, distribute, publish, sublicense, sell, train on, evaluate, "
    b"benchmark, reverse engineer, create derivative works from, or otherwise exploit any "
    b"portion of this repository.\n\n"
    b"Access to, possession of, or ability to download these files does not grant any rights. "
    b"Any use requires prior written permission from the copyright holder, except to the "
    b"extent that a right cannot lawfully be restricted under applicable law.\n"
)
_REPO_ID = re.compile(r"^[^/\s]+/[^/\s]+$")
_PUBLIC_ROOT_FILES = frozenset({".gitattributes", "README.md", "LICENSE"})


@dataclass(frozen=True)
class MilestonePublishPlan:
    checkpoint_root: Path
    repo_id: str
    milestones: tuple[int, ...]
    status_path: Path

    def __post_init__(self) -> None:
        milestones = tuple(self.milestones)
        if _REPO_ID.fullmatch(self.repo_id) is None:
            raise ValueError("HF model repo id is invalid")
        if (
            not milestones
            or any(
                isinstance(step, bool) or not isinstance(step, int) or step <= 0
                for step in milestones
            )
            or tuple(sorted(set(milestones))) != milestones
        ):
            raise ValueError("SFT milestones must be sorted unique positive integers")
        object.__setattr__(self, "checkpoint_root", self.checkpoint_root.resolve())
        object.__setattr__(self, "status_path", self.status_path.resolve())
        object.__setattr__(self, "milestones", milestones)

    @property
    def allow_patterns(self) -> tuple[str, ...]:
        return tuple(f"{step}/**" for step in self.milestones)

    def validate_local(self) -> None:
        if not self.checkpoint_root.is_dir():
            raise FileNotFoundError(f"SFT checkpoint root does not exist: {self.checkpoint_root}")
        temporary = [
            path for path in self.checkpoint_root.rglob("*") if "orbax-checkpoint-tmp" in path.name
        ]
        if temporary:
            raise ValueError("SFT checkpoint root contains temporary Orbax paths")
        numeric = tuple(
            sorted(
                int(path.name)
                for path in self.checkpoint_root.iterdir()
                if path.is_dir() and path.name.isdigit()
            )
        )
        if numeric != self.milestones:
            raise ValueError("SFT checkpoint root has the wrong milestone inventory")
        for step in self.milestones:
            directory = self.checkpoint_root / str(step)
            required_file = directory / "_CHECKPOINT_METADATA"
            if not required_file.is_file():
                raise ValueError(f"milestone {step} is missing checkpoint metadata")
            for name in ("params", "train_state"):
                subtree = directory / name
                if not subtree.is_dir() or not any(path.is_file() for path in subtree.rglob("*")):
                    raise ValueError(f"milestone {step} has an incomplete {name} subtree")
            norm_stats = tuple((directory / "assets").rglob("norm_stats.json"))
            if len(norm_stats) != 1 or not norm_stats[0].is_file():
                raise ValueError(f"milestone {step} must contain exactly one norm_stats.json")


def validate_remote_checkpoint_files(files: set[str], *, milestones: tuple[int, ...]) -> None:
    allowed_prefixes = tuple(f"{step}/" for step in milestones)
    unexpected = sorted(
        path
        for path in files
        if path not in _PUBLIC_ROOT_FILES and not path.startswith(allowed_prefixes)
    )
    if unexpected:
        raise ValueError(f"public checkpoint repo contains unexpected files: {unexpected}")
    missing_root = sorted(_PUBLIC_ROOT_FILES - files)
    if missing_root:
        raise ValueError(f"public checkpoint repo is missing files: {missing_root}")
    for step in milestones:
        prefix = f"{step}/"
        required = (
            f"{prefix}_CHECKPOINT_METADATA",
            f"{prefix}params/",
            f"{prefix}train_state/",
            f"{prefix}assets/",
        )
        if required[0] not in files or any(
            not any(path.startswith(item) for path in files) for item in required[1:]
        ):
            raise ValueError(f"public checkpoint repo has an incomplete milestone {step}")
        if not any(
            path.startswith(required[3]) and path.endswith("/norm_stats.json") for path in files
        ):
            raise ValueError(f"public checkpoint milestone {step} is missing norm stats")


def write_publish_status(path: Path, phase: str, **extra: Any) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    payload = {"phase": phase, "updated_unix": time.time(), **extra}
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)
