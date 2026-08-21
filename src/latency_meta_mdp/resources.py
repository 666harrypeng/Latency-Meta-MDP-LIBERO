"""Third-party and project-owned resource provenance contracts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ResourceRecord:
    resource_id: str
    source_repository: str
    source_revision: str
    source_path: str
    local_path: str
    upstream_origin: str
    sha256: str
    license_id: str
    license_url: str
    license_text_path: str
    attribution: str
    modifications: str

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        required_strings = {
            "resource_id": self.resource_id,
            "source_repository": self.source_repository,
            "source_revision": self.source_revision,
            "source_path": self.source_path,
            "local_path": self.local_path,
            "upstream_origin": self.upstream_origin,
            "license_id": self.license_id,
            "license_url": self.license_url,
            "license_text_path": self.license_text_path,
            "attribution": self.attribution,
            "modifications": self.modifications,
        }
        for name, value in required_strings.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not _SHA256_RE.fullmatch(self.sha256):
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
        if self.source_repository == "project-owned":
            if not self.source_revision.startswith("project-owned-"):
                raise ValueError("project-owned revision must start with 'project-owned-'")
        elif not re.fullmatch(r"[0-9a-f]{40}", self.source_revision):
            raise ValueError("third-party source_revision must be a full 40-character commit")
        for name, value in {
            "local_path": self.local_path,
            "license_text_path": self.license_text_path,
        }.items():
            path = PurePosixPath(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"{name} must be a repository-relative path")

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


def load_resource_manifest(
    path: str | Path, *, repository_root: str | Path
) -> tuple[ResourceRecord, ...]:
    """Load a resource manifest and verify every referenced local artifact."""
    manifest_path = Path(path)
    root = Path(repository_root).resolve()
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1 or not isinstance(raw.get("resources"), list):
        raise ValueError("resource manifest must use schema_version 1 and a resources list")
    records = tuple(ResourceRecord(**item) for item in raw["resources"])
    ids = [record.resource_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("resource manifest contains duplicate resource IDs")
    for record in records:
        local_path = (root / record.local_path).resolve()
        license_path = (root / record.license_text_path).resolve()
        if root not in local_path.parents or root not in license_path.parents:
            raise ValueError(f"resource {record.resource_id} escapes repository root")
        if not local_path.is_file():
            raise ValueError(f"resource file is missing: {record.local_path}")
        if not license_path.is_file():
            raise ValueError(f"license text is missing: {record.license_text_path}")
        actual = hashlib.sha256(local_path.read_bytes()).hexdigest()
        if actual != record.sha256:
            raise ValueError(f"resource checksum mismatch: {record.resource_id}")
    return records
