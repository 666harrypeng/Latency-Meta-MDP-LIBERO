"""Validate packaged dataset inventories without hashing large payloads."""

import json
from pathlib import Path


def verify_bundle_files(root: Path) -> dict:
    manifest = json.loads((root / "bundle.json").read_text())
    if (
        manifest["schema_version"] != 1
        or manifest["format_id"] != "structured_pi05_training_bundle_v1"
    ):
        raise ValueError("Unsupported training bundle")
    for name, size in manifest["files"].items():
        path = (root / name).resolve()
        if (
            not path.is_relative_to(root.resolve())
            or not path.is_file()
            or path.stat().st_size != size
        ):
            raise ValueError(f"Missing or truncated bundle file: {name}")
    for key in ("dataset_root", "preparation_root"):
        if not (root / manifest[key]).resolve().is_relative_to(root.resolve()):
            raise ValueError("Bundle path escapes its root")
    return manifest
