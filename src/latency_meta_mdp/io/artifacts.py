"""Immutable gate-artifact publication with one canonical cross-gate index."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_GATE_RE = re.compile(r"^g[0-9]+$")
_SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_REQUIRED_PREREQUISITES = {"g1": ("g0",)}


@dataclass(frozen=True)
class ImplementationProvenance:
    revision: str
    source_sha256: str
    dirty: bool


def collect_implementation_provenance(project_root: Path) -> ImplementationProvenance:
    """Hash the complete Python implementation and report its Git worktree state."""

    root = project_root.resolve()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    source_paths = (
        root / "pyproject.toml",
        root / "uv.lock",
        *sorted((root / "src/latency_meta_mdp").rglob("*.py")),
    )
    digest = hashlib.sha256()
    for path in source_paths:
        relative = path.relative_to(root).as_posix().encode()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return ImplementationProvenance(
        revision=revision,
        source_sha256=digest.hexdigest(),
        dirty=dirty,
    )


def sha256_file(path: Path) -> str:
    """Hash artifacts with bounded memory, including multi-GiB weights and caches."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_index(output_root: Path) -> dict[str, Any]:
    index_path = output_root / "artifact_index.json"
    if not index_path.exists():
        return {"gates": {}, "schema_version": 1}
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("schema_version") != 1 or not isinstance(index.get("gates"), dict):
        raise ValueError("artifact index must use schema_version 1 and a gates mapping")
    return index


def _validate_inputs(*, gate: str, run_id: str, report_name: str, report: dict[str, Any]) -> None:
    if not _GATE_RE.fullmatch(gate):
        raise ValueError("gate must use the form g followed by digits")
    if not _SAFE_COMPONENT_RE.fullmatch(run_id):
        raise ValueError("run_id must be one safe path component")
    if not _SAFE_COMPONENT_RE.fullmatch(report_name) or not report_name.endswith(".json"):
        raise ValueError("report_name must be one safe JSON filename")
    if not isinstance(report.get("eligible"), bool):
        raise ValueError("report eligible must be a boolean")
    if not isinstance(report.get("blockers"), list) or not all(
        isinstance(blocker, str) for blocker in report["blockers"]
    ):
        raise ValueError("report blockers must be a list of strings")


def resolve_gate_reference(*, output_root: Path, gate: str) -> dict[str, str]:
    """Resolve and verify one eligible indexed prerequisite and all manifest artifacts."""
    index = _load_index(output_root)
    if gate not in index["gates"]:
        raise ValueError(f"required prerequisite {gate} is missing")
    reference = index["gates"][gate]
    if not isinstance(reference, dict) or set(reference) != {"manifest", "sha256"}:
        raise ValueError(f"{gate} index reference is malformed")
    relative_manifest = reference["manifest"]
    expected_hash = reference["sha256"]
    if not isinstance(relative_manifest, str) or not isinstance(expected_hash, str):
        raise ValueError(f"{gate} index reference is malformed")
    manifest_path = (output_root / relative_manifest).resolve()
    output_resolved = output_root.resolve()
    try:
        manifest_path.relative_to(output_resolved)
    except ValueError as error:
        raise ValueError(f"{gate} manifest escapes output root") from error
    if not manifest_path.is_file():
        raise ValueError(f"{gate} manifest is missing")
    if sha256_file(manifest_path) != expected_hash:
        raise ValueError(f"{gate} manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("gate") != gate or manifest.get("eligible") is not True:
        raise ValueError(f"{gate} manifest is not eligible")
    if manifest.get("blockers") != [] or not isinstance(manifest.get("artifacts"), dict):
        raise ValueError(f"{gate} manifest contents are malformed")
    for relative_artifact, artifact_hash in manifest["artifacts"].items():
        artifact_path = (manifest_path.parent / relative_artifact).resolve()
        try:
            artifact_path.relative_to(manifest_path.parent.resolve())
        except ValueError as error:
            raise ValueError(f"{gate} artifact escapes its run directory") from error
        if not artifact_path.is_file() or sha256_file(artifact_path) != artifact_hash:
            raise ValueError(f"{gate} artifact hash mismatch: {relative_artifact}")
    return {"manifest": relative_manifest, "sha256": expected_hash}


def preflight_gate_output(
    *, output_root: Path, gate: str, run_id: str
) -> dict[str, dict[str, str]]:
    """Reject an existing run or canonical gate before expensive computation begins."""
    if not _GATE_RE.fullmatch(gate):
        raise ValueError("gate must use the form g followed by digits")
    if not _SAFE_COMPONENT_RE.fullmatch(run_id):
        raise ValueError("run_id must be one safe path component")
    run_root = output_root / "certification" / gate / run_id
    if run_root.exists():
        raise FileExistsError(f"run already exists: {run_root}")
    if gate in _load_index(output_root)["gates"]:
        raise FileExistsError(f"{gate} artifact index entry already exists")
    return {
        prerequisite: resolve_gate_reference(output_root=output_root, gate=prerequisite)
        for prerequisite in _REQUIRED_PREREQUISITES.get(gate, ())
    }


def publish_gate_report(
    *,
    output_root: Path,
    gate: str,
    run_id: str,
    report_name: str,
    report: dict[str, Any],
) -> Path:
    """Publish one no-overwrite report and index it only when eligible."""
    _validate_inputs(gate=gate, run_id=run_id, report_name=report_name, report=report)
    prerequisites = preflight_gate_output(output_root=output_root, gate=gate, run_id=run_id)
    if report.get("prerequisites", {}) != prerequisites:
        raise ValueError(f"{gate} report prerequisites do not match canonical artifact index")
    index = _load_index(output_root)
    run_root = output_root / "certification" / gate / run_id

    gate_root = run_root.parent
    gate_root.mkdir(parents=True, exist_ok=True)
    staging = gate_root / f".{run_id}.building-{os.getpid()}"
    try:
        staging.mkdir()
        report_path = staging / report_name
        _write_json(report_path, report)
        manifest = {
            "artifacts": {report_name: sha256_file(report_path)},
            "blockers": list(report["blockers"]),
            "eligible": report["eligible"],
            "gate": gate,
            "run_id": run_id,
            "schema_version": 2,
        }
        _write_json(staging / "manifest.json", manifest)
        staging.rename(run_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    if report["eligible"]:
        index["gates"][gate] = {
            "manifest": f"certification/{gate}/{run_id}/manifest.json",
            "sha256": sha256_file(run_root / "manifest.json"),
        }
        index_path = output_root / "artifact_index.json"
        index_staging = output_root / f".artifact_index.json.building-{os.getpid()}"
        try:
            _write_json(index_staging, index)
            if index_path.exists():
                os.replace(index_staging, index_path)
            else:
                index_staging.rename(index_path)
        except BaseException:
            index_staging.unlink(missing_ok=True)
            shutil.rmtree(run_root, ignore_errors=True)
            raise
    return run_root / "manifest.json"
