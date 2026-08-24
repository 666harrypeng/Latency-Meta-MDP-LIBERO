"""Atomic per-level training run for frozen-vision state probes."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from latency_meta_mdp.artifacts import collect_implementation_provenance, sha256_file
from latency_meta_mdp.vision_encoder import load_vision_encoder_spec
from latency_meta_mdp.vision_probe_config import load_vision_probe_config
from latency_meta_mdp.vision_probe_corpus import load_level_probe_corpus
from latency_meta_mdp.vision_probe_training import train_level_state_probe


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def train_vision_state_probe_run(
    *,
    project_root: Path,
    source_bulk_manifest: Path,
    cache_run_manifest: Path,
    vision_config_path: Path,
    probe_config_path: Path,
    output_dir: Path,
    levels: tuple[int, ...],
    device: str,
) -> Path:
    selected_levels = tuple(sorted(set(levels)))
    if not selected_levels or selected_levels != levels or any(
        level not in (1, 2, 3) for level in levels
    ):
        raise ValueError("state probe levels must be sorted unique values from 1, 2, 3")
    source_path = source_bulk_manifest.resolve()
    cache_path = cache_run_manifest.resolve()
    vision_path = vision_config_path.resolve()
    probe_path = probe_config_path.resolve()
    for path in (source_path, cache_path, vision_path, probe_path):
        if not path.is_file():
            raise FileNotFoundError(f"state probe input does not exist: {path}")
    spec = load_vision_encoder_spec(vision_path)
    config = load_vision_probe_config(probe_path)
    provenance = collect_implementation_provenance(project_root)
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"state probe run already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f"{target.name}.building-{uuid.uuid4().hex}"
    building.mkdir()
    level_manifests = {}
    try:
        for level in selected_levels:
            corpus = load_level_probe_corpus(
                source_bulk_manifest=source_path,
                cache_run_manifest=cache_path,
                expected_spec=spec,
                level=level,
                history_sample_count=config.history_sample_count,
            )
            level_manifest = train_level_state_probe(
                corpus=corpus,
                config=config,
                output_dir=building / f"L{level}",
                device=device,
            )
            level_manifests[f"L{level}"] = level_manifest.relative_to(building).as_posix()
        artifacts = {
            path.relative_to(building).as_posix(): sha256_file(path)
            for path in sorted(building.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "schema_version": 1,
            "format_id": "temporal_state_probe_run_v1",
            "checkpoint_scope": "per_level_only",
            "eligible": not provenance.dirty,
            "blockers": [] if not provenance.dirty else ["implementation_worktree_dirty"],
            "implementation_revision": provenance.revision,
            "implementation_source_sha256": provenance.source_sha256,
            "implementation_dirty": provenance.dirty,
            "source_bulk_manifest_sha256": sha256_file(source_path),
            "cache_run_manifest_sha256": sha256_file(cache_path),
            "vision_config_sha256": sha256_file(vision_path),
            "probe_config_sha256": sha256_file(probe_path),
            "encoder_id": spec.encoder_id,
            "encoder_fingerprint": spec.fingerprint,
            "probe_id": config.probe_id,
            "levels": list(selected_levels),
            "level_manifests": level_manifests,
            "artifacts": artifacts,
        }
        _write_json(building / "manifest.json", manifest)
        if target.exists():
            raise FileExistsError(f"state probe run already exists: {target}")
        os.rename(building, target)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return target / "manifest.json"
