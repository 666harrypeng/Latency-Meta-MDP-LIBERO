"""Audited structured-source runner for frozen vision feature caches."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import uuid
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

import numpy as np

from latency_meta_mdp.artifacts import ImplementationProvenance
from latency_meta_mdp.expert_realization.artifacts import (
    _cleanup_owned_staging,
    _fsync_directory,
    _hash_file,
    _rename_noreplace,
    _write_file_fsynced,
)
from latency_meta_mdp.expert_realization.source_corpus.parquet import decode_png
from latency_meta_mdp.vision_encoder import VisionEncoderSpec
from latency_meta_mdp.vision_feature_cache import (
    CAMERA_ORDER,
    VisionFeatureEncoder,
    load_episode_vision_feature_cache,
    write_episode_vision_feature_cache,
)

_RUN_FORMAT_ID = "vision_feature_cache_run_v2"
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUN_FIELDS = frozenset(
    {
        "schema_version",
        "format_id",
        "eligible",
        "blockers",
        "implementation",
        "source_corpus_id",
        "source_corpus_manifest_sha256",
        "encoder_id",
        "model_id",
        "model_revision",
        "weights_sha256",
        "encoder_fingerprint",
        "runtime",
        "levels",
        "episode_range",
        "boundary_batch_size",
        "maximum_image_batch_size",
        "episode_count",
        "boundary_count",
        "feature_payload_bytes",
        "feature_artifact_bytes",
        "episodes",
        "artifacts",
    }
)
_EPISODE_FIELDS = frozenset(
    {
        "episode_id",
        "task_instance_id",
        "logical_master_task_index",
        "level",
        "accepted_slot",
        "realization_draw_index",
        "boundary_count",
        "source_episode_metadata_sha256",
        "cache_manifest",
    }
)
_PROVENANCE_PATHS = (
    "pyproject.toml",
    "requirements/action-conditioned-jepa.in",
    "requirements/action-conditioned-jepa.lock",
    "src/latency_meta_mdp/hf_dino_encoder.py",
    "src/latency_meta_mdp/vision_encoder.py",
    "src/latency_meta_mdp/vision_feature_cache.py",
    "src/latency_meta_mdp/vision_feature_cache_run.py",
    "src/latency_meta_mdp/cli/cache_vision_features.py",
    "src/latency_meta_mdp/expert_realization/artifacts.py",
    "src/latency_meta_mdp/expert_realization/source_corpus/loader.py",
    "src/latency_meta_mdp/expert_realization/source_corpus/parquet.py",
)


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _metadata_sha256(metadata: Any) -> str:
    if not hasattr(metadata, "items"):
        raise TypeError("source episode metadata must be a mapping")
    payload = json.dumps(
        dict(metadata),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _safe_relative(value: Any) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise ValueError("cache artifact path must be normalized and relative")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or "." in path.parts or ".." in path.parts:
        raise ValueError("cache artifact path must be normalized and relative")
    return value


def _validate_selection(
    *,
    levels: tuple[int, ...],
    episode_range: tuple[int, int] | None,
    boundary_batch_size: int,
) -> tuple[int, ...]:
    if (
        type(levels) is not tuple
        or not levels
        or levels != tuple(sorted(set(levels)))
        or any(type(level) is not int or level not in (1, 2, 3) for level in levels)
    ):
        raise ValueError("vision cache levels must be sorted unique values from 1, 2, 3")
    if type(boundary_batch_size) is not int or boundary_batch_size <= 0:
        raise ValueError("boundary_batch_size must be a positive integer")
    if episode_range is not None and (
        type(episode_range) is not tuple
        or len(episode_range) != 2
        or any(type(value) is not int for value in episode_range)
        or episode_range[0] < 0
        or episode_range[1] <= episode_range[0]
    ):
        raise ValueError("episode_range must be a non-empty half-open integer range")
    return levels


def _collect_vision_cache_provenance(project_root: Path) -> ImplementationProvenance:
    """Bind the cache to its exact implementation without unrelated worktree noise."""

    root = project_root.resolve()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", *_PROVENANCE_PATHS],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    digest = hashlib.sha256()
    for relative in _PROVENANCE_PATHS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"vision cache implementation input is missing: {relative}")
        relative_bytes = relative.encode()
        payload = path.read_bytes()
        digest.update(len(relative_bytes).to_bytes(4, "big"))
        digest.update(relative_bytes)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return ImplementationProvenance(
        revision=revision,
        source_sha256=digest.hexdigest(),
        dirty=bool(status.strip()),
    )


def _decode_source_episode(source_episode: Any) -> Any:
    metadata = source_episode.metadata
    required = {
        "episode_id",
        "task_instance_id",
        "logical_master_task_index",
        "level",
        "accepted_slot",
        "realization_draw_index",
        "row_count",
    }
    if not hasattr(metadata, "keys") or not required.issubset(metadata.keys()):
        raise ValueError("source episode metadata lacks cache identity fields")
    frames = source_episode.frames
    if frames.num_rows != metadata["row_count"] or not {
        "agentview_rgb",
        "wrist_rgb",
    }.issubset(frames.column_names):
        raise ValueError("source episode frames disagree with cache metadata")

    def decode_camera(name: str) -> np.ndarray:
        values = frames[name].to_pylist()
        if any(
            type(value) is not dict or type(value.get("bytes")) is not bytes for value in values
        ):
            raise ValueError(f"source episode {name} column is not canonical PNG data")
        return np.stack([decode_png(value["bytes"]) for value in values])

    return SimpleNamespace(
        episode_id=metadata["episode_id"],
        task_instance_id=metadata["task_instance_id"],
        logical_master_task_index=metadata["logical_master_task_index"],
        level=metadata["level"],
        accepted_slot=metadata["accepted_slot"],
        realization_draw_index=metadata["realization_draw_index"],
        boundary_count=metadata["row_count"],
        deployment=SimpleNamespace(
            agentview_rgb=decode_camera("agentview_rgb"),
            wrist_rgb=decode_camera("wrist_rgb"),
        ),
    )


def write_vision_feature_cache_run(
    *,
    project_root: Path,
    source_root: Path,
    encoder: VisionFeatureEncoder,
    output_dir: Path,
    levels: tuple[int, ...],
    episode_range: tuple[int, int] | None,
    boundary_batch_size: int,
    load_fn: Callable[[Path], Any] | None = None,
    provenance_fn: Callable[[Path], ImplementationProvenance] = _collect_vision_cache_provenance,
    progress_fn: Callable[[str], None] | None = None,
) -> Path:
    """Extract one immutable cache run from the canonical structured source."""

    selected_levels = _validate_selection(
        levels=levels,
        episode_range=episode_range,
        boundary_batch_size=boundary_batch_size,
    )
    if load_fn is None:
        from latency_meta_mdp.expert_realization.source_corpus.loader import (
            load_verified_source_corpus,
        )

        load_fn = load_verified_source_corpus
    source_root = Path(source_root).resolve()
    source_manifest = source_root / "manifest.json"
    source_manifest_sha = _hash_file(source_manifest)
    corpus = load_fn(source_root)
    if Path(corpus.root).resolve() != source_root:
        raise ValueError("verified source corpus root changed during loading")
    provenance = provenance_fn(Path(project_root))
    target = Path(output_dir).absolute()
    if target.exists():
        raise FileExistsError(f"vision feature cache run already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    parent_stat = os.lstat(target.parent)
    building = target.parent / f".{target.name}.building-{os.getpid()}-{uuid.uuid4().hex}"
    building.mkdir()
    building_stat = os.lstat(building)
    episodes: list[dict[str, Any]] = []
    try:
        total_selected = 0
        selections: list[tuple[int, tuple[str, ...]]] = []
        for level in selected_levels:
            episode_ids = corpus.episode_ids(level=level)
            if episode_range is not None:
                start, stop = episode_range
                if stop > len(episode_ids):
                    raise ValueError(f"episode_range exceeds the L{level} episode inventory")
                episode_ids = episode_ids[start:stop]
            selections.append((level, episode_ids))
            total_selected += len(episode_ids)
        if total_selected == 0:
            raise ValueError("vision cache selection contains no episodes")

        completed = 0
        for level, episode_ids in selections:
            for episode_id in episode_ids:
                if _SAFE_COMPONENT.fullmatch(episode_id) is None:
                    raise ValueError("source episode_id is not a safe cache path component")
                source_episode = corpus.read_episode(episode_id)
                metadata_sha = _metadata_sha256(source_episode.metadata)
                decoded = _decode_source_episode(source_episode)
                if decoded.level != level or decoded.episode_id != episode_id:
                    raise ValueError("source episode selection identity is inconsistent")
                relative_root = Path(f"L{level}") / episode_id
                cache_manifest = write_episode_vision_feature_cache(
                    episode=decoded,
                    source_corpus_manifest_sha256=source_manifest_sha,
                    source_episode_metadata_sha256=metadata_sha,
                    encoder=encoder,
                    output_dir=building / relative_root,
                    boundary_batch_size=boundary_batch_size,
                )
                episodes.append(
                    {
                        "episode_id": decoded.episode_id,
                        "task_instance_id": decoded.task_instance_id,
                        "logical_master_task_index": decoded.logical_master_task_index,
                        "level": decoded.level,
                        "accepted_slot": decoded.accepted_slot,
                        "realization_draw_index": decoded.realization_draw_index,
                        "boundary_count": decoded.boundary_count,
                        "source_episode_metadata_sha256": metadata_sha,
                        "cache_manifest": cache_manifest.relative_to(building).as_posix(),
                    }
                )
                completed += 1
                if progress_fn is not None:
                    progress_fn(
                        f"vision-cache episode {completed}/{total_selected}: "
                        f"L{level} {episode_id} ({decoded.boundary_count} boundaries)"
                    )

        artifacts = {
            path.relative_to(building).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": _hash_file(path),
            }
            for path in sorted(building.rglob("*"))
            if path.is_file()
        }
        feature_paths = tuple(
            relative for relative in artifacts if relative.endswith("/features.npy")
        )
        feature_artifact_bytes = sum(artifacts[path]["bytes"] for path in feature_paths)
        feature_payload_bytes = sum(
            json.loads((building / row["cache_manifest"]).read_text(encoding="utf-8"))[
                "feature_payload_bytes"
            ]
            for row in episodes
        )
        spec = encoder.spec
        manifest = {
            "schema_version": 2,
            "format_id": _RUN_FORMAT_ID,
            "eligible": not provenance.dirty,
            "blockers": [] if not provenance.dirty else ["vision_cache_implementation_dirty"],
            "implementation": asdict(provenance),
            "source_corpus_id": corpus.manifest.corpus_id,
            "source_corpus_manifest_sha256": source_manifest_sha,
            "encoder_id": spec.encoder_id,
            "model_id": spec.model_id,
            "model_revision": spec.revision,
            "weights_sha256": spec.weights_sha256,
            "encoder_fingerprint": spec.fingerprint,
            "runtime": asdict(encoder.runtime_info),
            "levels": list(selected_levels),
            "episode_range": None if episode_range is None else list(episode_range),
            "boundary_batch_size": boundary_batch_size,
            "maximum_image_batch_size": boundary_batch_size * len(CAMERA_ORDER),
            "episode_count": len(episodes),
            "boundary_count": sum(row["boundary_count"] for row in episodes),
            "feature_payload_bytes": feature_payload_bytes,
            "feature_artifact_bytes": feature_artifact_bytes,
            "episodes": episodes,
            "artifacts": artifacts,
        }
        _write_file_fsynced(building / "manifest.json", _json_bytes(manifest))
        _fsync_directory(building)
        _rename_noreplace(building, target)
        _fsync_directory(target.parent)
    except BaseException:
        _cleanup_owned_staging(
            building,
            expected_device=building_stat.st_dev,
            expected_inode=building_stat.st_ino,
            parent=target.parent,
            expected_parent_device=parent_stat.st_dev,
            expected_parent_inode=parent_stat.st_ino,
        )
        raise
    return target / "manifest.json"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError("vision feature cache run manifest must be a mapping")
    return value


def load_verified_vision_feature_cache_run(
    cache_dir: Path,
    *,
    expected_source_manifest_sha256: str | None = None,
    expected_spec: VisionEncoderSpec | None = None,
    verify_payloads: bool = True,
) -> dict[str, Any]:
    """Strictly verify a v2 cache run and every episode cache it admits."""

    root = Path(cache_dir).resolve()
    value = _load_json(root / "manifest.json")
    if (
        set(value) != _RUN_FIELDS
        or value["schema_version"] != 2
        or value["format_id"] != _RUN_FORMAT_ID
    ):
        raise ValueError("vision feature cache run manifest fields are invalid")
    source_sha = value["source_corpus_manifest_sha256"]
    if type(source_sha) is not str or _SHA256.fullmatch(source_sha) is None:
        raise ValueError("vision feature cache run source identity is invalid")
    if (
        expected_source_manifest_sha256 is not None
        and source_sha != expected_source_manifest_sha256
    ):
        raise ValueError("vision feature cache run source manifest mismatch")
    if expected_spec is not None and value["encoder_fingerprint"] != expected_spec.fingerprint:
        raise ValueError("vision feature cache run encoder fingerprint mismatch")
    implementation = value["implementation"]
    if (
        type(implementation) is not dict
        or set(implementation) != {"revision", "source_sha256", "dirty"}
        or type(implementation["revision"]) is not str
        or type(implementation["source_sha256"]) is not str
        or _SHA256.fullmatch(implementation["source_sha256"]) is None
        or type(implementation["dirty"]) is not bool
        or value["eligible"] is not (not implementation["dirty"])
        or value["blockers"]
        != ([] if not implementation["dirty"] else ["vision_cache_implementation_dirty"])
    ):
        raise ValueError("vision feature cache run provenance is invalid")
    levels = value["levels"]
    if (
        type(levels) is not list
        or not levels
        or levels != sorted(set(levels))
        or any(type(level) is not int or level not in (1, 2, 3) for level in levels)
    ):
        raise ValueError("vision feature cache run levels are invalid")
    episode_range = value["episode_range"]
    if episode_range is not None and (
        type(episode_range) is not list
        or len(episode_range) != 2
        or any(type(item) is not int for item in episode_range)
        or episode_range[0] < 0
        or episode_range[1] <= episode_range[0]
    ):
        raise ValueError("vision feature cache run episode range is invalid")
    for name in (
        "boundary_batch_size",
        "maximum_image_batch_size",
        "episode_count",
        "boundary_count",
        "feature_payload_bytes",
        "feature_artifact_bytes",
    ):
        if type(value[name]) is not int or value[name] <= 0:
            raise ValueError(f"vision feature cache run {name} is invalid")
    if value["maximum_image_batch_size"] != value["boundary_batch_size"] * len(CAMERA_ORDER):
        raise ValueError("vision feature cache run image batch size is inconsistent")

    artifacts = value["artifacts"]
    if type(artifacts) is not dict or not artifacts:
        raise ValueError("vision feature cache run artifact inventory is invalid")
    for relative, metadata in artifacts.items():
        relative = _safe_relative(relative)
        path = root / relative
        if (
            type(metadata) is not dict
            or set(metadata) != {"bytes", "sha256"}
            or type(metadata["bytes"]) is not int
            or metadata["bytes"] < 0
            or type(metadata["sha256"]) is not str
            or _SHA256.fullmatch(metadata["sha256"]) is None
            or not path.is_file()
            or path.stat().st_size != metadata["bytes"]
            or (
                (verify_payloads or path.suffix != ".npy")
                and _hash_file(path) != metadata["sha256"]
            )
        ):
            raise ValueError(f"vision feature cache run artifact verification failed: {relative}")

    episodes = value["episodes"]
    if type(episodes) is not list or len(episodes) != value["episode_count"]:
        raise ValueError("vision feature cache run episode inventory is invalid")
    expected_artifacts: set[str] = set()
    seen_ids: set[str] = set()
    boundary_count = 0
    payload_bytes = 0
    artifact_bytes = 0
    for row in episodes:
        if type(row) is not dict or set(row) != _EPISODE_FIELDS:
            raise ValueError("vision feature cache run episode row is invalid")
        episode_id = row["episode_id"]
        level = row["level"]
        if (
            type(episode_id) is not str
            or _SAFE_COMPONENT.fullmatch(episode_id) is None
            or episode_id in seen_ids
            or type(level) is not int
            or level not in levels
            or type(row["boundary_count"]) is not int
            or row["boundary_count"] <= 0
            or type(row["source_episode_metadata_sha256"]) is not str
            or _SHA256.fullmatch(row["source_episode_metadata_sha256"]) is None
        ):
            raise ValueError("vision feature cache run episode identity is invalid")
        seen_ids.add(episode_id)
        cache_manifest = _safe_relative(row["cache_manifest"])
        expected_manifest = f"L{level}/{episode_id}/manifest.json"
        if cache_manifest != expected_manifest:
            raise ValueError("vision feature cache run episode path is invalid")
        expected_artifacts.update(
            {cache_manifest, cache_manifest.replace("manifest.json", "features.npy")}
        )
        cache = load_episode_vision_feature_cache(
            root / f"L{level}" / episode_id,
            expected_spec=expected_spec,
            verify_payloads=verify_payloads,
        )
        child = cache.manifest
        for name in (
            "episode_id",
            "task_instance_id",
            "logical_master_task_index",
            "level",
            "accepted_slot",
            "realization_draw_index",
            "boundary_count",
            "source_episode_metadata_sha256",
        ):
            if child[name] != row[name]:
                raise ValueError("vision feature cache run episode identity disagrees with child")
        if (
            child["source_corpus_manifest_sha256"] != source_sha
            or child["encoder_fingerprint"] != value["encoder_fingerprint"]
        ):
            raise ValueError("vision feature cache run child provenance is inconsistent")
        boundary_count += child["boundary_count"]
        payload_bytes += child["feature_payload_bytes"]
        artifact_bytes += child["feature_artifact_bytes"]
    if set(artifacts) != expected_artifacts:
        raise ValueError("vision feature cache run artifact inventory is incomplete")
    if (
        boundary_count != value["boundary_count"]
        or payload_bytes != value["feature_payload_bytes"]
        or artifact_bytes != value["feature_artifact_bytes"]
    ):
        raise ValueError("vision feature cache run aggregate counts are inconsistent")
    return value
