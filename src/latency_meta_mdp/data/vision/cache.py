"""Atomic, provenance-bound caches for frozen spatial vision features."""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from latency_meta_mdp.data.collection.artifacts import (
    _cleanup_owned_staging,
    _fsync_directory,
    _hash_file,
    _rename_noreplace,
    _write_file_fsynced,
)
from latency_meta_mdp.data.vision.contracts import VisionEncoderSpec

CAMERA_ORDER = ("agentview", "wrist")
_FORMAT_ID_V1 = "vision_feature_cache_v1"
_FORMAT_ID_V2 = "vision_feature_cache_v2"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_FIELDS_V1 = frozenset(
    {
        "schema_version",
        "format_id",
        "episode_id",
        "level",
        "scene_seed",
        "boundary_count",
        "camera_order",
        "feature_shape",
        "dtype",
        "real_boundaries_only",
        "source_episode_manifest_sha256",
        "encoder_id",
        "encoder_family",
        "model_id",
        "model_revision",
        "weights_sha256",
        "encoder_fingerprint",
        "preprocessing",
        "runtime",
        "boundary_batch_size",
        "maximum_image_batch_size",
        "artifacts",
    }
)
_MANIFEST_FIELDS_V2 = frozenset(
    {
        "schema_version",
        "format_id",
        "episode_id",
        "task_instance_id",
        "logical_master_task_index",
        "level",
        "accepted_slot",
        "realization_draw_index",
        "boundary_count",
        "camera_order",
        "feature_shape",
        "dtype",
        "real_boundaries_only",
        "source_corpus_manifest_sha256",
        "source_episode_metadata_sha256",
        "encoder_id",
        "encoder_family",
        "model_id",
        "model_revision",
        "weights_sha256",
        "encoder_fingerprint",
        "preprocessing",
        "runtime",
        "boundary_batch_size",
        "maximum_image_batch_size",
        "feature_payload_bytes",
        "feature_artifact_bytes",
        "artifacts",
    }
)


class VisionFeatureEncoder(Protocol):
    spec: VisionEncoderSpec
    runtime_info: Any

    def encode_numpy(self, images: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class EpisodeVisionFeatureCache:
    manifest: dict[str, Any]
    features: np.ndarray


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _require_sha256(value: Any, *, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validate_episode_images(episode: Any) -> tuple[np.ndarray, np.ndarray, int]:
    if (
        not isinstance(episode.episode_id, str)
        or not episode.episode_id
        or not isinstance(episode.task_instance_id, str)
        or not episode.task_instance_id
        or type(episode.logical_master_task_index) is not int
        or episode.logical_master_task_index < 0
        or type(episode.level) is not int
        or episode.level not in (1, 2, 3)
        or type(episode.accepted_slot) is not int
        or not 0 <= episode.accepted_slot < 4
        or type(episode.realization_draw_index) is not int
        or episode.realization_draw_index < 0
        or isinstance(episode.boundary_count, bool)
        or not isinstance(episode.boundary_count, int)
        or episode.boundary_count <= 0
    ):
        raise ValueError("vision cache episode identity is invalid")
    agent = np.asarray(episode.deployment.agentview_rgb)
    wrist = np.asarray(episode.deployment.wrist_rgb)
    expected_prefix = (episode.boundary_count,)
    if (
        agent.ndim != 4
        or wrist.ndim != 4
        or agent.shape[:1] != expected_prefix
        or wrist.shape[:1] != expected_prefix
        or agent.shape != wrist.shape
        or agent.shape[-1] != 3
        or agent.dtype != np.uint8
        or wrist.dtype != np.uint8
    ):
        raise ValueError("vision cache requires aligned uint8 RGB camera boundaries")
    return agent, wrist, episode.boundary_count


def _runtime_dict(encoder: VisionFeatureEncoder) -> dict[str, Any]:
    value = asdict(encoder.runtime_info)
    if not isinstance(value, dict) or not value:
        raise ValueError("vision encoder runtime metadata is invalid")
    return value


def write_episode_vision_feature_cache(
    *,
    episode: Any,
    source_corpus_manifest_sha256: str,
    source_episode_metadata_sha256: str,
    encoder: VisionFeatureEncoder,
    output_dir: Path,
    boundary_batch_size: int,
) -> Path:
    target = Path(output_dir).absolute()
    if target.exists():
        raise FileExistsError(f"vision feature cache already exists: {target}")
    if (
        isinstance(boundary_batch_size, bool)
        or not isinstance(boundary_batch_size, int)
        or boundary_batch_size <= 0
    ):
        raise ValueError("vision boundary batch size must be a positive integer")
    source_corpus_sha = _require_sha256(
        source_corpus_manifest_sha256,
        name="source_corpus_manifest_sha256",
    )
    source_episode_sha = _require_sha256(
        source_episode_metadata_sha256,
        name="source_episode_metadata_sha256",
    )
    agent, wrist, boundary_count = _validate_episode_images(episode)
    spec = encoder.spec
    feature_shape = (
        boundary_count,
        len(CAMERA_ORDER),
        spec.patch_token_count,
        spec.feature_dim,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    parent_stat = os.lstat(target.parent)
    building = target.parent / f".{target.name}.building-{os.getpid()}-{uuid.uuid4().hex}"
    building.mkdir()
    building_stat = os.lstat(building)
    try:
        feature_path = building / "features.npy"
        features = np.lib.format.open_memmap(
            feature_path,
            mode="w+",
            dtype=np.float16,
            shape=feature_shape,
        )
        for start in range(0, boundary_count, boundary_batch_size):
            stop = min(start + boundary_batch_size, boundary_count)
            paired = np.stack((agent[start:stop], wrist[start:stop]), axis=1)
            flat = paired.reshape((-1, *paired.shape[2:]))
            encoded = np.asarray(encoder.encode_numpy(flat))
            expected = (
                (stop - start) * len(CAMERA_ORDER),
                spec.patch_token_count,
                spec.feature_dim,
            )
            if (
                encoded.shape != expected
                or encoded.dtype != np.float16
                or not np.all(np.isfinite(encoded))
            ):
                raise ValueError("vision encoder output does not match the cache contract")
            features[start:stop] = encoded.reshape(
                stop - start,
                len(CAMERA_ORDER),
                spec.patch_token_count,
                spec.feature_dim,
            )
        features.flush()
        del features
        with feature_path.open("rb") as handle:
            os.fsync(handle.fileno())
        feature_artifact_bytes = feature_path.stat().st_size
        feature_payload_bytes = int(np.prod(feature_shape)) * np.dtype(np.float16).itemsize
        manifest = {
            "schema_version": 2,
            "format_id": _FORMAT_ID_V2,
            "episode_id": episode.episode_id,
            "task_instance_id": episode.task_instance_id,
            "logical_master_task_index": episode.logical_master_task_index,
            "level": episode.level,
            "accepted_slot": episode.accepted_slot,
            "realization_draw_index": episode.realization_draw_index,
            "boundary_count": boundary_count,
            "camera_order": list(CAMERA_ORDER),
            "feature_shape": list(feature_shape),
            "dtype": "float16",
            "real_boundaries_only": True,
            "source_corpus_manifest_sha256": source_corpus_sha,
            "source_episode_metadata_sha256": source_episode_sha,
            "encoder_id": spec.encoder_id,
            "encoder_family": spec.family,
            "model_id": spec.model_id,
            "model_revision": spec.revision,
            "weights_sha256": spec.weights_sha256,
            "encoder_fingerprint": spec.fingerprint,
            "preprocessing": asdict(spec.preprocessing),
            "runtime": _runtime_dict(encoder),
            "boundary_batch_size": boundary_batch_size,
            "maximum_image_batch_size": boundary_batch_size * len(CAMERA_ORDER),
            "feature_payload_bytes": feature_payload_bytes,
            "feature_artifact_bytes": feature_artifact_bytes,
            "artifacts": {
                "features.npy": {
                    "bytes": feature_artifact_bytes,
                    "sha256": _hash_file(feature_path),
                }
            },
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


def load_episode_vision_feature_cache(
    cache_dir: Path,
    *,
    expected_spec: VisionEncoderSpec | None = None,
    verify_payloads: bool = True,
) -> EpisodeVisionFeatureCache:
    root = cache_dir.resolve()
    manifest_path = root / "manifest.json"
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("vision feature cache manifest fields are invalid")
    format_id = value.get("format_id")
    schema_version = value.get("schema_version")
    if format_id == _FORMAT_ID_V1 and schema_version == 1:
        expected_fields = _MANIFEST_FIELDS_V1
    elif format_id == _FORMAT_ID_V2 and schema_version == 2:
        expected_fields = _MANIFEST_FIELDS_V2
    else:
        raise ValueError("unsupported vision feature cache format")
    if set(value) != expected_fields:
        raise ValueError("vision feature cache manifest fields are invalid")
    if expected_spec is not None and value["encoder_fingerprint"] != expected_spec.fingerprint:
        raise ValueError("vision feature cache encoder fingerprint mismatch")
    artifacts = value["artifacts"]
    feature_path = root / "features.npy"
    if not isinstance(artifacts, dict) or set(artifacts) != {"features.npy"}:
        raise ValueError("vision feature cache artifact verification failed")
    if schema_version == 1:
        artifact_valid = _hash_file(feature_path) == artifacts["features.npy"]
    else:
        metadata = artifacts["features.npy"]
        artifact_valid = (
            type(metadata) is dict
            and set(metadata) == {"bytes", "sha256"}
            and type(metadata["bytes"]) is int
            and metadata["bytes"] == feature_path.stat().st_size
            and (not verify_payloads or _hash_file(feature_path) == metadata["sha256"])
            and value["feature_artifact_bytes"] == feature_path.stat().st_size
            and value["feature_payload_bytes"]
            == int(np.prod(value["feature_shape"])) * np.dtype(np.float16).itemsize
            and type(value["source_corpus_manifest_sha256"]) is str
            and _SHA256.fullmatch(value["source_corpus_manifest_sha256"]) is not None
            and type(value["source_episode_metadata_sha256"]) is str
            and _SHA256.fullmatch(value["source_episode_metadata_sha256"]) is not None
        )
    if not artifact_valid:
        raise ValueError("vision feature cache artifact verification failed")
    features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    expected_shape = tuple(value["feature_shape"])
    if (
        features.shape != expected_shape
        or features.dtype != np.float16
        or expected_shape
        != (
            value["boundary_count"],
            len(CAMERA_ORDER),
            196,
            384,
        )
        or value["camera_order"] != list(CAMERA_ORDER)
        or value["real_boundaries_only"] is not True
    ):
        raise ValueError("vision feature cache array contract is invalid")
    return EpisodeVisionFeatureCache(manifest=value, features=features)
