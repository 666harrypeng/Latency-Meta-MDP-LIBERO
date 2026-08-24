"""Atomic, provenance-bound caches for frozen spatial vision features."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from latency_meta_mdp.artifacts import sha256_file
from latency_meta_mdp.vision_encoder import VisionEncoderSpec

CAMERA_ORDER = ("agentview", "wrist")
_FORMAT_ID = "vision_feature_cache_v1"
_MANIFEST_FIELDS = frozenset(
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


class VisionFeatureEncoder(Protocol):
    spec: VisionEncoderSpec
    runtime_info: Any

    def encode_numpy(self, images: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class EpisodeVisionFeatureCache:
    manifest: dict[str, Any]
    features: np.ndarray


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _validate_episode_images(episode: Any) -> tuple[np.ndarray, np.ndarray, int]:
    if (
        not isinstance(episode.episode_id, str)
        or not episode.episode_id
        or episode.level not in (1, 2, 3)
        or isinstance(episode.scene_seed, bool)
        or not isinstance(episode.scene_seed, int)
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
    source_episode_manifest: Path,
    encoder: VisionFeatureEncoder,
    output_dir: Path,
    boundary_batch_size: int,
) -> Path:
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"vision feature cache already exists: {target}")
    if (
        isinstance(boundary_batch_size, bool)
        or not isinstance(boundary_batch_size, int)
        or boundary_batch_size <= 0
    ):
        raise ValueError("vision boundary batch size must be a positive integer")
    source_manifest = source_episode_manifest.resolve()
    if not source_manifest.is_file():
        raise FileNotFoundError(f"source episode manifest does not exist: {source_manifest}")
    agent, wrist, boundary_count = _validate_episode_images(episode)
    spec = encoder.spec
    feature_shape = (
        boundary_count,
        len(CAMERA_ORDER),
        spec.patch_token_count,
        spec.feature_dim,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f"{target.name}.building-{uuid.uuid4().hex}"
    building.mkdir()
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
        manifest = {
            "schema_version": 1,
            "format_id": _FORMAT_ID,
            "episode_id": episode.episode_id,
            "level": episode.level,
            "scene_seed": episode.scene_seed,
            "boundary_count": boundary_count,
            "camera_order": list(CAMERA_ORDER),
            "feature_shape": list(feature_shape),
            "dtype": "float16",
            "real_boundaries_only": True,
            "source_episode_manifest_sha256": sha256_file(source_manifest),
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
            "artifacts": {"features.npy": sha256_file(feature_path)},
        }
        _write_json(building / "manifest.json", manifest)
        if target.exists():
            raise FileExistsError(f"vision feature cache already exists: {target}")
        os.rename(building, target)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return target / "manifest.json"


def load_episode_vision_feature_cache(
    cache_dir: Path,
    *,
    expected_spec: VisionEncoderSpec | None = None,
) -> EpisodeVisionFeatureCache:
    root = cache_dir.resolve()
    manifest_path = root / "manifest.json"
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != _MANIFEST_FIELDS:
        raise ValueError("vision feature cache manifest fields are invalid")
    if value["schema_version"] != 1 or value["format_id"] != _FORMAT_ID:
        raise ValueError("unsupported vision feature cache format")
    if expected_spec is not None and value["encoder_fingerprint"] != expected_spec.fingerprint:
        raise ValueError("vision feature cache encoder fingerprint mismatch")
    artifacts = value["artifacts"]
    feature_path = root / "features.npy"
    if (
        not isinstance(artifacts, dict)
        or set(artifacts) != {"features.npy"}
        or sha256_file(feature_path) != artifacts["features.npy"]
    ):
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
