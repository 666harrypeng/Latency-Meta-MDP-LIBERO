"""Sanitized inference-only Hugging Face publication contract."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from latency_meta_mdp.data.collection.artifacts import (
    _fsync_directory,
    _hash_file,
    _rename_noreplace,
    _write_file_fsynced,
)

_INFERENCE_FIELDS = {
    "schema_version",
    "format_id",
    "level",
    "temporal_config_id",
    "model_stride_ticks",
    "history_observation_count",
    "native_rollout_steps",
    "vision_encoder_id",
    "camera_order",
    "proprio_dim",
    "action_dim",
}
_FORBIDDEN_TEXT = (
    "optimizer",
    "wandb",
    "fold",
    "seed",
    "source_manifest",
    "cache_manifest",
    "/home/",
    "/root/",
    "hf_token",
    "wandb_api_key",
)


@dataclass(frozen=True)
class HfPublicationConfig:
    schema_version: int
    config_id: str
    namespace: str
    repo_name_template: str
    repo_type: str
    visibility: str
    gating: str
    readme_policy: str
    allowed_files: tuple[str, ...]

    def __post_init__(self) -> None:
        expected_files = (
            "README.md",
            "checksums.json",
            "inference_config.json",
            "model.safetensors",
            "proprio_normalization.json",
        )
        if (
            self.schema_version != 1
            or self.config_id != "action_conditioned_jepa_hf_inference"
            or self.namespace != "yypeng666"
            or self.repo_name_template != "metamdp-jepa-return-l{level}-s4-h160ms-t400ms-final-v1"
            or self.repo_type != "model"
            or self.visibility != "public"
            or self.gating != "manual"
            or self.readme_policy != "empty"
            or self.allowed_files != expected_files
        ):
            raise ValueError("unsupported JEPA Hugging Face publication semantics")

    def repo_id(self, level: int) -> str:
        if type(level) is not int or level not in (1, 2, 3):
            raise ValueError("level must be 1, 2, or 3")
        return f"{self.namespace}/{self.repo_name_template.format(level=level)}"


@dataclass(frozen=True)
class HfInferenceBundle:
    repo_id: str
    root: Path


def load_hf_publication_config(path: Path) -> HfPublicationConfig:
    raw: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    expected = {field.name for field in fields(HfPublicationConfig)}
    if type(raw) is not dict or set(raw) != expected:
        raise ValueError("JEPA Hugging Face publication config fields are invalid")
    raw = dict(raw)
    raw["allowed_files"] = tuple(raw["allowed_files"])
    return HfPublicationConfig(**raw)


def build_hf_inference_bundle(
    *,
    output_dir: Path,
    publication_config: HfPublicationConfig,
    level: int,
    model_checkpoint: Path,
    proprio_normalization: Path,
    inference_config: dict[str, Any],
) -> HfInferenceBundle:
    if not isinstance(publication_config, HfPublicationConfig):
        raise TypeError("publication_config must be HfPublicationConfig")
    repo_id = publication_config.repo_id(level)
    if (
        type(inference_config) is not dict
        or set(inference_config) != _INFERENCE_FIELDS
        or inference_config["schema_version"] != 1
        or inference_config["format_id"] != "action_conditioned_jepa_inference_config"
        or inference_config["level"] != level
    ):
        raise ValueError("inference config fields are invalid")
    encoded_config = json.dumps(
        inference_config,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    lowered = encoded_config.lower()
    if any(value in lowered for value in _FORBIDDEN_TEXT):
        raise ValueError("inference config contains forbidden training or private metadata")
    model_path = Path(model_checkpoint)
    normalization_path = Path(proprio_normalization)
    if not model_path.is_file() or not normalization_path.is_file():
        raise FileNotFoundError("model checkpoint and normalization must exist")
    try:
        internal_normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("internal proprio normalization is invalid") from error
    internal_fields = {
        "schema_version",
        "format_id",
        "level",
        "mean",
        "scale",
        "constant_dimension_mask",
        "episode_ids",
        "boundary_count",
        "source_manifest_sha256",
        "split_manifest_sha256",
        "sample_index_sha256",
    }
    if (
        type(internal_normalization) is not dict
        or set(internal_normalization) != internal_fields
        or internal_normalization["schema_version"] != 1
        or internal_normalization["level"] != level
        or any(
            type(internal_normalization[name]) is not list
            or len(internal_normalization[name]) != 16
            for name in ("mean", "scale", "constant_dimension_mask")
        )
    ):
        raise ValueError("internal proprio normalization fields are invalid")
    public_normalization = {
        "schema_version": 1,
        "format_id": "action_conditioned_jepa_inference_proprio_normalization",
        "level": level,
        "mean": internal_normalization["mean"],
        "scale": internal_normalization["scale"],
        "constant_dimension_mask": internal_normalization["constant_dimension_mask"],
    }
    encoded_normalization = json.dumps(
        public_normalization,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    target = Path(output_dir).absolute()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f".{target.name}.building-{os.getpid()}-{uuid.uuid4().hex}"
    building.mkdir()
    try:
        _write_file_fsynced(building / "README.md", b"")
        _write_file_fsynced(building / "model.safetensors", model_path.read_bytes())
        _write_file_fsynced(
            building / "proprio_normalization.json",
            (encoded_normalization + "\n").encode(),
        )
        _write_file_fsynced(
            building / "inference_config.json",
            (encoded_config + "\n").encode(),
        )
        checksums = {
            name: _hash_file(building / name)
            for name in publication_config.allowed_files
            if name != "checksums.json"
        }
        _write_file_fsynced(
            building / "checksums.json",
            (json.dumps(checksums, indent=2, sort_keys=True) + "\n").encode(),
        )
        if tuple(sorted(path.name for path in building.iterdir())) != tuple(
            sorted(publication_config.allowed_files)
        ):
            raise ValueError("inference bundle contains an unexpected file")
        _fsync_directory(building)
        _rename_noreplace(building, target)
        _fsync_directory(target.parent)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return HfInferenceBundle(repo_id=repo_id, root=target)


def build_hf_publication_commands(
    *,
    config: HfPublicationConfig,
    level: int,
    bundle_dir: Path,
) -> tuple[tuple[str, ...], ...]:
    if not isinstance(config, HfPublicationConfig):
        raise TypeError("config must be HfPublicationConfig")
    repo_id = config.repo_id(level)
    bundle = str(Path(bundle_dir))
    return (
        (
            "hf",
            "repos",
            "create",
            repo_id,
            "--type",
            "model",
            "--public",
            "--exist-ok",
        ),
        (
            "hf",
            "repos",
            "settings",
            repo_id,
            "--gated",
            "manual",
            "--public",
            "--type",
            "model",
        ),
        (
            "hf",
            "upload",
            repo_id,
            bundle,
            "--type",
            "model",
            "--commit-message",
            "Publish inference bundle",
        ),
    )
