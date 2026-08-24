"""Model-neutral contract for frozen spatial vision encoders."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

SPATIAL_TOKEN_COUNT = 196
SPATIAL_FEATURE_DIM = 384

_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_SUPPORTED_FAMILIES = frozenset({"dinov2", "dinov3"})
_SUPPORTED_ACCESS_MODES = frozenset({"public", "gated_manual"})
_SUPPORTED_RESIZE_MODES = frozenset({"bicubic", "bilinear"})


def _validate_triplet(name: str, value: tuple[float, ...]) -> None:
    if len(value) != 3 or not all(isinstance(item, (int, float)) for item in value):
        raise ValueError(f"{name} must contain three numeric values")


@dataclass(frozen=True)
class VisionPreprocessingSpec:
    resize_mode: str
    rescale_factor: float
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]

    def __post_init__(self) -> None:
        if self.resize_mode not in _SUPPORTED_RESIZE_MODES:
            raise ValueError("unsupported vision resize mode")
        if not isinstance(self.rescale_factor, (int, float)) or not (
            0.0 < self.rescale_factor <= 1.0
        ):
            raise ValueError("vision rescale factor must lie in (0, 1]")
        _validate_triplet("image_mean", self.image_mean)
        _validate_triplet("image_std", self.image_std)
        if any(value <= 0.0 for value in self.image_std):
            raise ValueError("image_std entries must be positive")


@dataclass(frozen=True)
class VisionEncoderSpec:
    schema_version: int
    encoder_id: str
    family: str
    model_id: str
    revision: str
    weights_sha256: str
    input_size_px: tuple[int, int]
    patch_size_px: int
    patch_grid: tuple[int, int]
    patch_token_count: int
    feature_dim: int
    special_token_count: int
    frozen: bool
    output_dtype: str
    access_mode: str
    license_id: str
    preprocessing: VisionPreprocessingSpec

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported vision encoder schema")
        if not self.encoder_id or not self.model_id or not self.license_id:
            raise ValueError("vision encoder identity fields cannot be empty")
        if self.family not in _SUPPORTED_FAMILIES:
            raise ValueError("unsupported vision encoder family")
        if _REVISION_PATTERN.fullmatch(self.revision) is None:
            raise ValueError("vision encoder revision must be a full lowercase Git SHA")
        if _SHA256_PATTERN.fullmatch(self.weights_sha256) is None:
            raise ValueError("vision encoder weights must have a full lowercase SHA256")
        if (
            len(self.input_size_px) != 2
            or len(self.patch_grid) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in (*self.input_size_px, self.patch_size_px, *self.patch_grid)
            )
        ):
            raise ValueError("vision encoder image and patch sizes must be positive integers")
        expected_grid = tuple(
            size // self.patch_size_px for size in self.input_size_px
        )
        if any(size % self.patch_size_px for size in self.input_size_px):
            raise ValueError("vision input size must be divisible by the patch size")
        if self.patch_grid != expected_grid:
            raise ValueError("vision patch grid disagrees with input and patch sizes")
        if self.patch_token_count != self.patch_grid[0] * self.patch_grid[1]:
            raise ValueError("vision patch token count disagrees with patch grid")
        if (
            self.patch_token_count != SPATIAL_TOKEN_COUNT
            or self.feature_dim != SPATIAL_FEATURE_DIM
        ):
            raise ValueError("vision encoder does not satisfy the 196 x 384 output contract")
        if (
            isinstance(self.special_token_count, bool)
            or not isinstance(self.special_token_count, int)
            or self.special_token_count <= 0
        ):
            raise ValueError("vision special-token count must be a positive integer")
        if self.frozen is not True:
            raise ValueError("belief vision encoder must be frozen")
        if self.output_dtype != "float16":
            raise ValueError("belief vision cache dtype must be float16")
        if self.access_mode not in _SUPPORTED_ACCESS_MODES:
            raise ValueError("unsupported vision model access mode")

    @property
    def output_shape(self) -> tuple[int, int]:
        return (self.patch_token_count, self.feature_dim)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _load_mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("vision encoder config must be a mapping")
    return value


def load_vision_encoder_spec(path: Path) -> VisionEncoderSpec:
    raw = _load_mapping(path)
    expected = set(VisionEncoderSpec.__dataclass_fields__)
    if set(raw) != expected:
        raise ValueError("vision encoder config fields are invalid")
    preprocessing = raw["preprocessing"]
    if not isinstance(preprocessing, dict) or set(preprocessing) != set(
        VisionPreprocessingSpec.__dataclass_fields__
    ):
        raise ValueError("vision preprocessing config fields are invalid")
    try:
        preprocessing_spec = VisionPreprocessingSpec(
            resize_mode=preprocessing["resize_mode"],
            rescale_factor=preprocessing["rescale_factor"],
            image_mean=tuple(preprocessing["image_mean"]),
            image_std=tuple(preprocessing["image_std"]),
        )
        return VisionEncoderSpec(
            **{
                **raw,
                "input_size_px": tuple(raw["input_size_px"]),
                "patch_grid": tuple(raw["patch_grid"]),
                "preprocessing": preprocessing_spec,
            }
        )
    except (KeyError, TypeError) as error:
        raise ValueError("vision encoder config values are invalid") from error


def select_spatial_tokens(hidden_state: Any, *, spec: VisionEncoderSpec) -> Any:
    shape = tuple(int(value) for value in hidden_state.shape)
    expected = (
        shape[0] if len(shape) == 3 else None,
        spec.special_token_count + spec.patch_token_count,
        spec.feature_dim,
    )
    if len(shape) != 3 or shape != expected:
        raise ValueError(
            "model hidden-state token layout does not match the vision encoder contract"
        )
    return hidden_state[:, spec.special_token_count :, :]
