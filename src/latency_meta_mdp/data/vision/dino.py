"""Lazy Hugging Face backend for frozen DINO spatial patch extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from latency_meta_mdp.data.vision.contracts import (
    VisionEncoderSpec,
    select_spatial_tokens,
)
from latency_meta_mdp.io.artifacts import sha256_file


def prepare_rgb_batch(images: np.ndarray, *, spec: VisionEncoderSpec) -> np.ndarray:
    """Resize and normalize a batch without importing the model runtime."""

    source = np.asarray(images)
    if source.ndim != 4 or source.shape[-1] != 3 or source.dtype != np.uint8:
        raise ValueError("vision input must be a batch of uint8 RGB images")
    resampling = {
        "bicubic": Image.Resampling.BICUBIC,
        "bilinear": Image.Resampling.BILINEAR,
    }[spec.preprocessing.resize_mode]
    height, width = spec.input_size_px
    resized = np.stack(
        [
            np.asarray(
                Image.fromarray(image, mode="RGB").resize(
                    (width, height),
                    resample=resampling,
                ),
                dtype=np.float32,
            )
            for image in source
        ],
        axis=0,
    )
    channel_first = resized.transpose(0, 3, 1, 2)
    channel_first *= np.float32(spec.preprocessing.rescale_factor)
    mean = np.asarray(spec.preprocessing.image_mean, dtype=np.float32)[None, :, None, None]
    std = np.asarray(spec.preprocessing.image_std, dtype=np.float32)[None, :, None, None]
    return np.asarray((channel_first - mean) / std, dtype=np.float32)


def freeze_for_inference(model: Any) -> Any:
    """Put a model in evaluation mode and structurally disable parameter gradients."""

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def verify_model_snapshot(snapshot_dir: Path, *, spec: VisionEncoderSpec) -> Path:
    weight_path = snapshot_dir / "model.safetensors"
    if not weight_path.is_file():
        raise ValueError("DINO snapshot is missing model.safetensors")
    if sha256_file(weight_path) != spec.weights_sha256:
        raise ValueError("DINO model.safetensors SHA256 disagrees with the encoder spec")
    return weight_path


@dataclass(frozen=True)
class HfDinoRuntimeInfo:
    torch_version: str
    transformers_version: str
    device: str
    compute_dtype: str


class HfDinoPatchEncoder:
    """Frozen DINO adapter with a common `[B, 196, 384]` output contract."""

    def __init__(
        self,
        *,
        spec: VisionEncoderSpec,
        model: Any,
        torch_module: Any,
        transformers_version: str,
        device: str,
        compute_dtype: Any,
    ) -> None:
        self.spec = spec
        self._torch = torch_module
        self._device = device
        self._compute_dtype = compute_dtype
        self._model = freeze_for_inference(model)
        self.runtime_info = HfDinoRuntimeInfo(
            torch_version=str(torch_module.__version__),
            transformers_version=transformers_version,
            device=device,
            compute_dtype=str(compute_dtype).removeprefix("torch."),
        )

    @classmethod
    def from_pretrained(
        cls,
        *,
        spec: VisionEncoderSpec,
        device: str = "cuda",
        local_files_only: bool = False,
    ) -> HfDinoPatchEncoder:
        try:
            import torch
            import transformers
            from huggingface_hub import snapshot_download
            from transformers import AutoModel
        except ImportError as error:
            raise RuntimeError(
                "Hugging Face DINO extraction requires the isolated belief-vision environment"
            ) from error
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for DINO extraction but is unavailable")
        compute_dtype = torch.float16 if device.startswith("cuda") else torch.float32
        snapshot = Path(
            snapshot_download(
                repo_id=spec.model_id,
                revision=spec.revision,
                allow_patterns=("config.json", "model.safetensors"),
                local_files_only=local_files_only,
            )
        )
        verify_model_snapshot(snapshot, spec=spec)
        model = AutoModel.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
            dtype=compute_dtype,
        )
        model.to(device=device, dtype=compute_dtype)
        return cls(
            spec=spec,
            model=model,
            torch_module=torch,
            transformers_version=str(transformers.__version__),
            device=device,
            compute_dtype=compute_dtype,
        )

    def encode(self, images: np.ndarray) -> Any:
        pixels = prepare_rgb_batch(images, spec=self.spec)
        tensor = self._torch.from_numpy(pixels).to(
            device=self._device,
            dtype=self._compute_dtype,
        )
        with self._torch.inference_mode():
            output = self._model(pixel_values=tensor)
            spatial = select_spatial_tokens(
                output.last_hidden_state,
                spec=self.spec,
            )
        if not bool(self._torch.isfinite(spatial).all().item()):
            raise ValueError("DINO encoder produced non-finite spatial tokens")
        if bool(spatial.requires_grad):
            raise RuntimeError("frozen DINO output unexpectedly requires gradients")
        return spatial.to(dtype=self._torch.float16)

    def encode_numpy(self, images: np.ndarray) -> np.ndarray:
        encoded = self.encode(images)
        return encoded.detach().cpu().numpy()
