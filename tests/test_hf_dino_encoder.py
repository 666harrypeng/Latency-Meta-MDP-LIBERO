from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


def _spec():
    from latency_meta_mdp.vision_encoder import load_vision_encoder_spec

    return load_vision_encoder_spec(
        Path("configs/vision/dinov2_vits14_lvd142m_196_v1.yaml")
    )


def test_rgb_preprocessing_is_chw_float32_and_uses_declared_normalization() -> None:
    from latency_meta_mdp.hf_dino_encoder import prepare_rgb_batch

    black = np.zeros((4, 4, 3), dtype=np.uint8)
    white = np.full((4, 4, 3), 255, dtype=np.uint8)
    images = np.stack((black, white), axis=0)

    pixels = prepare_rgb_batch(images, spec=_spec())

    assert pixels.shape == (2, 3, 196, 196)
    assert pixels.dtype == np.float32
    np.testing.assert_allclose(
        pixels[0, :, 0, 0],
        np.asarray([-2.1179039, -2.0357141, -1.8044444]),
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        pixels[1, :, -1, -1],
        np.asarray([2.2489083, 2.4285715, 2.64]),
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_array_equal(images[0], black)
    np.testing.assert_array_equal(images[1], white)


@pytest.mark.parametrize(
    "images",
    (
        np.zeros((2, 8, 8), dtype=np.uint8),
        np.zeros((2, 8, 8, 4), dtype=np.uint8),
        np.zeros((2, 8, 8, 3), dtype=np.float32),
    ),
)
def test_rgb_preprocessing_rejects_non_contract_images(images: np.ndarray) -> None:
    from latency_meta_mdp.hf_dino_encoder import prepare_rgb_batch

    with pytest.raises(ValueError, match="uint8 RGB"):
        prepare_rgb_batch(images, spec=_spec())


def test_freeze_for_inference_disables_gradients_and_training_mode() -> None:
    from latency_meta_mdp.hf_dino_encoder import freeze_for_inference

    class Parameter:
        def __init__(self) -> None:
            self.requires_grad = True

        def requires_grad_(self, enabled: bool):
            self.requires_grad = enabled
            return self

    class Model:
        def __init__(self) -> None:
            self.training = True
            self._parameters = [Parameter(), Parameter()]

        def eval(self):
            self.training = False
            return self

        def parameters(self):
            return iter(self._parameters)

    model = Model()

    result = freeze_for_inference(model)

    assert result is model
    assert model.training is False
    assert all(parameter.requires_grad is False for parameter in model._parameters)


def test_model_snapshot_verification_binds_the_actual_weight_bytes(
    tmp_path: Path,
) -> None:
    from latency_meta_mdp.hf_dino_encoder import verify_model_snapshot

    weight_bytes = b"pinned model weights"
    weight_path = tmp_path / "model.safetensors"
    weight_path.write_bytes(weight_bytes)
    spec = replace(
        _spec(),
        weights_sha256=hashlib.sha256(weight_bytes).hexdigest(),
    )

    assert verify_model_snapshot(tmp_path, spec=spec) == weight_path

    weight_path.write_bytes(b"different weights")
    with pytest.raises(ValueError, match="SHA256"):
        verify_model_snapshot(tmp_path, spec=spec)
