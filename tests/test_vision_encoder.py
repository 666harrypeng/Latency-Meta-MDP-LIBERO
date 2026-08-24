from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


def test_dino_specs_share_the_full_spatial_output_contract() -> None:
    from latency_meta_mdp.vision_encoder import load_vision_encoder_spec

    v2 = load_vision_encoder_spec(
        Path("configs/vision/dinov2_vits14_lvd142m_196_v1.yaml")
    )
    v3 = load_vision_encoder_spec(
        Path("configs/vision/dinov3_vits16_lvd1689m_224_v1.yaml")
    )

    assert v2.family == "dinov2"
    assert v2.input_size_px == (196, 196)
    assert v2.patch_size_px == 14
    assert v2.special_token_count == 1
    assert v2.output_shape == (196, 384)
    assert v2.access_mode == "public"
    assert (
        v2.weights_sha256
        == "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"
    )

    assert v3.family == "dinov3"
    assert v3.input_size_px == (224, 224)
    assert v3.patch_size_px == 16
    assert v3.special_token_count == 5
    assert v3.output_shape == (196, 384)
    assert v3.access_mode == "gated_manual"
    assert (
        v3.weights_sha256
        == "4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d"
    )

    assert v2.frozen is True
    assert v3.frozen is True
    assert v2.output_dtype == "float16"
    assert v3.output_dtype == "float16"
    assert v2.fingerprint != v3.fingerprint


def test_encoder_fingerprint_binds_revision_and_preprocessing() -> None:
    from latency_meta_mdp.vision_encoder import load_vision_encoder_spec

    spec = load_vision_encoder_spec(
        Path("configs/vision/dinov2_vits14_lvd142m_196_v1.yaml")
    )

    assert spec.fingerprint == replace(spec).fingerprint
    assert spec.fingerprint != replace(spec, revision="0" * 40).fingerprint
    assert spec.fingerprint != replace(
        spec,
        preprocessing=replace(spec.preprocessing, resize_mode="bilinear"),
    ).fingerprint


def test_spatial_token_selection_excludes_only_declared_special_tokens() -> None:
    from latency_meta_mdp.vision_encoder import (
        load_vision_encoder_spec,
        select_spatial_tokens,
    )

    spec = load_vision_encoder_spec(
        Path("configs/vision/dinov3_vits16_lvd1689m_224_v1.yaml")
    )
    hidden = np.full((2, 201, 384), -17.0, dtype=np.float32)
    expected = np.arange(2 * 196 * 384, dtype=np.float32).reshape(2, 196, 384)
    hidden[:, 5:, :] = expected

    selected = select_spatial_tokens(hidden, spec=spec)

    assert selected.shape == (2, 196, 384)
    np.testing.assert_array_equal(selected, expected)


def test_spatial_token_selection_rejects_wrong_model_layout() -> None:
    from latency_meta_mdp.vision_encoder import (
        load_vision_encoder_spec,
        select_spatial_tokens,
    )

    spec = load_vision_encoder_spec(
        Path("configs/vision/dinov2_vits14_lvd142m_196_v1.yaml")
    )

    with pytest.raises(ValueError, match="token layout"):
        select_spatial_tokens(
            np.zeros((1, 196, 384), dtype=np.float32),
            spec=spec,
        )


def test_encoder_spec_rejects_grid_that_disagrees_with_input_size() -> None:
    from latency_meta_mdp.vision_encoder import (
        VisionEncoderSpec,
        VisionPreprocessingSpec,
    )

    with pytest.raises(ValueError, match="patch grid"):
        VisionEncoderSpec(
            schema_version=1,
            encoder_id="broken",
            family="dinov2",
            model_id="example/broken",
            revision="1" * 40,
            weights_sha256="2" * 64,
            input_size_px=(196, 196),
            patch_size_px=14,
            patch_grid=(13, 14),
            patch_token_count=182,
            feature_dim=384,
            special_token_count=1,
            frozen=True,
            output_dtype="float16",
            access_mode="public",
            license_id="apache-2.0",
            preprocessing=VisionPreprocessingSpec(
                resize_mode="bicubic",
                rescale_factor=1.0 / 255.0,
                image_mean=(0.485, 0.456, 0.406),
                image_std=(0.229, 0.224, 0.225),
            ),
        )
