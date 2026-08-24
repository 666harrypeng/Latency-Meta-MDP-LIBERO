from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from latency_meta_mdp.artifacts import sha256_file
from latency_meta_mdp.vision_encoder import load_vision_encoder_spec


@dataclass(frozen=True)
class _RuntimeInfo:
    torch_version: str = "test-torch"
    transformers_version: str = "test-transformers"
    device: str = "test-device"
    compute_dtype: str = "float16"


class _DeterministicEncoder:
    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self.spec = load_vision_encoder_spec(
            Path("configs/vision/dinov3_vits16_lvd1689m_224_v1.yaml")
        )
        self.runtime_info = _RuntimeInfo()
        self.fail_on_call = fail_on_call
        self.call_count = 0

    def encode_numpy(self, images: np.ndarray) -> np.ndarray:
        self.call_count += 1
        if self.call_count == self.fail_on_call:
            raise RuntimeError("injected extraction failure")
        values = images[:, 0, 0, 0].astype(np.float16)
        return np.broadcast_to(
            values[:, None, None],
            (len(images), 196, 384),
        ).copy()


def _episode(boundary_count: int = 3):
    agent = np.stack(
        [np.full((4, 4, 3), 10 + tick, dtype=np.uint8) for tick in range(boundary_count)]
    )
    wrist = np.stack(
        [np.full((4, 4, 3), 100 + tick, dtype=np.uint8) for tick in range(boundary_count)]
    )
    return SimpleNamespace(
        episode_id="l2-seed-001000-attempt-00",
        level=2,
        scene_seed=1000,
        boundary_count=boundary_count,
        deployment=SimpleNamespace(
            agentview_rgb=agent,
            wrist_rgb=wrist,
        ),
    )


def test_episode_cache_preserves_boundary_and_camera_order(tmp_path: Path) -> None:
    from latency_meta_mdp.vision_feature_cache import (
        load_episode_vision_feature_cache,
        write_episode_vision_feature_cache,
    )

    source_manifest = tmp_path / "source_manifest.json"
    source_manifest.write_text('{"source": "episode"}\n', encoding="utf-8")
    output = tmp_path / "cache"
    encoder = _DeterministicEncoder()

    manifest_path = write_episode_vision_feature_cache(
        episode=_episode(),
        source_episode_manifest=source_manifest,
        encoder=encoder,
        output_dir=output,
        boundary_batch_size=2,
    )
    cache = load_episode_vision_feature_cache(
        output,
        expected_spec=encoder.spec,
    )

    assert manifest_path == output / "manifest.json"
    assert cache.features.shape == (3, 2, 196, 384)
    assert cache.features.dtype == np.float16
    np.testing.assert_array_equal(cache.features[:, 0, 0, 0], [10, 11, 12])
    np.testing.assert_array_equal(cache.features[:, 1, 0, 0], [100, 101, 102])
    assert cache.manifest["camera_order"] == ["agentview", "wrist"]
    assert cache.manifest["real_boundaries_only"] is True
    assert cache.manifest["source_episode_manifest_sha256"] == sha256_file(
        source_manifest
    )
    assert cache.manifest["encoder_fingerprint"] == encoder.spec.fingerprint
    assert cache.manifest["weights_sha256"] == encoder.spec.weights_sha256
    assert cache.manifest["boundary_batch_size"] == 2
    assert cache.manifest["maximum_image_batch_size"] == 4
    assert cache.manifest["artifacts"]["features.npy"] == sha256_file(
        output / "features.npy"
    )


def test_episode_cache_is_no_overwrite(tmp_path: Path) -> None:
    from latency_meta_mdp.vision_feature_cache import write_episode_vision_feature_cache

    source_manifest = tmp_path / "source_manifest.json"
    source_manifest.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "cache"
    encoder = _DeterministicEncoder()
    write_episode_vision_feature_cache(
        episode=_episode(),
        source_episode_manifest=source_manifest,
        encoder=encoder,
        output_dir=output,
        boundary_batch_size=2,
    )

    with pytest.raises(FileExistsError, match="already exists"):
        write_episode_vision_feature_cache(
            episode=_episode(),
            source_episode_manifest=source_manifest,
            encoder=encoder,
            output_dir=output,
            boundary_batch_size=2,
        )


def test_episode_cache_failure_leaves_no_partial_output(tmp_path: Path) -> None:
    from latency_meta_mdp.vision_feature_cache import write_episode_vision_feature_cache

    source_manifest = tmp_path / "source_manifest.json"
    source_manifest.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "cache"

    with pytest.raises(RuntimeError, match="injected extraction failure"):
        write_episode_vision_feature_cache(
            episode=_episode(boundary_count=4),
            source_episode_manifest=source_manifest,
            encoder=_DeterministicEncoder(fail_on_call=2),
            output_dir=output,
            boundary_batch_size=2,
        )

    assert not output.exists()
    assert list(tmp_path.glob("cache.building-*")) == []


def test_episode_cache_loader_rejects_wrong_encoder(tmp_path: Path) -> None:
    from latency_meta_mdp.vision_feature_cache import (
        load_episode_vision_feature_cache,
        write_episode_vision_feature_cache,
    )

    source_manifest = tmp_path / "source_manifest.json"
    source_manifest.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "cache"
    encoder = _DeterministicEncoder()
    write_episode_vision_feature_cache(
        episode=_episode(),
        source_episode_manifest=source_manifest,
        encoder=encoder,
        output_dir=output,
        boundary_batch_size=2,
    )
    wrong = load_vision_encoder_spec(
        Path("configs/vision/dinov2_vits14_lvd142m_196_v1.yaml")
    )

    with pytest.raises(ValueError, match="encoder fingerprint"):
        load_episode_vision_feature_cache(output, expected_spec=wrong)
