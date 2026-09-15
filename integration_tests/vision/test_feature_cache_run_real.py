from __future__ import annotations

import json
from pathlib import Path

import pytest

from latency_meta_mdp.data.vision.cache import load_episode_vision_feature_cache
from latency_meta_mdp.data.vision.contracts import load_vision_encoder_spec
from latency_meta_mdp.data.vision.dino import HfDinoPatchEncoder


def test_dinov3_cache_run_selects_one_seed_from_each_level(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("real DINOv3 cache integration requires CUDA")
    source = Path("outputs/bulk/expert/panda-ball-first-tranche-e58eee5/manifest.json")
    if not source.is_file():
        pytest.skip("real DINOv3 cache integration requires the local first tranche")
    from latency_meta_mdp.data.vision.extract import (
        write_vision_feature_cache_run,
    )

    spec = load_vision_encoder_spec(
        Path("configs/models/vision/dinov3_vits16_lvd1689m_224_v1.yaml")
    )
    encoder = HfDinoPatchEncoder.from_pretrained(
        spec=spec,
        device="cuda",
        local_files_only=True,
    )

    manifest_path = write_vision_feature_cache_run(
        project_root=Path.cwd(),
        source_bulk_manifest=source,
        encoder=encoder,
        output_dir=tmp_path / "run",
        levels=(1, 2, 3),
        seed_start=1000,
        seed_count=1,
        boundary_batch_size=6,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["episode_count"] == 3
    assert manifest["levels"] == [1, 2, 3]
    assert manifest["seed_start"] == 1000
    assert manifest["seed_count"] == 1
    assert manifest["encoder_fingerprint"] == spec.fingerprint
    assert manifest["boundary_batch_size"] == 6
    assert manifest["maximum_image_batch_size"] == 12
    assert len(manifest["implementation_revision"]) == 40
    assert len(manifest["implementation_source_sha256"]) == 64
    assert isinstance(manifest["implementation_dirty"], bool)
    assert manifest["eligible"] is (not manifest["implementation_dirty"])
    assert {(item["level"], item["scene_seed"]) for item in manifest["episodes"]} == {
        (1, 1000),
        (2, 1000),
        (3, 1000),
    }
    for item in manifest["episodes"]:
        cache = load_episode_vision_feature_cache(
            manifest_path.parent / Path(item["cache_manifest"]).parent,
            expected_spec=spec,
        )
        assert cache.features.shape[1:] == (2, 196, 384)


def test_dinov3_cache_run_accepts_formal_corpus_layout(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("real DINOv3 cache integration requires CUDA")
    source = Path("outputs/bulk/expert/panda-ball-formal-train-1000-1199-7571a4c/manifest.json")
    if not source.is_file():
        pytest.skip("formal cache integration requires the local formal corpus")
    from latency_meta_mdp.data.vision.extract import (
        write_vision_feature_cache_run,
    )

    spec = load_vision_encoder_spec(
        Path("configs/models/vision/dinov3_vits16_lvd1689m_224_v1.yaml")
    )
    encoder = HfDinoPatchEncoder.from_pretrained(
        spec=spec,
        device="cuda",
        local_files_only=True,
    )

    manifest_path = write_vision_feature_cache_run(
        project_root=Path.cwd(),
        source_bulk_manifest=source,
        encoder=encoder,
        output_dir=tmp_path / "formal-cache",
        levels=(1, 2, 3),
        seed_start=1000,
        seed_count=1,
        boundary_batch_size=6,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["episode_count"] == 3
    assert manifest["source_bulk_run_id"] == source.parent.name
