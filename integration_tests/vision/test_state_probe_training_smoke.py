from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from latency_meta_mdp.data.vision.contracts import load_vision_encoder_spec
from latency_meta_mdp.legacy.vision_probe_config import load_vision_probe_config
from latency_meta_mdp.legacy.vision_probe_corpus import load_level_probe_corpus


def test_one_epoch_l1_probe_smoke_writes_level_specific_artifacts(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("state probe training smoke requires CUDA")
    source = Path("outputs/bulk/expert/panda-ball-first-tranche-e58eee5/manifest.json")
    cache = Path(
        "outputs/derived/vision_features/dinov3-vits16-first-tranche-54b9562/manifest.json"
    )
    if not source.is_file() or not cache.is_file():
        pytest.skip("state probe training smoke requires the local first tranche")
    from latency_meta_mdp.legacy.vision_probe_training import train_level_state_probe

    spec = load_vision_encoder_spec(
        Path("configs/models/vision/dinov3_vits16_lvd1689m_224_v1.yaml")
    )
    config = replace(
        load_vision_probe_config(
            Path("configs/legacy/analysis/dinov3_temporal_state_probe_v1.yaml")
        ),
        max_epochs=1,
        early_stopping_patience=1,
    )
    corpus = load_level_probe_corpus(
        source_bulk_manifest=source,
        cache_run_manifest=cache,
        expected_spec=spec,
        level=1,
        history_sample_count=config.history_sample_count,
    )

    manifest_path = train_level_state_probe(
        corpus=corpus,
        config=config,
        output_dir=tmp_path / "L1",
        device="cuda",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = json.loads((manifest_path.parent / "metrics.json").read_text(encoding="utf-8"))

    assert manifest["level"] == 1
    assert manifest["checkpoint_scope"] == "single_level_only"
    assert manifest["epochs_completed"] == 1
    assert manifest["artifacts"].keys() == {
        "model.safetensors",
        "normalization.npz",
        "metrics.json",
        "training_history.json",
    }
    assert metrics["validation"]["probe"]["sample_count"] > 200
    assert metrics["holdout"]["probe"]["sample_count"] > 300
    assert (manifest_path.parent / "model.safetensors").is_file()
