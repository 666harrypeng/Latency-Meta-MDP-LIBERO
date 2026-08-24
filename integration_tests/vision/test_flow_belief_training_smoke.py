from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from latency_meta_mdp.vision_encoder import load_vision_encoder_spec


def _corpus_and_smoke_config():
    from latency_meta_mdp.belief.common.feature_corpus import (
        load_level_feature_belief_corpus,
    )
    from latency_meta_mdp.belief.flow.config import load_flow_belief_config

    source = Path(
        "outputs/bulk/expert/panda-ball-first-tranche-e58eee5/manifest.json"
    )
    cache = Path(
        "outputs/derived/vision_features/"
        "dinov3-vits16-first-tranche-54b9562/manifest.json"
    )
    if not source.is_file() or not cache.is_file():
        pytest.skip("Flow Belief training smoke requires the local first tranche")
    corpus = load_level_feature_belief_corpus(
        project_root=Path.cwd(),
        source_bulk_manifest=source,
        cache_run_manifest=cache,
        expected_spec=load_vision_encoder_spec(
            Path("configs/vision/dinov3_vits16_lvd1689m_224_v1.yaml")
        ),
        temporal_config_path=Path("configs/temporal/h50_e25_d20_k6_v1.yaml"),
        latency_law_path=Path("configs/latency/truncated_beta_5_26_400ms_v1.yaml"),
        level=1,
    )
    config = replace(
        load_flow_belief_config(
            Path("configs/belief/dinov3_flow_belief_v1.yaml")
        ),
        max_epochs=1,
        early_stopping_patience=1,
    )
    return corpus, config


def test_one_epoch_flow_belief_writes_independent_encoder_and_model(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Flow Belief training smoke requires CUDA")
    from latency_meta_mdp.belief.flow.training import train_level_flow_belief

    corpus, config = _corpus_and_smoke_config()

    manifest_path = train_level_flow_belief(
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
    assert manifest["initialization"] == "random_flow_encoder_and_vector_field"
    assert set(manifest["artifacts"]) == {
        "model.safetensors",
        "encoder.safetensors",
        "normalization.npz",
        "metrics.json",
        "training_history.json",
    }
    assert metrics["validation_fixed_flow_mse"] >= 0.0
    assert metrics["holdout_fixed_flow_mse"] >= 0.0
