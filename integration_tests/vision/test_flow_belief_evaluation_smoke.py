from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from latency_meta_mdp.data.vision.contracts import load_vision_encoder_spec


def test_flow_evaluation_samples_checkpoint_and_writes_metrics(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Flow Belief evaluation smoke requires CUDA")
    source = Path("outputs/bulk/expert/panda-ball-first-tranche-e58eee5/manifest.json")
    cache = Path(
        "outputs/derived/vision_features/dinov3-vits16-first-tranche-54b9562/manifest.json"
    )
    if not source.is_file() or not cache.is_file():
        pytest.skip("Flow Belief evaluation smoke requires the local first tranche")
    from latency_meta_mdp.legacy.belief.common.feature_corpus import (
        load_level_feature_belief_corpus,
    )
    from latency_meta_mdp.legacy.belief.flow.config import load_flow_belief_config
    from latency_meta_mdp.legacy.belief.flow.evaluation import evaluate_level_flow_belief
    from latency_meta_mdp.legacy.belief.flow.training import train_level_flow_belief

    config = replace(
        load_flow_belief_config(Path("configs/legacy/belief/dinov3_flow_belief_v1.yaml")),
        max_epochs=1,
        early_stopping_patience=1,
    )
    corpus = load_level_feature_belief_corpus(
        project_root=Path.cwd(),
        source_bulk_manifest=source,
        cache_run_manifest=cache,
        expected_spec=load_vision_encoder_spec(
            Path("configs/models/vision/dinov3_vits16_lvd1689m_224_v1.yaml")
        ),
        temporal_config_path=Path("configs/contracts/temporal/h50_e25_d20_k6_v1.yaml"),
        latency_law_path=Path("configs/runtime/latency/truncated_beta_5_26_400ms_v1.yaml"),
        level=1,
    )
    checkpoint = tmp_path / "checkpoint"
    train_level_flow_belief(
        corpus=corpus,
        config=config,
        output_dir=checkpoint,
        device="cuda",
    )

    manifest_path = evaluate_level_flow_belief(
        corpus=corpus,
        config=config,
        checkpoint_dir=checkpoint,
        output_dir=tmp_path / "evaluation",
        device="cuda",
        context_limit=2,
        sample_count=4,
        step_count=2,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = json.loads((manifest_path.parent / "metrics.json").read_text(encoding="utf-8"))

    assert manifest["level"] == 1
    assert manifest["context_count"] == 2
    assert manifest["sample_count"] == 4
    assert manifest["solver_step_count"] == 2
    assert metrics["overall"]["distribution"]["sample_count"] == 4
    assert "object_position" in metrics["overall"]["physical"]
    assert set(metrics["per_delay"]) == {str(delay) for delay in range(1, 21)}
