from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_multilaw_flow_training_runs_from_random_initialization(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("multi-law Flow training smoke requires CUDA")
    source = Path("outputs/bulk/expert/panda-ball-formal-train-1000-1199-7571a4c/manifest.json")
    cache = Path(
        "outputs/derived/vision_features/dinov3-vits16-formal-1000-1199-408dbe3/manifest.json"
    )
    data_artifact = Path(
        "outputs/analysis/multilaw_belief_data/formal-180-20-476ecff/manifest.json"
    )
    if not source.is_file() or not cache.is_file() or not data_artifact.is_file():
        pytest.skip("multi-law Flow training smoke requires formal local artifacts")
    from latency_meta_mdp.legacy.belief.flow.multilaw_run import (
        train_multilaw_flow_belief_run,
    )

    manifest_path = train_multilaw_flow_belief_run(
        project_root=Path.cwd(),
        source_bulk_manifest=source,
        cache_run_manifest=cache,
        multilaw_data_manifest=data_artifact,
        vision_config_path=Path("configs/models/vision/dinov3_vits16_lvd1689m_224_v1.yaml"),
        temporal_config_path=Path("configs/contracts/temporal/h50_e25_d20_k6_v1.yaml"),
        nominal_law_path=Path("configs/runtime/latency/truncated_beta_5_26_400ms_v1.yaml"),
        family_config_path=Path("configs/runtime/latency/truncated_beta_family_5_26_400ms_v1.yaml"),
        flow_config_path=Path("configs/legacy/belief/dinov3_flow_belief_v1.yaml"),
        multilaw_config_path=Path("configs/legacy/belief/dinov3_flow_belief_multilaw_v2.yaml"),
        split_config_path=Path("configs/legacy/data/formal_belief_train_val_v1.yaml"),
        output_dir=tmp_path / "run",
        levels=(1,),
        device="cuda",
        max_epochs=1,
        training_context_limit=8,
        validation_context_limit=4,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    level_manifest = json.loads(
        (manifest_path.parent / "L1/manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["format_id"] == "flow_belief_multilaw_run_v2"
    assert manifest["initialization"] == "random_multilaw_flow_models"
    assert manifest["levels"] == [1]
    assert level_manifest["initialization"] == "random_flow_encoder_and_vector_field"
    assert level_manifest["latency_law_family_id"] == ("truncated_beta_family_5_26_400ms_v1")
    assert level_manifest["delay_query_uniform_mix"] == 0.10
    assert (manifest_path.parent / "L1/encoder.safetensors").is_file()
