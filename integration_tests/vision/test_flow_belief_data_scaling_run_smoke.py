from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_data_scaling_run_trains_one_bounded_cell(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Flow data-scaling integration smoke requires CUDA")
    source = Path("outputs/bulk/expert/panda-ball-formal-train-1000-1199-7571a4c/manifest.json")
    cache = Path(
        "outputs/derived/vision_features/dinov3-vits16-formal-1000-1199-408dbe3/manifest.json"
    )
    physical_audit = Path(
        "outputs/analysis/flow_belief_data_sufficiency/physical-coverage-v2-3b4b201/manifest.json"
    )
    if not source.is_file() or not cache.is_file() or not physical_audit.is_file():
        pytest.skip("Flow data-scaling smoke requires formal local artifacts")
    from latency_meta_mdp.legacy.belief.flow.data_scaling_run import (
        run_flow_belief_data_scaling,
    )

    manifest_path = run_flow_belief_data_scaling(
        project_root=Path.cwd(),
        source_bulk_manifest=source,
        cache_run_manifest=cache,
        physical_audit_manifest=physical_audit,
        vision_config_path=Path("configs/models/vision/dinov3_vits16_lvd1689m_224_v1.yaml"),
        temporal_config_path=Path("configs/contracts/temporal/h50_e25_d20_k6_v1.yaml"),
        latency_law_path=Path("configs/runtime/latency/truncated_beta_5_26_400ms_v1.yaml"),
        flow_config_path=Path("configs/legacy/belief/dinov3_flow_belief_v1.yaml"),
        split_config_path=Path("configs/legacy/data/formal_belief_train_val_v1.yaml"),
        audit_config_path=Path("configs/legacy/analysis/dinov3_flow_belief_data_scaling_v1.yaml"),
        output_dir=tmp_path / "scaling",
        levels=(1,),
        device="cuda",
        subset_sizes=(60,),
        model_seeds=(20260829,),
        max_epochs=1,
        training_context_limit=8,
        validation_context_limit=4,
        evaluation_sample_count=2,
        solver_step_count=2,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cell = manifest_path.parent / "L1/subset_0060/seed_20260829"
    assert manifest["format_id"] == "flow_belief_data_scaling_run_v1"
    assert manifest["levels"] == [1]
    assert manifest["subset_sizes"] == [60]
    assert manifest["model_seeds"] == [20260829]
    assert (cell / "checkpoint/model.safetensors").is_file()
    assert (cell / "evaluation/metrics.json").is_file()
    assert (cell / "scaling_metrics.json").is_file()
