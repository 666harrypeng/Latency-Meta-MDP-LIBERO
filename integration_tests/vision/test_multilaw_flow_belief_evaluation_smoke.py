from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def test_multilaw_evaluation_runs_all_law_regimes(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("multi-law Flow evaluation smoke requires CUDA")
    source = Path("outputs/bulk/expert/panda-ball-formal-train-1000-1199-7571a4c/manifest.json")
    cache = Path(
        "outputs/derived/vision_features/dinov3-vits16-formal-1000-1199-408dbe3/manifest.json"
    )
    data_artifact = Path(
        "outputs/analysis/multilaw_belief_data/formal-180-20-476ecff/manifest.json"
    )
    if not source.is_file() or not cache.is_file() or not data_artifact.is_file():
        pytest.skip("multi-law Flow evaluation smoke requires formal local artifacts")
    from latency_meta_mdp.legacy.belief.flow.multilaw_evaluation_run import (
        evaluate_multilaw_flow_belief_run,
    )
    from latency_meta_mdp.legacy.belief.flow.multilaw_run import (
        train_multilaw_flow_belief_run,
    )

    common = dict(
        project_root=Path.cwd(),
        source_bulk_manifest=source,
        cache_run_manifest=cache,
        vision_config_path=Path("configs/models/vision/dinov3_vits16_lvd1689m_224_v1.yaml"),
        temporal_config_path=Path("configs/contracts/temporal/h50_e25_d20_k6_v1.yaml"),
        nominal_law_path=Path("configs/runtime/latency/truncated_beta_5_26_400ms_v1.yaml"),
        family_config_path=Path("configs/runtime/latency/truncated_beta_family_5_26_400ms_v1.yaml"),
        flow_config_path=Path("configs/legacy/belief/dinov3_flow_belief_v1.yaml"),
        multilaw_config_path=Path("configs/legacy/belief/dinov3_flow_belief_multilaw_v2.yaml"),
        split_config_path=Path("configs/legacy/data/formal_belief_train_val_v1.yaml"),
        levels=(1,),
        device="cuda",
    )
    training_manifest = train_multilaw_flow_belief_run(
        **common,
        multilaw_data_manifest=data_artifact,
        output_dir=tmp_path / "train",
        max_epochs=1,
        training_context_limit=8,
        validation_context_limit=4,
    )
    manifest_path = evaluate_multilaw_flow_belief_run(
        **common,
        multilaw_run_manifest=training_manifest,
        evaluation_config_path=Path(
            "configs/legacy/analysis/dinov3_flow_belief_multilaw_evaluation_v1.yaml"
        ),
        output_dir=tmp_path / "evaluation",
        context_limit=4,
        sample_count=2,
        solver_step_count=2,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["format_id"] == "flow_belief_multilaw_evaluation_run_v1"
    assert manifest["regimes"] == [
        "nominal",
        "in_family",
        "shifted_fast",
        "shifted_slow",
        "shifted_wide",
    ]
    nominal = np.load(
        manifest_path.parent / "L1/nominal/summary_arrays.npz",
        allow_pickle=False,
    )
    in_family = np.load(
        manifest_path.parent / "L1/in_family/summary_arrays.npz",
        allow_pickle=False,
    )
    try:
        assert nominal["latency_probabilities"].shape == (4, 20)
        assert in_family["latency_probabilities"].shape == (4, 20)
        assert not np.array_equal(
            nominal["latency_probabilities"],
            in_family["latency_probabilities"],
        )
    finally:
        nominal.close()
        in_family.close()
