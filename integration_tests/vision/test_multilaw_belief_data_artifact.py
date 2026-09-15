from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_multilaw_belief_data_artifact_certifies_runtime_view(tmp_path: Path) -> None:
    source = Path("outputs/bulk/expert/panda-ball-formal-train-1000-1199-7571a4c/manifest.json")
    cache = Path(
        "outputs/derived/vision_features/dinov3-vits16-formal-1000-1199-408dbe3/manifest.json"
    )
    if not source.is_file() or not cache.is_file():
        pytest.skip("multi-law Belief data artifact requires formal local artifacts")
    from latency_meta_mdp.legacy.belief.flow.multilaw_data_artifact import (
        certify_multilaw_belief_data,
    )

    manifest_path = certify_multilaw_belief_data(
        project_root=Path.cwd(),
        source_bulk_manifest=source,
        cache_run_manifest=cache,
        vision_config_path=Path("configs/models/vision/dinov3_vits16_lvd1689m_224_v1.yaml"),
        temporal_config_path=Path("configs/contracts/temporal/h50_e25_d20_k6_v1.yaml"),
        nominal_law_path=Path("configs/runtime/latency/truncated_beta_5_26_400ms_v1.yaml"),
        family_config_path=Path("configs/runtime/latency/truncated_beta_family_5_26_400ms_v1.yaml"),
        split_config_path=Path("configs/legacy/data/formal_belief_train_val_v1.yaml"),
        multilaw_config_path=Path("configs/legacy/belief/dinov3_flow_belief_multilaw_v2.yaml"),
        output_dir=tmp_path / "multilaw-data",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["format_id"] == "multilaw_flow_belief_data_artifact_v1"
    assert manifest["levels"] == [1, 2, 3]
    for level in (1, 2, 3):
        summary = json.loads(
            (manifest_path.parent / f"L{level}/summary.json").read_text(encoding="utf-8")
        )
        assert summary["episode_counts"] == {
            "holdout": 0,
            "train": 180,
            "validation": 20,
        }
        assert summary["unique_episode_law_count"] == 200
        assert summary["realized_delay_exposed"] is False
        assert summary["minimum_training_query_probability"] >= 0.005
        assert (manifest_path.parent / f"L{level}/normalization.npz").is_file()
