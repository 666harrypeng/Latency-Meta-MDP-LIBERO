from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_physical_data_audit_reads_all_formal_episodes(tmp_path: Path) -> None:
    source = Path("outputs/bulk/expert/panda-ball-formal-train-1000-1199-7571a4c/manifest.json")
    if not source.is_file():
        pytest.skip("physical data audit requires the formal expert corpus")
    from latency_meta_mdp.legacy.belief.flow.data_sufficiency import (
        write_flow_belief_physical_data_audit,
    )

    manifest_path = write_flow_belief_physical_data_audit(
        project_root=Path.cwd(),
        source_bulk_manifest=source,
        split_config_path=Path("configs/legacy/data/formal_belief_train_val_v1.yaml"),
        audit_config_path=Path("configs/legacy/analysis/dinov3_flow_belief_data_scaling_v1.yaml"),
        output_dir=tmp_path / "audit",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["format_id"] == "flow_belief_physical_data_audit_v1"
    assert manifest["levels"] == [1, 2, 3]
    assert manifest["episode_counts"] == {"L1": 200, "L2": 200, "L3": 200}
    for level in (1, 2, 3):
        level_dir = manifest_path.parent / f"L{level}"
        summary = json.loads((level_dir / "summary.json").read_text(encoding="utf-8"))
        subsets = json.loads((level_dir / "nested_subsets.json").read_text(encoding="utf-8"))
        assert summary["episode_count"] == 200
        assert summary["training_episode_count"] == 180
        assert summary["validation_episode_count"] == 20
        assert {key: len(value) for key, value in subsets.items()} == {
            "60": 60,
            "120": 120,
            "180": 180,
        }
