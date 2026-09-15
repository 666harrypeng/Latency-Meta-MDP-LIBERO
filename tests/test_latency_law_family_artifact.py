from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_latency_law_family_certifies_formal_episode_assignments(tmp_path: Path) -> None:
    source = Path("outputs/bulk/expert/panda-ball-formal-train-1000-1199-7571a4c/manifest.json")
    if not source.is_file():
        pytest.skip("latency-law family artifact requires the formal expert corpus")
    from latency_meta_mdp.runtime.latency_law_family_artifact import (
        certify_episode_latency_law_family,
    )

    manifest_path = certify_episode_latency_law_family(
        project_root=Path.cwd(),
        source_bulk_manifest=source,
        nominal_law_path=Path("configs/runtime/latency/truncated_beta_5_26_400ms_v1.yaml"),
        family_config_path=Path("configs/runtime/latency/truncated_beta_family_5_26_400ms_v1.yaml"),
        output_dir=tmp_path / "family",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads((manifest_path.parent / "summary.json").read_text(encoding="utf-8"))
    assignments = json.loads(
        (manifest_path.parent / "assignments.json").read_text(encoding="utf-8")
    )
    assert manifest["format_id"] == "episode_latency_law_family_artifact_v1"
    assert summary["assignment_count"] == 600
    assert summary["unique_scene_seed_count"] == 200
    assert summary["unique_probability_count"] == 200
    assert summary["cross_level_probability_mismatch_count"] == 0
    assert len(assignments) == 600
    assert len(assignments[0]["probabilities"]) == 20
