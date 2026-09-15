from __future__ import annotations

import json
from pathlib import Path

from latency_meta_mdp.legacy.belief.conditional_return_flow.branch_artifacts import (
    load_verified_control_branch_corpus,
)
from latency_meta_mdp.legacy.belief.conditional_return_flow.branch_collection import (
    collect_control_branch_corpus,
)


def test_bounded_real_l1_collection_publishes_complete_branch_inventory(
    tmp_path: Path,
) -> None:
    project_root = Path.cwd()
    output = tmp_path / "branches"

    manifest = collect_control_branch_corpus(
        project_root=project_root,
        source_bulk_manifest=project_root
        / "outputs/bulk/expert/panda-ball-formal-train-1000-1199-7571a4c/manifest.json",
        split_config_path=project_root / "configs/legacy/data/formal_belief_train_val_v1.yaml",
        temporal_config_path=project_root / "configs/contracts/temporal/h50_e25_d20_k6_v1.yaml",
        branch_config_path=project_root
        / "configs/legacy/belief/conditional_return_flow/branch_corpus.yaml",
        output_dir=output,
        levels=(1,),
        maximum_contexts_per_level=1,
        allowed_scene_seed_ranges=((1000, 1001),),
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    loaded = load_verified_control_branch_corpus(manifest)

    assert payload["context_count"] == 1
    assert payload["branch_count"] == 6
    assert payload["allowed_scene_seed_ranges"] == [[1000, 1001]]
    assert payload["selected_episode_count"] == 1
    assert len(payload["selected_episode_identities_sha256"]) == 64
    assert payload["selection_slot_count"] == 4
    assert payload["selection_exclusion_count"] == 0
    assert payload["selection_truncation_count"] == 3
    assert payload["collection_wall_time_seconds"] > 0
    assert payload["peak_rss_bytes"] > 0
    assert payload["scientific_gate_pass"] is True
    assert payload["artifact_eligible"] is False
    assert "bounded_review" in payload["artifact_blockers"]
    assert loaded.source_context_index.tolist() == [0, 0, 0, 0, 0, 0]
    assert set(loaded.branch_kind.tolist()) == {
        "nominal",
        "hold",
        "arm_scale_0.5",
        "arm_scale_0.8",
        "prefix_hold_5",
        "prefix_hold_10",
    }
    assert loaded.source_fingerprint_match.tolist() == [True] * 6


def test_seed_scoped_collection_is_bounded_without_context_limit(tmp_path: Path) -> None:
    project_root = Path.cwd()
    manifest = collect_control_branch_corpus(
        project_root=project_root,
        source_bulk_manifest=project_root
        / "outputs/bulk/expert/panda-ball-formal-train-1000-1199-7571a4c/manifest.json",
        split_config_path=project_root / "configs/legacy/data/formal_belief_train_val_v1.yaml",
        temporal_config_path=project_root / "configs/contracts/temporal/h50_e25_d20_k6_v1.yaml",
        branch_config_path=project_root
        / "configs/legacy/belief/conditional_return_flow/branch_corpus.yaml",
        output_dir=tmp_path / "seed-scoped-branches",
        levels=(1,),
        allowed_scene_seed_ranges=((1000, 1001),),
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert payload["bounded"] is True
    assert "bounded_review" in payload["artifact_blockers"]
