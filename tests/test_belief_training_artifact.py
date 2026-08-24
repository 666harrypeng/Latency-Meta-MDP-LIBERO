from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from latency_meta_mdp.artifacts import sha256_file
from latency_meta_mdp.episode_artifacts import write_synchronized_episode_artifact
from latency_meta_mdp.expert_collection import ExpertEpisodeSpec, collect_expert_episode
from latency_meta_mdp.recording import RecordProfile


def _source_manifest(tmp_path: Path) -> Path:
    episode = collect_expert_episode(
        project_root=Path.cwd(),
        spec=ExpertEpisodeSpec(
            episode_id="l1-seed-001000-belief-index",
            level=1,
            scene_seed=1_000,
            motion_seed=1_000,
            expert_seed=1_000,
            record_profile=RecordProfile.BELIEF,
            camera_width=8,
            camera_height=8,
        ),
    )
    root = tmp_path / "source"
    episode_root = root / "attempts" / "L1" / "seed_001000"
    episode_manifest = write_synchronized_episode_artifact(
        episode=episode,
        output_dir=episode_root,
    )
    relative = episode_manifest.relative_to(root).as_posix()
    source = root / "manifest.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format_id": "panda_ball_bulk_first_tranche_v1",
                "implementation_dirty": False,
                "eligible": True,
                "admitted_episode_manifests": [relative],
                "artifacts": {relative: sha256_file(episode_manifest)},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return source


def test_belief_training_split_plan_is_episode_level_and_shared_across_levels() -> None:
    from latency_meta_mdp.belief_training_artifact import (
        SplitName,
        load_belief_training_split_plan,
    )

    plan = load_belief_training_split_plan(
        Path("configs/data/belief_training_split_v1.yaml")
    )

    assert plan.formal_counts == (160, 20, 20)
    assert plan.first_tranche_counts == (20, 2, 3)
    assert plan.split_for_seed(1_000, first_tranche=True) is SplitName.TRAIN
    assert plan.split_for_seed(1_019, first_tranche=True) is SplitName.TRAIN
    assert plan.split_for_seed(1_020, first_tranche=True) is SplitName.VALIDATION
    assert plan.split_for_seed(1_022, first_tranche=True) is SplitName.HOLDOUT
    assert plan.split_for_seed(1_159, first_tranche=False) is SplitName.TRAIN
    assert plan.split_for_seed(1_160, first_tranche=False) is SplitName.VALIDATION
    assert plan.split_for_seed(1_180, first_tranche=False) is SplitName.HOLDOUT


def test_belief_training_index_artifact_is_atomic_and_contains_no_images(
    tmp_path: Path,
) -> None:
    from latency_meta_mdp.belief_training_artifact import (
        write_belief_training_index_artifact,
    )

    source = _source_manifest(tmp_path)
    output = tmp_path / "derived-index"
    manifest_path = write_belief_training_index_artifact(
        project_root=Path.cwd(),
        source_bulk_manifest=source,
        split_config_path=Path("configs/data/belief_training_split_v1.yaml"),
        bulk_plan_path=Path("configs/collection/panda_ball_bulk_v1.yaml"),
        view_config_path=Path("configs/data/belief_data_view_h50_v2.yaml"),
        latency_law_path=Path(
            "configs/latency/truncated_beta_5_26_400ms_v1.yaml"
        ),
        output_dir=output,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert manifest["format_id"] == "belief_training_index_artifact_v1"
    assert manifest["eligible"] is (not manifest["implementation_dirty"])
    assert set(manifest["artifacts"]) == {"index.npz", "summary.json"}
    assert summary["format_id"] == "belief_training_index_summary_v1"
    assert summary["episode_count"] == 1
    assert summary["master_seed_count"] == 1
    assert summary["context_count"] > 0
    assert summary["splits"]["train"]["episode_count"] == 1
    assert summary["splits"]["train"]["master_seed_count"] == 1
    assert summary["splits"]["train"]["context_count"] == summary["context_count"]
    assert summary["context_contract"]["image_history_ticks"] == 6
    assert summary["context_contract"]["robot_proprio_dim"] == 16
    assert summary["context_contract"]["remaining_action_shape"] == [25, 7]
    assert summary["target_contract"]["delay_count"] == 20
    assert summary["target_contract"]["state_dim"] == 22
    assert summary["target_contract"]["absorbing_tail_ticks"] == 70

    with np.load(output / "index.npz", allow_pickle=False) as arrays:
        assert all(arrays[name].dtype != object for name in arrays.files)
        assert len(arrays["source_tick"]) == summary["context_count"]
        assert np.all(arrays["scene_seed"] == 1_000)
        assert np.all(arrays["split"] == "train")
        assert not any("image" in name for name in arrays.files)

    with pytest.raises(FileExistsError, match="already exists"):
        write_belief_training_index_artifact(
            project_root=Path.cwd(),
            source_bulk_manifest=source,
            split_config_path=Path(
                "configs/data/belief_training_split_v1.yaml"
            ),
            bulk_plan_path=Path("configs/collection/panda_ball_bulk_v1.yaml"),
            view_config_path=Path("configs/data/belief_data_view_h50_v2.yaml"),
            latency_law_path=Path(
                "configs/latency/truncated_beta_5_26_400ms_v1.yaml"
            ),
            output_dir=output,
        )
