from __future__ import annotations

import json
from pathlib import Path

import pytest

from latency_meta_mdp.data.expert_collection import ExpertEpisodeSpec, collect_expert_episode
from latency_meta_mdp.data.recording import RecordProfile
from latency_meta_mdp.io.artifacts import sha256_file
from latency_meta_mdp.io.episode_artifacts import write_synchronized_episode_artifact


def _source_bulk_manifest(tmp_path: Path) -> Path:
    episode = collect_expert_episode(
        project_root=Path.cwd(),
        spec=ExpertEpisodeSpec(
            episode_id="l2-seed-000010-belief-certification",
            level=2,
            scene_seed=10,
            motion_seed=10,
            expert_seed=10,
            record_profile=RecordProfile.BELIEF,
            camera_width=8,
            camera_height=8,
        ),
    )
    source = tmp_path / "source"
    episode_root = source / "attempts" / "L2" / "seed_000010"
    episode_manifest = write_synchronized_episode_artifact(
        episode=episode,
        output_dir=episode_root,
    )
    relative = episode_manifest.relative_to(source).as_posix()
    manifest = {
        "schema_version": 1,
        "format_id": "panda_ball_bulk_first_tranche_v1",
        "implementation_dirty": False,
        "eligible": True,
        "admitted_episode_manifests": [relative],
        "artifacts": {relative: sha256_file(episode_manifest)},
    }
    (source / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return source / "manifest.json"


def test_checked_in_belief_data_view_config_is_valid() -> None:
    from latency_meta_mdp.legacy.belief_data_artifact import load_belief_data_view_config

    config = load_belief_data_view_config(Path("configs/legacy/data/belief_data_view_h50_v2.yaml"))

    assert config.view_id == "belief_data_view_h50_v2"
    assert config.history_ticks == 6
    assert config.temporal_contract.contract_id == "h50_e25_d20_k6_v1"
    assert config.buffer_protocol_status == "sharp_teacher_buffer_bound"
    assert config.action_chunk_alignment_status == "return_time"


def test_belief_data_certification_is_atomic_and_preserves_open_design_items(
    tmp_path: Path,
) -> None:
    from latency_meta_mdp.legacy.belief_data_artifact import certify_belief_data_contract

    source_manifest = _source_bulk_manifest(tmp_path)
    output = tmp_path / "certification"

    result = certify_belief_data_contract(
        project_root=Path.cwd(),
        source_bulk_manifest=source_manifest,
        view_config_path=Path("configs/legacy/data/belief_data_view_h50_v2.yaml"),
        latency_law_path=Path("configs/runtime/latency/truncated_beta_5_26_400ms_v1.yaml"),
        output_dir=output,
    )

    report = json.loads(result.read_text(encoding="utf-8"))
    assert report["format_id"] == "belief_raw_index_certification_v2"
    assert report["raw_index_ready"]
    assert report["teacher_buffer_ready"]
    assert report["eligible"] is (not report["implementation_dirty"])
    assert not report["belief_training_ready"]
    assert report["open_design_items"] == [
        "belief_target_representation",
        "belief_action_policy_interface",
    ]
    assert report["temporal_contract_id"] == "h50_e25_d20_k6_v1"
    assert report["prediction_horizon"] == 50
    assert report["launch_trigger_horizon"] == 25
    assert report["latency_condition_dim"] == 20
    assert report["history_sample_count"] == 6
    assert report["history_span_ms"] == 100
    assert report["levels"]["2"]["episode_count"] == 1
    assert report["levels"]["2"]["branch_index_count"] > 0
    assert set(report["levels"]["2"]["per_delay_index_count"]) == {
        str(delay) for delay in range(1, 21)
    }
    with pytest.raises(FileExistsError, match="already exists"):
        certify_belief_data_contract(
            project_root=Path.cwd(),
            source_bulk_manifest=source_manifest,
            view_config_path=Path("configs/legacy/data/belief_data_view_h50_v2.yaml"),
            latency_law_path=Path("configs/runtime/latency/truncated_beta_5_26_400ms_v1.yaml"),
            output_dir=output,
        )
