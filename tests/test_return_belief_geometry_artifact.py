from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from latency_meta_mdp.artifacts import sha256_file
from latency_meta_mdp.episode_artifacts import write_synchronized_episode_artifact
from latency_meta_mdp.expert_collection import ExpertEpisodeSpec, collect_expert_episode
from latency_meta_mdp.recording import RecordProfile


def _source_bulk_manifest(tmp_path: Path) -> Path:
    episode = collect_expert_episode(
        project_root=Path.cwd(),
        spec=ExpertEpisodeSpec(
            episode_id="l2-seed-000010-return-belief-audit",
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
    source_manifest = source / "manifest.json"
    source_manifest.write_text(
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
    return source_manifest


def _module():
    return importlib.import_module(
        "latency_meta_mdp.return_belief_geometry_artifact"
    )


def test_return_belief_geometry_artifact_is_atomic_numeric_and_stratified(
    tmp_path: Path,
) -> None:
    module = _module()
    source = _source_bulk_manifest(tmp_path)
    output = tmp_path / "analysis"

    manifest_path = module.write_return_belief_geometry_artifact(
        project_root=Path.cwd(),
        source_bulk_manifest=source,
        audit_config_path=Path(
            "configs/analysis/return_belief_geometry_v1.yaml"
        ),
        view_config_path=Path("configs/data/belief_data_view_h50_v2.yaml"),
        latency_law_path=Path(
            "configs/latency/truncated_beta_5_26_400ms_v1.yaml"
        ),
        output_dir=output,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["format_id"] == "return_belief_geometry_artifact_v1"
    assert manifest["eligible"] is (not manifest["implementation_dirty"])
    assert set(manifest["artifacts"]) == {
        "context_metrics.npz",
        "summary.json",
        "summary.svg",
    }
    assert all(len(value) == 64 for value in manifest["artifacts"].values())

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["analysis_id"] == "return_belief_geometry_v1"
    assert summary["episode_count"] == 1
    assert summary["levels"]["2"]["episode_count"] == 1
    assert summary["levels"]["2"]["state_context_count"] > 0
    assert summary["levels"]["2"]["action_context_count"] > 0
    assert set(summary["levels"]["2"]["phases"]) <= {
        "pregrasp",
        "approach",
        "close",
        "lift",
    }
    assert summary["state_schema"]["dimension"] == 22
    assert len(summary["state_normalization"]["scale"]) == 22
    assert summary["latency_law"]["delay_ticks"] == list(range(1, 21))

    with np.load(output / "context_metrics.npz", allow_pickle=False) as arrays:
        assert all(arrays[name].dtype != object for name in arrays.files)
        state_count = len(arrays["state_source_tick"])
        action_count = len(arrays["action_source_tick"])
        assert state_count == summary["levels"]["2"]["state_context_count"]
        assert action_count == summary["levels"]["2"]["action_context_count"]
        for name in arrays.files:
            if name.startswith(("state_metric_", "action_metric_")):
                assert np.all(np.isfinite(arrays[name]))

    svg = (output / "summary.svg").read_text(encoding="utf-8")
    assert svg.startswith("<svg")
    assert "Return-Belief Geometry" in svg
    assert "L2" in svg
    assert 'data-metric="object_position_rms_spread"' in svg

    with pytest.raises(FileExistsError, match="already exists"):
        module.write_return_belief_geometry_artifact(
            project_root=Path.cwd(),
            source_bulk_manifest=source,
            audit_config_path=Path(
                "configs/analysis/return_belief_geometry_v1.yaml"
            ),
            view_config_path=Path("configs/data/belief_data_view_h50_v2.yaml"),
            latency_law_path=Path(
                "configs/latency/truncated_beta_5_26_400ms_v1.yaml"
            ),
            output_dir=output,
        )


def test_return_belief_geometry_rejects_ineligible_source_before_output(
    tmp_path: Path,
) -> None:
    module = _module()
    source = _source_bulk_manifest(tmp_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["eligible"] = False
    source.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    output = tmp_path / "analysis"

    with pytest.raises(ValueError, match="eligible clean bulk source"):
        module.write_return_belief_geometry_artifact(
            project_root=Path.cwd(),
            source_bulk_manifest=source,
            audit_config_path=Path(
                "configs/analysis/return_belief_geometry_v1.yaml"
            ),
            view_config_path=Path("configs/data/belief_data_view_h50_v2.yaml"),
            latency_law_path=Path(
                "configs/latency/truncated_beta_5_26_400ms_v1.yaml"
            ),
            output_dir=output,
        )
    assert not output.exists()
