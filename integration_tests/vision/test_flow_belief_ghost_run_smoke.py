from __future__ import annotations

import json
from pathlib import Path

import pytest

from latency_meta_mdp.artifacts import ImplementationProvenance, sha256_file

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = (
    _PROJECT_ROOT / "outputs/bulk/expert/panda-ball-formal-train-1000-1199-7571a4c/manifest.json"
)
_SAMPLES = (
    _PROJECT_ROOT
    / "outputs/analysis/flow_belief_quality_samples/dinov3-formal-v1-6a85f90/manifest.json"
)


def _require_formal_artifacts() -> None:
    if not _SOURCE.is_file() or not _SAMPLES.is_file():
        pytest.skip("ghost run smoke requires formal source and quality samples")


def _run_kwargs(tmp_path: Path) -> dict:
    return {
        "project_root": _PROJECT_ROOT,
        "source_bulk_manifest": _SOURCE,
        "quality_sample_manifest": _SAMPLES,
        "task_config_path": _PROJECT_ROOT / "configs/task/dynamic_grasp_lift_l0.yaml",
        "control_config_path": _PROJECT_ROOT / "configs/control/panda_osc_pose_delta_v1.yaml",
        "temporal_config_path": _PROJECT_ROOT / "configs/temporal/h50_e25_d20_k6_v1.yaml",
        "ghost_config_path": _PROJECT_ROOT / "configs/analysis/flow_belief_agentview_ghost_v1.yaml",
        "output_dir": tmp_path / "ghost",
        "levels": (1, 2, 3),
    }


def _fake_level_renderer(*, level: int, output_dir: Path, **kwargs) -> Path:
    del kwargs
    output_dir.mkdir()
    panel = output_dir / "panel.png"
    panel.write_bytes(f"L{level}".encode())
    manifest = {
        "schema_version": 1,
        "format_id": "level_flow_belief_ghost_v1",
        "eligible": True,
        "level": level,
        "context_count": 8,
        "invalid_sample_count": level,
        "sample_count": 8 * 5 * 32,
        "artifacts": {"panel.png": sha256_file(panel)},
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_ghost_run_publishes_three_aligned_levels_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_formal_artifacts()
    from latency_meta_mdp.belief.flow import ghost_run

    monkeypatch.setattr(
        ghost_run,
        "collect_implementation_provenance",
        lambda project_root: ImplementationProvenance(
            revision="a" * 40,
            source_sha256="b" * 64,
            dirty=False,
        ),
    )
    manifest_path = ghost_run.render_flow_belief_quality_run(
        **_run_kwargs(tmp_path),
        level_renderer=_fake_level_renderer,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["format_id"] == "flow_belief_ghost_run_v1"
    assert manifest["eligible"] is True
    assert manifest["blockers"] == []
    assert manifest["levels"] == [1, 2, 3]
    assert manifest["level_manifests"] == {
        "L1": "L1/manifest.json",
        "L2": "L2/manifest.json",
        "L3": "L3/manifest.json",
    }
    assert manifest["selected_context_counts"] == {"L1": 8, "L2": 8, "L3": 8}
    assert manifest["invalid_sample_counts"] == {"L1": 1, "L2": 2, "L3": 3}
    assert manifest["implementation_revision"] == "a" * 40

    with pytest.raises(FileExistsError):
        ghost_run.render_flow_belief_quality_run(
            **_run_kwargs(tmp_path),
            level_renderer=_fake_level_renderer,
        )


def test_ghost_run_cleans_staging_after_injected_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_formal_artifacts()
    from latency_meta_mdp.belief.flow import ghost_run

    monkeypatch.setattr(
        ghost_run,
        "collect_implementation_provenance",
        lambda project_root: ImplementationProvenance(
            revision="a" * 40,
            source_sha256="b" * 64,
            dirty=False,
        ),
    )

    def fail_on_l2(*, level: int, output_dir: Path, **kwargs) -> Path:
        if level == 2:
            raise RuntimeError("injected ghost failure")
        return _fake_level_renderer(level=level, output_dir=output_dir, **kwargs)

    with pytest.raises(RuntimeError, match="injected ghost failure"):
        ghost_run.render_flow_belief_quality_run(
            **_run_kwargs(tmp_path),
            level_renderer=fail_on_l2,
        )

    assert not list(tmp_path.glob(".*.building-*"))
    assert not (tmp_path / "ghost").exists()


def test_ghost_run_marks_dirty_implementation_ineligible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_formal_artifacts()
    from latency_meta_mdp.belief.flow import ghost_run

    monkeypatch.setattr(
        ghost_run,
        "collect_implementation_provenance",
        lambda project_root: ImplementationProvenance(
            revision="a" * 40,
            source_sha256="b" * 64,
            dirty=True,
        ),
    )
    manifest_path = ghost_run.render_flow_belief_quality_run(
        **_run_kwargs(tmp_path),
        level_renderer=_fake_level_renderer,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["eligible"] is False
    assert manifest["blockers"] == ["dirty_implementation"]


def test_ghost_run_rejects_quality_manifest_from_another_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_formal_artifacts()
    from latency_meta_mdp.belief.flow import ghost_run

    copied = json.loads(_SAMPLES.read_text(encoding="utf-8"))
    copied["input_sha256"]["source_bulk_manifest"] = "0" * 64
    mismatched = tmp_path / "mismatched_samples.json"
    mismatched.write_text(json.dumps(copied), encoding="utf-8")
    kwargs = _run_kwargs(tmp_path)
    kwargs["quality_sample_manifest"] = mismatched
    monkeypatch.setattr(ghost_run, "_verify_manifest_artifacts", lambda path, manifest: None)

    with pytest.raises(ValueError, match="source corpus"):
        ghost_run.render_flow_belief_quality_run(
            **kwargs,
            level_renderer=_fake_level_renderer,
        )


def test_ghost_cli_exposes_only_selected_case_rendering() -> None:
    from latency_meta_mdp.cli.render_flow_belief_quality import build_parser

    help_text = build_parser().format_help()
    assert "--quality-sample-manifest" in help_text
    assert "--source-bulk-manifest" in help_text
    assert "--output-dir" in help_text
    assert "--levels" in help_text
    assert "--checkpoint" not in help_text


def test_real_l1_ghost_renderer_emits_selected_case_panels(tmp_path: Path) -> None:
    _require_formal_artifacts()
    from latency_meta_mdp.belief.flow.ghost_render_level import (
        render_flow_belief_ghost_level,
    )

    manifest_path = render_flow_belief_ghost_level(
        level=1,
        output_dir=tmp_path / "L1",
        project_root=_PROJECT_ROOT,
        source_bulk_manifest=_SOURCE,
        quality_sample_manifest=_SAMPLES,
        quality_level_manifest=_SAMPLES.parent / "L1/manifest.json",
        task_config_path=_PROJECT_ROOT / "configs/task/dynamic_grasp_lift_l0.yaml",
        control_config_path=_PROJECT_ROOT / "configs/control/panda_osc_pose_delta_v1.yaml",
        temporal_config_path=_PROJECT_ROOT / "configs/temporal/h50_e25_d20_k6_v1.yaml",
        ghost_config_path=_PROJECT_ROOT / "configs/analysis/flow_belief_agentview_ghost_v1.yaml",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["format_id"] == "level_flow_belief_ghost_v1"
    assert manifest["eligible"] is True
    assert manifest["level"] == 1
    assert manifest["context_count"] == 8
    assert manifest["sample_count"] == 8 * 5 * 32
    assert 0 < manifest["invalid_sample_count"] < manifest["sample_count"]
    assert manifest["review_video_kind"] == "selected_case_sequence"
    assert (manifest_path.parent / "selected_cases.mp4").is_file()
    assert len(list(manifest_path.parent.glob("contexts/*/panel.png"))) == 8
    assert len(list(manifest_path.parent.glob("contexts/*/overlays/delay_*.png"))) == 40
