from __future__ import annotations

import json
from pathlib import Path

import pytest

from latency_meta_mdp.io.artifacts import ImplementationProvenance
from latency_meta_mdp.io.paths import repository_root

_PROJECT_ROOT = repository_root()
_SOURCE = (
    _PROJECT_ROOT / "outputs/bulk/expert/panda-ball-formal-train-1000-1199-7571a4c/manifest.json"
)
_CACHE = (
    _PROJECT_ROOT
    / "outputs/derived/vision_features/dinov3-vits16-formal-1000-1199-408dbe3/manifest.json"
)
_FLOW = _PROJECT_ROOT / "outputs/analysis/flow_belief/dinov3-formal-180-20-f6caf55/manifest.json"
_EVALUATION = (
    _PROJECT_ROOT
    / "outputs/analysis/flow_belief_evaluation/dinov3-formal-180-20-f6caf55/manifest.json"
)


def _require_formal_artifacts() -> None:
    for path in (_SOURCE, _CACHE, _FLOW, _EVALUATION):
        if not path.is_file():
            pytest.skip("quality run smoke requires formal Flow artifacts")


def _run_kwargs(tmp_path: Path) -> dict:
    return {
        "project_root": _PROJECT_ROOT,
        "source_bulk_manifest": _SOURCE,
        "cache_run_manifest": _CACHE,
        "flow_run_manifest": _FLOW,
        "evaluation_run_manifest": _EVALUATION,
        "vision_config_path": _PROJECT_ROOT
        / "configs/models/vision/dinov3_vits16_lvd1689m_224_v1.yaml",
        "temporal_config_path": _PROJECT_ROOT / "configs/contracts/temporal/h50_e25_d20_k6_v1.yaml",
        "latency_law_path": _PROJECT_ROOT
        / "configs/runtime/latency/truncated_beta_5_26_400ms_v1.yaml",
        "flow_config_path": _PROJECT_ROOT / "configs/legacy/belief/dinov3_flow_belief_v1.yaml",
        "split_config_path": _PROJECT_ROOT / "configs/legacy/data/formal_belief_train_val_v1.yaml",
        "quality_config_path": _PROJECT_ROOT
        / "configs/legacy/analysis/flow_belief_quality_samples_v1.yaml",
        "output_dir": tmp_path / "quality",
        "levels": (1, 2, 3),
        "device": "cuda",
    }


def _fake_level_processor(*, level: int, output_dir: Path, **kwargs) -> Path:
    del kwargs
    output_dir.mkdir()
    manifest = {
        "schema_version": 1,
        "format_id": "level_flow_belief_quality_samples_v1",
        "eligible": True,
        "level": level,
        "context_count": 2,
        "sample_count": 32,
        "role_counts": {"typical": 1, "p95_hard": 1},
        "artifacts": {},
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_quality_run_publishes_three_level_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_formal_artifacts()
    from latency_meta_mdp.legacy.belief.flow import quality_run

    monkeypatch.setattr(
        quality_run,
        "collect_implementation_provenance",
        lambda project_root: ImplementationProvenance(
            revision="a" * 40,
            source_sha256="b" * 64,
            dirty=False,
        ),
    )
    manifest_path = quality_run.export_flow_belief_quality_sample_run(
        **_run_kwargs(tmp_path),
        level_processor=_fake_level_processor,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["format_id"] == "flow_belief_quality_sample_run_v1"
    assert manifest["levels"] == [1, 2, 3]
    assert manifest["eligible"] is True
    assert manifest["blockers"] == []
    assert manifest["implementation_revision"] == "a" * 40
    assert manifest["wall_seconds"] >= 0.0
    assert manifest["level_manifests"] == {
        "L1": "L1/manifest.json",
        "L2": "L2/manifest.json",
        "L3": "L3/manifest.json",
    }
    assert manifest["selected_context_counts"] == {"L1": 2, "L2": 2, "L3": 2}

    with pytest.raises(FileExistsError):
        quality_run.export_flow_belief_quality_sample_run(
            **_run_kwargs(tmp_path),
            level_processor=_fake_level_processor,
        )


def test_quality_run_removes_staging_after_level_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_formal_artifacts()
    from latency_meta_mdp.legacy.belief.flow import quality_run

    monkeypatch.setattr(
        quality_run,
        "collect_implementation_provenance",
        lambda project_root: ImplementationProvenance(
            revision="a" * 40,
            source_sha256="b" * 64,
            dirty=False,
        ),
    )

    def fail_on_l2(*, level: int, output_dir: Path, **kwargs) -> Path:
        if level == 2:
            raise RuntimeError("injected level failure")
        return _fake_level_processor(level=level, output_dir=output_dir, **kwargs)

    with pytest.raises(RuntimeError, match="injected level failure"):
        quality_run.export_flow_belief_quality_sample_run(
            **_run_kwargs(tmp_path),
            level_processor=fail_on_l2,
        )

    assert not list(tmp_path.glob("*.building-*"))
    assert not (tmp_path / "quality").exists()


def test_quality_cli_exposes_only_validation_sample_export() -> None:
    from latency_meta_mdp.legacy.cli.export_flow_belief_quality_samples import build_parser

    help_text = build_parser().format_help()
    assert "--evaluation-run-manifest" in help_text
    assert "--split-config" in help_text
    assert "--output-dir" in help_text
    assert "--device" in help_text
    assert "--evaluation-split" not in help_text
