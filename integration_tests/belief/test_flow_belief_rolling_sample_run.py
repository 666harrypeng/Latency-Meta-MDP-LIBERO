# ruff: noqa: E402

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from latency_meta_mdp.artifacts import ImplementationProvenance, sha256_file

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = (
    _PROJECT_ROOT / "outputs/bulk/expert/panda-ball-formal-train-1000-1199-7571a4c/manifest.json"
)
_CACHE = (
    _PROJECT_ROOT
    / "outputs/derived/vision_features/dinov3-vits16-formal-1000-1199-408dbe3/manifest.json"
)
_FLOW = _PROJECT_ROOT / "outputs/analysis/flow_belief/dinov3-formal-180-20-f6caf55/manifest.json"


def _require_formal_artifacts() -> None:
    for path in (_SOURCE, _CACHE, _FLOW):
        if not path.is_file():
            pytest.skip("rolling sample run requires formal Flow artifacts")


def _run_kwargs(tmp_path: Path) -> dict:
    return {
        "project_root": _PROJECT_ROOT,
        "source_bulk_manifest": _SOURCE,
        "cache_run_manifest": _CACHE,
        "flow_run_manifest": _FLOW,
        "vision_config_path": _PROJECT_ROOT / "configs/vision/dinov3_vits16_lvd1689m_224_v1.yaml",
        "temporal_config_path": _PROJECT_ROOT / "configs/temporal/h50_e25_d20_k6_v1.yaml",
        "latency_law_path": _PROJECT_ROOT / "configs/latency/truncated_beta_5_26_400ms_v1.yaml",
        "flow_config_path": _PROJECT_ROOT / "configs/belief/dinov3_flow_belief_v1.yaml",
        "split_config_path": _PROJECT_ROOT / "configs/data/formal_belief_train_val_v1.yaml",
        "rolling_config_path": _PROJECT_ROOT
        / "configs/analysis/flow_belief_rolling_inspection_v1.yaml",
        "output_dir": tmp_path / "rolling",
        "level_seeds": ((1, 1180), (2, 1199), (3, 1193)),
        "device": "cpu",
    }


def _fake_level_processor(*, level: int, scene_seed: int, output_dir: Path, **kwargs) -> Path:
    del kwargs
    output_dir.mkdir(parents=True)
    samples = output_dir / "samples.npz"
    samples.write_bytes(f"L{level}-{scene_seed}".encode())
    manifest = {
        "schema_version": 1,
        "format_id": "level_flow_belief_rolling_samples_v1",
        "eligible": True,
        "level": level,
        "scene_seed": scene_seed,
        "window_count": 2,
        "artifacts": {"samples.npz": sha256_file(samples)},
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_rolling_sample_run_publishes_independent_level_seed_bundles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_formal_artifacts()
    from latency_meta_mdp.belief.flow import rolling_sample_run

    monkeypatch.setattr(
        rolling_sample_run,
        "collect_implementation_provenance",
        lambda project_root: ImplementationProvenance(
            revision="a" * 40,
            source_sha256="b" * 64,
            dirty=False,
        ),
    )
    manifest_path = rolling_sample_run.export_flow_belief_rolling_sample_run(
        **_run_kwargs(tmp_path),
        level_processor=_fake_level_processor,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["format_id"] == "flow_belief_rolling_sample_run_v1"
    assert manifest["eligible"] is True
    assert manifest["level_seeds"] == {"L1": 1180, "L2": 1199, "L3": 1193}
    assert manifest["level_manifests"] == {
        "L1": "L1/seed_001180/manifest.json",
        "L2": "L2/seed_001199/manifest.json",
        "L3": "L3/seed_001193/manifest.json",
    }
    assert manifest["window_counts"] == {"L1": 2, "L2": 2, "L3": 2}

    with pytest.raises(FileExistsError):
        rolling_sample_run.export_flow_belief_rolling_sample_run(
            **_run_kwargs(tmp_path),
            level_processor=_fake_level_processor,
        )


def test_rolling_sample_run_cleans_staging_after_level_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_formal_artifacts()
    from latency_meta_mdp.belief.flow import rolling_sample_run

    monkeypatch.setattr(
        rolling_sample_run,
        "collect_implementation_provenance",
        lambda project_root: ImplementationProvenance(
            revision="a" * 40,
            source_sha256="b" * 64,
            dirty=False,
        ),
    )

    def fail_on_l2(*, level: int, scene_seed: int, output_dir: Path, **kwargs) -> Path:
        if level == 2:
            raise RuntimeError("injected rolling failure")
        return _fake_level_processor(
            level=level,
            scene_seed=scene_seed,
            output_dir=output_dir,
            **kwargs,
        )

    with pytest.raises(RuntimeError, match="injected rolling failure"):
        rolling_sample_run.export_flow_belief_rolling_sample_run(
            **_run_kwargs(tmp_path),
            level_processor=fail_on_l2,
        )
    assert not list(tmp_path.glob(".*.building-*"))
    assert not (tmp_path / "rolling").exists()


def test_rolling_sample_run_marks_dirty_implementation_ineligible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_formal_artifacts()
    from latency_meta_mdp.belief.flow import rolling_sample_run

    monkeypatch.setattr(
        rolling_sample_run,
        "collect_implementation_provenance",
        lambda project_root: ImplementationProvenance(
            revision="a" * 40,
            source_sha256="b" * 64,
            dirty=True,
        ),
    )
    manifest_path = rolling_sample_run.export_flow_belief_rolling_sample_run(
        **_run_kwargs(tmp_path),
        level_processor=_fake_level_processor,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["eligible"] is False
    assert manifest["blockers"] == ["dirty_implementation"]


def test_rolling_sample_run_rejects_duplicate_levels(tmp_path: Path) -> None:
    _require_formal_artifacts()
    from latency_meta_mdp.belief.flow.rolling_sample_run import (
        export_flow_belief_rolling_sample_run,
    )

    kwargs = _run_kwargs(tmp_path)
    kwargs["level_seeds"] = ((1, 1180), (1, 1181))
    with pytest.raises(ValueError, match="sorted unique levels"):
        export_flow_belief_rolling_sample_run(
            **kwargs,
            level_processor=_fake_level_processor,
        )


def test_rolling_sample_cli_exposes_level_seed_mapping() -> None:
    from latency_meta_mdp.cli.export_flow_belief_rolling_samples import build_parser

    help_text = build_parser().format_help()
    assert "--level-seed" in help_text
    assert "--rolling-config" in help_text
    assert "--output-dir" in help_text
    assert "--device" in help_text
