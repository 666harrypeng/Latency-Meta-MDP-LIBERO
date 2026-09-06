from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

_CONFIG_STRIDES = {
    "dense_20ms_history_100ms": 1,
    "stride2_40ms_history_120ms": 2,
    "stride4_80ms_history_160ms": 4,
    "stride5_100ms_history_200ms": 5,
}

_ASSIGNED_DELAYS = {
    1: list(range(1, 21)),
    2: [2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14, 16, 16, 18, 18, 20, 20],
    4: [4, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16, 16, 20, 20, 20],
    5: [5, 5, 5, 5, 5, 5, 5, 10, 10, 10, 10, 10, 15, 15, 15, 15, 15, 20, 20, 20],
}


def _write_fold_manifest(root: Path, *, config_id: str, fold: int) -> None:
    stride = _CONFIG_STRIDES[config_id]
    native_ticks = list(range(stride, 21, stride))
    scalar = float(fold + 1)
    native_curve = [scalar + tick / 100.0 for tick in native_ticks]
    dense_curve = [scalar + tick / 100.0 for tick in range(1, 21)]
    source_count = 10 ** (fold + 1)
    payload = {
        "schema_version": 1,
        "format_id": "action_conditioned_jepa_fold_run_v1",
        "qualification_only": False,
        "preflight": {
            "development_context_count": source_count,
            "development_episode_count": 80,
            "device": "cuda:0",
            "fit_context_count": 1000 + fold,
            "fit_episode_count": 240,
            "fold_index": fold,
            "gradient_accumulation_steps": 8,
            "logical_global_batch_size": 256,
            "max_epochs": 50,
            "microbatch_size": 32,
            "model_seed": 7,
            "num_workers": 4,
            "optimizer_steps_per_epoch": 131,
            "output_dir": f"/remote/{config_id}/fold-{fold}/seed-7",
            "temporal_config_id": config_id,
            "total_optimizer_steps": 6550,
            "wandb_api_key": "SET",
            "wandb_run_id": f"run-{config_id}-{fold}",
        },
        "progress": {
            "completed_epochs": 50,
            "examples_seen": 1676800,
            "fold_index": fold,
            "model_seed": 7,
            "optimizer_steps": 6550,
            "temporal_config_id": config_id,
        },
        "final_development": {
            "native": {
                "native_delay_ticks": native_ticks,
                "per_anchor_latent_predictive_skill": native_curve,
                "per_anchor_latent_rmse": native_curve,
                "per_anchor_persistence_latent_rmse": native_curve,
                "per_anchor_persistence_proprio_rmse": native_curve,
                "per_anchor_proprio_rmse": native_curve,
                "source_count": source_count,
                "source_mean_latent_rmse": scalar,
                "source_mean_proprio_rmse": scalar + 10.0,
            },
            "deployed_d20": {
                "assigned_native_delay_ticks": _ASSIGNED_DELAYS[stride],
                "dense_delay_ticks": list(range(1, 21)),
                "per_delay_assigned_model_latent_rmse": dense_curve,
                "per_delay_assigned_model_proprio_rmse": dense_curve,
                "per_delay_deployed_latent_rmse": dense_curve,
                "per_delay_deployed_proprio_rmse": dense_curve,
                "per_delay_temporal_quantization_latent_rmse": dense_curve,
                "per_delay_temporal_quantization_proprio_rmse": dense_curve,
                "source_count": source_count,
                "source_mean_assigned_model_latent_rmse": scalar + 20.0,
                "source_mean_assigned_model_proprio_rmse": scalar + 30.0,
                "source_mean_deployed_latent_rmse": scalar + 40.0,
                "source_mean_deployed_proprio_rmse": scalar + 50.0,
                "source_mean_temporal_quantization_latent_rmse": scalar + 60.0,
                "source_mean_temporal_quantization_proprio_rmse": scalar + 70.0,
            },
        },
        "wall_seconds": 100.0 + fold,
        "input_sha256": {
            "cache_manifest": "cache",
            "model_config": "model",
            "selection_manifest": "selection",
            "source_manifest": "source",
            "split_manifest": "split",
            "temporal_config": f"temporal-{config_id}",
            "training_config": "training",
        },
        "history": [],
    }
    target = root / config_id / f"fold-{fold}" / "seed-7" / "manifest.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(payload), encoding="utf-8")


def _write_complete_selection(root: Path) -> None:
    for config_id in _CONFIG_STRIDES:
        for fold in range(4):
            _write_fold_manifest(root, config_id=config_id, fold=fold)


def test_report_aggregates_each_fold_equally_instead_of_weighting_source_count(
    tmp_path: Path,
) -> None:
    """Catches silently giving folds with more launch contexts greater selection weight."""

    from latency_meta_mdp.belief.action_conditioned_jepa.selection_report import (
        build_temporal_selection_report,
    )

    _write_complete_selection(tmp_path)
    report = build_temporal_selection_report(tmp_path, model_seed=7)

    dense = report["configurations"]["dense_20ms_history_100ms"]
    native_latent = dense["scalar_metrics"]["native_latent_rmse"]
    assert native_latent["fold_values"] == [1.0, 2.0, 3.0, 4.0]
    assert native_latent["mean"] == 2.5
    assert native_latent["sample_std"] == pytest.approx(math.sqrt(5.0 / 3.0))
    assert dense["development_source_counts"] == [10, 100, 1000, 10000]
    assert "score" not in dense
    assert "ranking" not in report


def test_report_writes_common_d20_and_native_anchor_curve_tables(tmp_path: Path) -> None:
    """Catches comparing configs on different implicit time axes or dropping native anchors."""

    from latency_meta_mdp.belief.action_conditioned_jepa.selection_report import (
        write_temporal_selection_report,
    )

    source = tmp_path / "runs"
    output = tmp_path / "report"
    _write_complete_selection(source)
    paths = write_temporal_selection_report(source, output, model_seed=7)

    assert paths.summary_json == output / "summary.json"
    assert paths.deployed_d20_csv == output / "deployed_d20.csv"
    assert paths.native_anchors_csv == output / "native_anchors.csv"
    summary = json.loads(paths.summary_json.read_text(encoding="utf-8"))
    assert summary["format_id"] == "action_conditioned_jepa_temporal_selection_report_v1"
    with paths.deployed_d20_csv.open(newline="", encoding="utf-8") as handle:
        deployed = list(csv.DictReader(handle))
    with paths.native_anchors_csv.open(newline="", encoding="utf-8") as handle:
        native = list(csv.DictReader(handle))

    assert len(deployed) == 80
    assert deployed[0]["config_id"] == "dense_20ms_history_100ms"
    assert deployed[0]["delay_tick"] == "1"
    assert deployed[0]["delay_ms"] == "20"
    assert float(deployed[0]["deployed_latent_rmse_mean"]) == 2.51
    assert float(deployed[0]["deployed_latent_rmse_sample_std"]) == pytest.approx(
        math.sqrt(5.0 / 3.0)
    )
    assert len(native) == 39
    stride4 = [row for row in native if row["config_id"] == "stride4_80ms_history_160ms"]
    assert [int(row["delay_tick"]) for row in stride4] == [4, 8, 12, 16, 20]


def test_report_rejects_an_incomplete_fold_inventory(tmp_path: Path) -> None:
    """Catches publishing a comparison after one config silently loses a development fold."""

    from latency_meta_mdp.belief.action_conditioned_jepa.selection_report import (
        build_temporal_selection_report,
    )

    _write_complete_selection(tmp_path)
    missing = (
        tmp_path
        / "stride5_100ms_history_200ms"
        / "fold-3"
        / "seed-7"
        / "manifest.json"
    )
    missing.unlink()

    with pytest.raises(ValueError, match="complete four-fold inventory"):
        build_temporal_selection_report(tmp_path, model_seed=7)


def test_report_rejects_noncomparable_input_provenance(tmp_path: Path) -> None:
    """Catches aggregating a candidate trained against a different source or split."""

    from latency_meta_mdp.belief.action_conditioned_jepa.selection_report import (
        build_temporal_selection_report,
    )

    _write_complete_selection(tmp_path)
    target = (
        tmp_path
        / "stride4_80ms_history_160ms"
        / "fold-2"
        / "seed-7"
        / "manifest.json"
    )
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["input_sha256"]["source_manifest"] = "different-source"
    target.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="common input provenance"):
        build_temporal_selection_report(tmp_path, model_seed=7)


def test_selection_report_cli_materializes_the_canonical_outputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Catches a CLI that writes ad-hoc filenames or hides the summary location."""

    from latency_meta_mdp.cli.summarize_action_conditioned_jepa_selection import main

    source = tmp_path / "runs"
    output = tmp_path / "report"
    _write_complete_selection(source)

    assert main(["--selection-root", str(source), "--output-dir", str(output)]) == 0
    assert capsys.readouterr().out.strip() == str(output / "summary.json")
    assert sorted(path.name for path in output.iterdir()) == [
        "deployed_d20.csv",
        "native_anchors.csv",
        "summary.json",
    ]
