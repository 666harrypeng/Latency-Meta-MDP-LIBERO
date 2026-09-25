from pathlib import Path

import pytest
import yaml


def test_task_forecast_job_uses_clean_bundle_without_legacy_split_manifest(tmp_path):
    from latency_meta_mdp.data.forecast.assets import download_forecast_models, load_forecast_job

    root = Path.cwd()
    job = dict(
        schema_version=2,
        task_id="conveyor_sort",
        clean_job=str(root / "configs/experiments/conveyor_sort/clean_fsdp.yaml"),
        predictor_dir=str(tmp_path / "predictor"),
        decoder_dir=str(tmp_path / "decoder"),
        terminal_dir=str(tmp_path / "terminal"),
        conditioned_source_epochs=15,
    )
    for folder in ("predictor", "decoder", "terminal"):
        (tmp_path / folder).mkdir()
    path = tmp_path / "job.yaml"
    path.write_text(yaml.safe_dump(job))
    loaded = load_forecast_job(path, project_root=root)
    assert loaded["clean"]["task_id"] == "conveyor_sort"
    assert "split_manifest" not in loaded
    assert download_forecast_models(loaded, tmp_path / "work") == (
        tmp_path / "predictor",
        tmp_path / "decoder",
    )


def test_cache_must_match_job_model_versions_even_for_same_source(tmp_path, monkeypatch):
    from latency_meta_mdp.data.forecast import assets

    monkeypatch.setattr(assets, "download_forecast_models", lambda *_: (tmp_path, tmp_path))
    monkeypatch.setattr(
        assets,
        "forecast_bindings",
        lambda *_: {
            "direct_checkpoint_sha256": "a" * 64,
            "normalization_sha256": "b" * 64,
            "decoder_sha256": "c" * 64,
        },
    )
    binding = dict(
        predictor_sha256="a" * 64, jepa_normalization_sha256="b" * 64, decoder_sha256="c" * 64
    )
    assets.verify_forecast_models({}, binding, tmp_path)
    with pytest.raises(ValueError, match="pinned"):
        assets.verify_forecast_models({}, {**binding, "predictor_sha256": "d" * 64}, tmp_path)
