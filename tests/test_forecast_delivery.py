import json
from pathlib import Path
from types import SimpleNamespace


def test_pipeline_runs_stages_in_order_and_skips_completed_work(tmp_path, monkeypatch):
    import sys

    from latency_meta_mdp.cli import run_policy_sft_pipeline as cli

    config = Path("configs/training/pi05/l2_clean_conditioned.yaml").resolve()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pipeline",
            "--config",
            str(config),
            "--work-dir",
            str(tmp_path),
            "--preparation-python",
            "prep-python",
        ],
    )
    calls = []
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda args, **kw: calls.append(args) or SimpleNamespace(returncode=0),
    )
    cli.main()
    assert len(calls) == 4
    assert "--check-access" in calls[0]
    assert "latency_meta_mdp.cli.train_structured_pi05" in calls[1]
    assert "latency_meta_mdp.cli.prepare_forecast_sft" in calls[2]
    assert "latency_meta_mdp.cli.train_forecast_pi05" in calls[3]
    assert json.loads((tmp_path / "pipeline.json").read_text())["phase"] == "complete"
    calls.clear()
    cli.main()
    assert len(calls) == 1 and "--check-access" in calls[0]


def test_source_materialization_excludes_hf_metadata_and_preserves_declared_files(
    tmp_path, monkeypatch
):
    import huggingface_hub

    from latency_meta_mdp.forecast_delivery import stage_source

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "data.bin").write_bytes(b"1234")
    (snapshot / "manifest.json").write_text(json.dumps({"artifacts": {"data.bin": {"bytes": 4}}}))
    (snapshot / ".gitattributes").write_text("not part of source")
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda *a, **kw: str(snapshot / "manifest.json")
    )
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **kw: str(snapshot))
    target = stage_source(
        {"source_repo": "owner/source", "source_revision": "a" * 40}, tmp_path / "source"
    )
    assert {p.name for p in target.iterdir()} == {"manifest.json", "data.bin"}
    assert not (target / "data.bin").is_symlink()
    assert (target / "data.bin").read_bytes() == b"1234"


def test_scoped_vision_join_requires_every_requested_episode(monkeypatch, tmp_path):
    import pytest

    from latency_meta_mdp.belief.action_conditioned_jepa import data_adapter as data
    from latency_meta_mdp.belief.action_conditioned_jepa.config import (
        load_action_conditioned_jepa_config,
    )

    config = load_action_conditioned_jepa_config(
        model_path=Path("configs/belief/action_conditioned_jepa/model.yaml"),
        level_path=Path("configs/belief/action_conditioned_jepa/l2.yaml"),
        temporal_sampling_path=Path(
            "configs/belief/action_conditioned_jepa/stride4_80ms_history_160ms.yaml"
        ),
    )
    source = SimpleNamespace(
        root=tmp_path,
        manifest=SimpleNamespace(corpus_id="source"),
        episode_ids=lambda level: (f"L{level}",),
    )
    cache = {
        "eligible": True,
        "blockers": [],
        "episodes": [{"episode_id": "L2"}],
        "source_corpus_id": "source",
    }
    monkeypatch.setattr(data, "load_verified_source_corpus", lambda *a, **kw: source)
    monkeypatch.setattr(data, "load_verified_source_split", lambda *a, **kw: SimpleNamespace())
    monkeypatch.setattr(data, "load_verified_vision_feature_cache_run", lambda *a, **kw: cache)
    monkeypatch.setattr(data, "_hash_file", lambda p: "a" * 64)
    kwargs = dict(
        source_root=tmp_path,
        cache_run_manifest=tmp_path / "manifest.json",
        split_manifest_path=tmp_path / "split.json",
        config=config,
    )
    assert set(
        data.load_verified_jepa_inputs(**kwargs, required_episode_ids=("L2",)).cache_rows_by_episode
    ) == {"L2"}
    with pytest.raises(ValueError, match="incomplete"):
        data.load_verified_jepa_inputs(**kwargs)
    with pytest.raises(ValueError, match="incomplete"):
        data.load_verified_jepa_inputs(**kwargs, required_episode_ids=("L2", "L3"))
    with pytest.raises(ValueError, match="not part"):
        data.load_verified_jepa_inputs(**kwargs, required_episode_ids=("unknown",))
