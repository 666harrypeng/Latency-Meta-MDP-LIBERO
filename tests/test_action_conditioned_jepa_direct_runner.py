from pathlib import Path

import pytest
import yaml


def test_direct_job_uses_level_specific_inputs_and_shared_budget(tmp_path):
    from latency_meta_mdp.belief.jepa.job import (
        load_direct_job,
    )

    config = Path("configs/experiments/moving_ball/l2/belief.yaml")
    job = load_direct_job(config, project_root=Path.cwd())
    assert job.level == 2
    assert "l2-final-admission" in str(job.normalization)
    assert job.training.global_batch_size == job.microbatch_size * 16
    assert job.training.max_epochs == 75
    assert job.training.optimizer_steps_per_epoch == 175
    value = yaml.safe_load(config.read_text())
    value["microbatch_size"] = 17
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="microbatch"):
        load_direct_job(bad, project_root=Path.cwd())


def test_epoch_journal_reconciles_to_saved_checkpoint(tmp_path):
    import json

    from latency_meta_mdp.belief.jepa.job import (
        reconcile_epoch_journal,
    )

    p = tmp_path / "epochs.jsonl"
    rows = [{"completed_epochs": i, "optimizer_steps": 175 * i} for i in (1, 2)]
    p.write_text(json.dumps(rows[0]) + "\n" + "{incomplete")
    reconcile_epoch_journal(p, completed_epochs=2, last_report=rows[1])
    assert [json.loads(s) for s in p.read_text().splitlines()] == rows
    with pytest.raises(ValueError, match="missing"):
        reconcile_epoch_journal(p, completed_epochs=4, last_report={"completed_epochs": 4})


def test_training_entry_builds_identity_before_model_initialization(tmp_path, monkeypatch):
    import json
    import sys
    from dataclasses import asdict
    from types import SimpleNamespace

    from latency_meta_mdp.belief.jepa import job as runner
    from latency_meta_mdp.belief.jepa import train as cli
    from latency_meta_mdp.belief.jepa.optimization import (
        DirectTrainingConfig,
    )
    from latency_meta_mdp.io.artifacts import sha256_file

    norm = tmp_path / "norm.json"
    norm.write_text("{}")
    config = tmp_path / "job.yaml"
    config.write_text("{}")
    training = DirectTrainingConfig()
    job = SimpleNamespace(
        training=training,
        training_config=config,
        normalization=norm,
        level=2,
        task_id=None,
        label="L2",
        microbatch_size=16,
        device="cpu",
    )
    corpus = SimpleNamespace(
        source_manifest_sha256="a" * 64,
        cache_manifest_sha256="b" * 64,
        split_manifest_sha256="c" * 64,
    )
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "status": "preflight_passed",
                "level": 2,
                "microbatch_size": 16,
                "training_config": asdict(training),
                "normalization_sha256": sha256_file(norm),
                "source_manifest_sha256": "a" * 64,
                "cache_manifest_sha256": "b" * 64,
                "split_manifest_sha256": "c" * 64,
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--config",
            str(config),
            "--preflight",
            str(preflight),
            "--output-dir",
            str(tmp_path / "run"),
        ],
    )
    monkeypatch.setenv("WANDB_API_KEY", "test-not-a-real-key")
    monkeypatch.setattr(
        cli.subprocess, "check_output", lambda args, **kw: "" if "status" in args else "d" * 40
    )
    monkeypatch.setattr(runner, "load_direct_job", lambda *a, **kw: job)
    monkeypatch.setattr(runner, "load_direct_data", lambda *a, **kw: (None, None, corpus, None))

    class ReachedModel(Exception):
        pass

    def stop_at_model(**kwargs):
        raise ReachedModel

    monkeypatch.setattr(cli, "DirectJepaPredictor", stop_at_model)
    with pytest.raises(ReachedModel):
        cli.main()
