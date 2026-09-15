from pathlib import Path

import pytest
import yaml


def test_direct_job_uses_level_specific_inputs_and_shared_budget(tmp_path):
    from latency_meta_mdp.belief.action_conditioned_jepa.direct_prediction_run import (
        load_direct_job,
    )

    config = Path("configs/training/action_conditioned_jepa/direct_query_l2.yaml")
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

    from latency_meta_mdp.belief.action_conditioned_jepa.direct_prediction_run import (
        reconcile_epoch_journal,
    )

    p = tmp_path / "epochs.jsonl"
    rows = [{"completed_epochs": i, "optimizer_steps": 175 * i} for i in (1, 2)]
    p.write_text(json.dumps(rows[0]) + "\n" + "{incomplete")
    reconcile_epoch_journal(p, completed_epochs=2, last_report=rows[1])
    assert [json.loads(s) for s in p.read_text().splitlines()] == rows
    with pytest.raises(ValueError, match="missing"):
        reconcile_epoch_journal(p, completed_epochs=4, last_report={"completed_epochs": 4})
