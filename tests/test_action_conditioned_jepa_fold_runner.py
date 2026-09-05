from __future__ import annotations

import json
from pathlib import Path


def test_fold_runner_preflight_resolves_job_without_loading_or_writing_data(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    """Catches starting an expensive fold before job identity and credentials are reviewable."""

    from latency_meta_mdp.cli.train_action_conditioned_jepa import main

    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    output = tmp_path / "run"
    status = main(
        [
            "--project-root",
            ".",
            "--temporal-config",
            "configs/belief/action_conditioned_jepa/stride4_80ms_history_160ms.yaml",
            "--fold-index",
            "0",
            "--model-seed",
            "7",
            "--microbatch-size",
            "16",
            "--num-workers",
            "0",
            "--device",
            "cuda:0",
            "--output-dir",
            str(output),
            "--preflight-only",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert status == 0
    assert payload["temporal_config_id"] == "stride4_80ms_history_160ms"
    assert payload["fold_index"] == 0
    assert payload["fit_episode_count"] == 240
    assert payload["development_episode_count"] == 80
    assert payload["logical_global_batch_size"] == 256
    assert payload["microbatch_size"] == 16
    assert payload["gradient_accumulation_steps"] == 16
    assert payload["max_epochs"] == 50
    assert payload["wandb_api_key"] == "UNSET"
    assert not output.exists()
