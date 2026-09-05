from __future__ import annotations

import json
from pathlib import Path


def test_console_progress_cadence_and_message_are_live_and_informative() -> None:
    """Catches silent multi-minute epochs or progress lines missing job identity."""

    from latency_meta_mdp.belief.action_conditioned_jepa.fold_runner import (
        format_temporal_training_progress,
        should_emit_temporal_training_progress,
    )

    emitted = [
        step
        for step in range(1, 132)
        if should_emit_temporal_training_progress(
            epoch_step=step,
            optimizer_steps_per_epoch=131,
            period=5,
        )
    ]
    assert emitted == [1, *range(5, 131, 5), 131]
    message = format_temporal_training_progress(
        temporal_config_id="stride4_80ms_history_160ms",
        fold_index=2,
        model_seed=7,
        epoch=3,
        max_epochs=50,
        epoch_step=5,
        optimizer_steps_per_epoch=131,
        global_optimizer_step=267,
        total_optimizer_steps=6_550,
        examples_seen=68_352,
        total_loss=0.125,
        gradient_norm=1.5,
        learning_rate=5e-4,
        weight_decay=2e-7,
        elapsed_seconds=42.25,
    )
    assert message == (
        "[action-conditioned-jepa] config=stride4_80ms_history_160ms "
        "fold=2 seed=7 epoch=3/50 epoch_step=5/131 "
        "global_step=267/6550 examples=68352 loss=0.125000 "
        "grad_norm=1.500000 lr=0.0005 wd=2e-07 elapsed=42.2s"
    )


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
