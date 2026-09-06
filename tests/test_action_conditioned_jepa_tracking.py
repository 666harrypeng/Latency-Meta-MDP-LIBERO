from __future__ import annotations

from pathlib import Path


def test_wandb_config_and_run_identity_are_deterministic_and_secret_free(monkeypatch) -> None:
    """Catches unstable resume IDs or credentials leaking into logged config."""

    from latency_meta_mdp.belief.action_conditioned_jepa.tracking import (
        build_wandb_run_spec,
        credential_environment_status,
        load_jepa_wandb_config,
    )

    config = load_jepa_wandb_config(Path("configs/training/action_conditioned_jepa/wandb.yaml"))
    first = build_wandb_run_spec(
        config=config,
        selection_id="l3-trainpool-four-configs-v1",
        level=3,
        temporal_config_id="stride4_80ms_history_160ms",
        fold_index=2,
        model_seed=7,
    )
    second = build_wandb_run_spec(
        config=config,
        selection_id="l3-trainpool-four-configs-v1",
        level=3,
        temporal_config_id="stride4_80ms_history_160ms",
        fold_index=2,
        model_seed=7,
    )
    monkeypatch.setenv("WANDB_API_KEY", "wandb-secret-value")
    monkeypatch.setenv("HF_TOKEN", "hf-secret-value")
    status = credential_environment_status()

    assert first == second
    assert first.project == "latency-meta-mdp-action-conditioned-jepa"
    assert first.group == "l3-temporal-selection"
    assert first.name == "jepa-l3-stride4_80ms_history_160ms-fold2-seed7"
    assert first.resume == "allow"
    assert len(first.run_id) == 16
    assert first.logged_config == {
        "fold_index": 2,
        "level": 3,
        "model_seed": 7,
        "selection_id": "l3-trainpool-four-configs-v1",
        "temporal_config_id": "stride4_80ms_history_160ms",
    }
    assert status == {"HF_TOKEN": "SET", "WANDB_API_KEY": "SET"}
    assert "secret-value" not in repr(status)


def test_l3_admission_wandb_identity_has_no_fold_and_cannot_collide_with_selection() -> None:
    """Catches final seeds resuming or overwriting a cross-validation W&B run."""

    from latency_meta_mdp.belief.action_conditioned_jepa.tracking import (
        build_jepa_admission_wandb_run_spec,
        load_jepa_wandb_config,
    )

    config = load_jepa_wandb_config(Path("configs/training/action_conditioned_jepa/wandb.yaml"))
    first = build_jepa_admission_wandb_run_spec(
        config=config,
        stage_id="l3-stride4-final-admission-v1",
        temporal_config_id="stride4_80ms_history_160ms",
        model_seed=17,
    )
    repeated = build_jepa_admission_wandb_run_spec(
        config=config,
        stage_id="l3-stride4-final-admission-v1",
        temporal_config_id="stride4_80ms_history_160ms",
        model_seed=17,
    )

    assert first == repeated
    assert first.group == "l3-final-admission"
    assert first.name == "jepa-l3-stride4_80ms_history_160ms-admission-seed17"
    assert first.logged_config == {
        "level": 3,
        "model_seed": 17,
        "stage_id": "l3-stride4-final-admission-v1",
        "temporal_config_id": "stride4_80ms_history_160ms",
    }
    assert len(first.run_id) == 16


def test_final_runs_with_same_seed_on_different_levels_have_distinct_wandb_ids():
    from latency_meta_mdp.belief.action_conditioned_jepa.tracking import (
        build_jepa_admission_wandb_run_spec,
        load_jepa_wandb_config,
    )

    config = load_jepa_wandb_config(Path("configs/training/action_conditioned_jepa/wandb.yaml"))
    runs = [
        build_jepa_admission_wandb_run_spec(
            config=config,
            level=level,
            stage_id=f"l{level}-stride4-final-admission-v1",
            temporal_config_id="stride4_80ms_history_160ms",
            model_seed=27,
        )
        for level in (1, 2, 3)
    ]
    assert len({run.run_id for run in runs}) == 3
    assert [run.logged_config["level"] for run in runs] == [1, 2, 3]
