from __future__ import annotations

from pathlib import Path


def test_wandb_config_and_run_identity_are_deterministic_and_secret_free(monkeypatch) -> None:
    """Catches unstable resume IDs or credentials leaking into logged config."""

    from latency_meta_mdp.belief.action_conditioned_jepa.tracking import (
        build_wandb_run_spec,
        credential_environment_status,
        load_jepa_wandb_config,
    )

    config = load_jepa_wandb_config(
        Path("configs/training/action_conditioned_jepa/wandb.yaml")
    )
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
