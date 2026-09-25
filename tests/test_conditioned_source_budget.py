from dataclasses import dataclass, field

import pytest


@dataclass(frozen=True)
class Schedule:
    warmup_steps: int = 10
    decay_steps: int = 20


@dataclass(frozen=True)
class Config:
    batch_size: int = 384
    num_train_steps: int = 40630
    keep_period: int = 20315
    save_interval: int = 250
    lr_schedule: Schedule = field(default_factory=Schedule)
    policy_metadata: dict = field(default_factory=lambda: {"balanced_pair_epochs": 2})


def test_conditioned_budget_means_source_epochs_not_twenty_query_expansion():
    from latency_meta_mdp.policy.schedule import apply_conditioned_source_budget

    config = apply_conditioned_source_budget(Config(), source_count=390041, source_epochs=15)
    assert config.num_train_steps == 15237
    assert config.keep_period == 5079
    assert config.policy_metadata["checkpoint_steps"] == [5079, 10158, 15237]
    assert config.lr_schedule.decay_steps == 15237
    assert config.lr_schedule.warmup_steps == 762
    assert config.policy_metadata["training_examples"] == 5851008
    assert "balanced_pair_epochs" not in config.policy_metadata


@pytest.mark.parametrize("epochs", [0, -1, float("nan"), True])
def test_conditioned_source_budget_rejects_invalid_epochs(epochs):
    from latency_meta_mdp.policy.schedule import apply_conditioned_source_budget

    with pytest.raises(ValueError):
        apply_conditioned_source_budget(Config(), source_count=390041, source_epochs=epochs)


def test_conditioned_cli_supports_real_data_check_without_training_topology(monkeypatch):
    from latency_meta_mdp.policy import train_conditioned

    calls = []
    monkeypatch.setattr(train_conditioned, "run", calls.append)
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_conditioned",
            "--config",
            "job.yaml",
            "--output-dir",
            "out",
            "--mode",
            "smoke",
            "--check-data-only",
        ],
    )
    train_conditioned.main()
    assert calls[0].mode == "smoke" and calls[0].check_data_only
