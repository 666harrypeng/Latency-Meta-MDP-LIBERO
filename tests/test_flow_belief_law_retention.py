from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from latency_meta_mdp.legacy.belief.flow.law_retention import (
    evaluate_law_reconstruction,
    fit_linear_law_readout,
    law_invariant_tokens,
    select_evenly_spaced_offsets,
)
from latency_meta_mdp.legacy.belief.flow.law_retention_config import (
    load_law_retention_probe_config,
)


def test_law_retention_config_locks_lightweight_probe_contract() -> None:
    config = load_law_retention_probe_config(
        Path("configs/legacy/analysis/dinov3_flow_belief_law_retention_probe_v1.yaml")
    )

    assert config.probe_id == "dinov3_flow_belief_law_retention_probe_v1"
    assert config.training_context_limit == 1024
    assert config.validation_context_limit == 512
    assert config.encoder_batch_size == 8
    assert config.readout_batch_size == 256
    assert config.effective_mean_mae_ms_max == 5.0
    assert config.baseline_relative_improvement_min == 0.50
    assert config.ordering_accuracy_min == 0.95


def test_even_context_selection_covers_the_full_split() -> None:
    selected = select_evenly_spaced_offsets(total=100, limit=8)

    assert selected == (0, 14, 28, 42, 57, 71, 85, 99)
    assert select_evenly_spaced_offsets(total=4, limit=8) == (0, 1, 2, 3)


def test_law_invariant_baseline_removes_only_counterfactual_variation() -> None:
    tokens = np.arange(3 * 5 * 2 * 4, dtype=np.float32).reshape(3, 5, 2, 4)
    invariant = law_invariant_tokens(tokens)

    assert invariant.shape == tokens.shape
    np.testing.assert_allclose(invariant[:, 0], tokens.mean(axis=1))
    np.testing.assert_allclose(invariant[:, 4], tokens.mean(axis=1))
    assert not np.array_equal(invariant[0], tokens[0])


def test_perfect_law_reconstruction_has_zero_error_and_correct_ordering() -> None:
    variants = ("nominal", "in_family", "shifted_fast", "shifted_slow", "shifted_wide")
    ticks = np.arange(1, 21, dtype=np.float64) * 0.02
    centers = (0.12, 0.13, 0.08, 0.18, 0.12)
    targets = []
    for _ in range(4):
        rows = []
        for center in centers:
            probability = np.exp(-0.5 * np.square((ticks - center) / 0.025))
            rows.append(probability / probability.sum())
        targets.append(rows)
    target = np.asarray(targets, dtype=np.float64)

    metrics = evaluate_law_reconstruction(
        predicted=target,
        target=target,
        variant_names=variants,
    )

    assert metrics["jensen_shannon_divergence_mean"] == 0.0
    assert metrics["effective_mean_mae_ms"] == 0.0
    assert metrics["region_mass_mae"] == 0.0
    assert metrics["fast_nominal_slow_ordering_accuracy"] == 1.0


def test_linear_readout_recovers_explicit_law_signal_on_held_out_contexts() -> None:
    rng = np.random.default_rng(17)
    ticks = np.arange(1, 21, dtype=np.float32) * 0.02
    laws = []
    for center in (0.12, 0.13, 0.08, 0.18, 0.12):
        probability = np.exp(-0.5 * np.square((ticks - center) / 0.025))
        laws.append(probability / probability.sum())
    targets = np.asarray(laws, dtype=np.float32)

    def make(count: int) -> tuple[np.ndarray, np.ndarray]:
        target = np.broadcast_to(targets, (count, 5, 20)).copy()
        tokens = rng.normal(scale=0.02, size=(count, 5, 1, 24)).astype(np.float32)
        tokens[..., :20] += np.log(target[:, :, None, :] + 1e-6)
        return tokens, target

    train_tokens, train_targets = make(96)
    validation_tokens, validation_targets = make(24)
    config = replace(
        load_law_retention_probe_config(
            Path("configs/legacy/analysis/dinov3_flow_belief_law_retention_probe_v1.yaml")
        ),
        readout_batch_size=64,
        readout_max_epochs=120,
        readout_patience=20,
        learning_rate=0.03,
    )

    fit = fit_linear_law_readout(
        training_tokens=train_tokens,
        training_targets=train_targets,
        validation_tokens=validation_tokens,
        validation_targets=validation_targets,
        config=config,
        device="cpu",
        seed=23,
    )
    metrics = evaluate_law_reconstruction(
        predicted=fit.validation_probabilities,
        target=validation_targets,
        variant_names=(
            "nominal",
            "in_family",
            "shifted_fast",
            "shifted_slow",
            "shifted_wide",
        ),
    )

    assert fit.best_epoch > 0
    assert metrics["effective_mean_mae_ms"] < 2.0
    assert metrics["fast_nominal_slow_ordering_accuracy"] == 1.0
