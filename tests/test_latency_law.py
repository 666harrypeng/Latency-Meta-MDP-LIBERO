from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np

from latency_meta_mdp.runtime.latency_harness import LogicalLatencyHarness
from latency_meta_mdp.runtime.latency_law import CategoricalDelaySampler, load_latency_law


def _law():
    return load_latency_law(Path("configs/runtime/latency/truncated_beta_5_26_400ms_v1.yaml"))


def _earlier_law():
    return load_latency_law(Path("configs/runtime/latency/truncated_beta_8_65_400ms_v1.yaml"))


def test_truncated_beta_law_builds_twenty_normalized_grid_probabilities() -> None:
    law = _law()

    assert law.law_id == "truncated_beta_5_26_400ms_v1"
    assert law.shape_alpha == 5.0
    assert law.shape_beta == 26.0
    assert law.bin_count == 20
    assert law.delay_ticks == tuple(range(1, 21))
    assert law.probabilities.shape == (20,)
    assert law.probabilities.flags.writeable is False
    assert np.all(law.probabilities > 0.0)
    np.testing.assert_allclose(law.probabilities.sum(), 1.0, atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        law.continuous_mode_seconds,
        0.13793103448275862,
        atol=1e-15,
        rtol=0,
    )


def test_truncated_beta_law_matches_locked_latency_regions() -> None:
    law = _law()

    np.testing.assert_allclose(law.continuous_mean_seconds, 0.1609, atol=1e-4)
    np.testing.assert_allclose(law.effective_mean_seconds, 0.1709, atol=1e-4)
    np.testing.assert_allclose(
        law.region_probabilities,
        (0.176, 0.570, 0.225, 0.029),
        atol=1e-3,
    )
    assert sum(law.probabilities[5:10]) > 0.55
    assert law.probabilities[6] == max(law.probabilities)


def test_beta_8_65_law_matches_locked_earlier_latency_regime() -> None:
    law = _earlier_law()

    assert law.law_id == "truncated_beta_8_65_400ms_v1"
    assert law.shape_alpha == 8.0
    assert law.shape_beta == 65.0
    np.testing.assert_allclose(law.effective_mean_seconds, 0.1196, atol=1e-4)
    assert sum(law.probabilities[3:8]) > 0.84
    assert 0.03 < sum(law.probabilities[9:]) < 0.05
    assert law.probabilities[4] == max(law.probabilities)


def test_categorical_sampler_is_seeded_and_never_emits_zero_or_overflow() -> None:
    law = _law()
    first = CategoricalDelaySampler(law=law, seed=123)
    repeat = CategoricalDelaySampler(law=law, seed=123)

    first_samples = [first() for _ in range(1_000)]
    repeat_samples = [repeat() for _ in range(1_000)]

    assert first_samples == repeat_samples
    assert min(first_samples) >= 1
    assert max(first_samples) <= 20
    assert len(set(first_samples)) > 10


def test_harness_exposes_law_but_hides_this_request_realized_delay() -> None:
    law = _law()
    sampler = CategoricalDelaySampler(law=law, seed=9)
    seen_context_fields: set[str] = set()
    seen_law = None

    def infer(context):
        nonlocal seen_context_fields, seen_law
        seen_context_fields = {field.name for field in fields(context)}
        seen_law = context.observation["latency_law"]
        return "payload"

    harness = LogicalLatencyHarness(
        formal_tick_us=20_000,
        delay_sampler=sampler,
        simulation_time_reader=lambda: 0,
        monotonic_ns=iter((1, 2)).__next__,
    )
    harness.open_boundary(0)
    harness.launch(
        observation={"latency_law": law.probabilities},
        infer=infer,
    )

    np.testing.assert_array_equal(seen_law, law.probabilities)
    assert "realized_delay_ticks" not in seen_context_fields
