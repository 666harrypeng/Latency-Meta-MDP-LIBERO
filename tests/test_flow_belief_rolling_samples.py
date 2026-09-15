# ruff: noqa: E402

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest

pytest.importorskip("torch")

from latency_meta_mdp.legacy.belief.flow.context_sampling import (
    FlowValidationContextSamples,
)
from latency_meta_mdp.legacy.belief.flow.rolling_samples import (
    build_rolling_level_sample_bundle,
)
from latency_meta_mdp.legacy.belief.flow.rolling_selection import (
    RollingSeedSelection,
    RollingWindowIdentity,
)


def _selection() -> RollingSeedSelection:
    return RollingSeedSelection(
        level=1,
        episode_id="l1-seed-001180-attempt-000",
        scene_seed=1180,
        critical_start_tick=25,
        critical_end_tick=88,
        approach_start_tick=56,
        handoff_tick=97,
        windows=(
            RollingWindowIdentity(
                level=1,
                episode_id="l1-seed-001180-attempt-000",
                scene_seed=1180,
                validation_offset=0,
                source_tick=25,
                history_start_tick=20,
                source_phase="pregrasp",
                critical_end_tick=88,
                handoff_tick=97,
                handoff_distance_ticks=72,
            ),
            RollingWindowIdentity(
                level=1,
                episode_id="l1-seed-001180-attempt-000",
                scene_seed=1180,
                validation_offset=10,
                source_tick=68,
                history_start_tick=63,
                source_phase="approach",
                critical_end_tick=88,
                handoff_tick=97,
                handoff_distance_ticks=29,
            ),
        ),
    )


def _sampled(*, absorbing: bool = False) -> FlowValidationContextSamples:
    offsets = (0, 10)
    delays = np.asarray([1, 5, 10, 15, 20])
    physical_targets = np.zeros((2, 5, 22), dtype=np.float32)
    physical_targets[:, :, 0] = np.asarray([[26, 30, 35, 40, 45], [69, 73, 78, 83, 88]])
    return FlowValidationContextSamples(
        validation_offsets=offsets,
        delay_ticks=delays,
        normalized_samples=np.zeros((2, 5, 32, 22), dtype=np.float32),
        normalized_targets=physical_targets,
        physical_samples=np.zeros((2, 5, 32, 22), dtype=np.float32),
        physical_targets=physical_targets,
        latency_probabilities=np.full((2, 5), 0.05, dtype=np.float64),
        interaction_mode=np.zeros((2, 5), dtype=np.int8),
        absorbing=np.full((2, 5), absorbing, dtype=np.bool_),
        noise_sha256_by_offset=MappingProxyType({0: "a" * 64, 10: "b" * 64}),
    )


def _expert_phase() -> np.ndarray:
    return np.asarray(["pregrasp"] * 56 + ["approach"] * 32 + ["close"] * 9 + ["lift"] * 37)


def test_rolling_bundle_uses_boundary_phase_and_complete_window_metadata() -> None:
    bundle = build_rolling_level_sample_bundle(
        selection=_selection(),
        sampled=_sampled(),
        expert_phase=_expert_phase(),
    )

    np.testing.assert_array_equal(bundle.source_ticks, [25, 68])
    np.testing.assert_array_equal(bundle.history_start_ticks, [20, 63])
    np.testing.assert_array_equal(
        bundle.target_ticks,
        [[26, 30, 35, 40, 45], [69, 73, 78, 83, 88]],
    )
    assert bundle.target_phase[1, -1] == "approach"
    assert bundle.critical_end_tick == 88
    assert bundle.handoff_tick == 97
    assert bundle.physical_samples.shape == (2, 5, 32, 22)
    assert bundle.physical_samples.flags.writeable is False


def test_rolling_bundle_rejects_absorbing_or_misaligned_samples() -> None:
    with pytest.raises(ValueError, match="absorbing"):
        build_rolling_level_sample_bundle(
            selection=_selection(),
            sampled=_sampled(absorbing=True),
            expert_phase=_expert_phase(),
        )
    misaligned = replace(
        _sampled(),
        validation_offsets=(0, 11),
        noise_sha256_by_offset=MappingProxyType({0: "a" * 64, 11: "b" * 64}),
    )
    with pytest.raises(ValueError, match="validation offsets"):
        build_rolling_level_sample_bundle(
            selection=_selection(),
            sampled=misaligned,
            expert_phase=_expert_phase(),
        )
