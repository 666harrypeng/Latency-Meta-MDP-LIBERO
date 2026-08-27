from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from latency_meta_mdp.belief.flow.quality_config import (
    load_flow_belief_quality_sample_config,
)
from latency_meta_mdp.belief.flow.quality_types import (
    QualityContextIdentity,
    QualitySampleBundle,
    QualitySelection,
)
from latency_meta_mdp.vision_probe_data import ProbeSplit

_CONFIG_PATH = Path("configs/analysis/flow_belief_quality_samples_v1.yaml")


def _selection(offset: int = 0) -> QualitySelection:
    return QualitySelection(
        identity=QualityContextIdentity(
            level=1,
            episode_id="l1-seed-001180-attempt-000",
            scene_seed=1180,
            validation_offset=offset,
            source_tick=12 + offset,
        ),
        roles=("typical",),
        source_phase="approach",
        ranking_score=0.25,
        short_delay_error=0.1,
        long_delay_error=0.3,
        handoff_distance_ticks=8,
        motion_transition_score=0.02,
        object_position_rmse_m=0.001,
        robot_qpos_rmse_rad=0.003,
    )


def test_quality_config_locks_formal_sample_protocol() -> None:
    config = load_flow_belief_quality_sample_config(_CONFIG_PATH)

    assert config.quality_id == "flow_belief_quality_samples_v1"
    assert config.evaluation_split is ProbeSplit.VALIDATION
    assert config.display_delay_ticks == (1, 5, 10, 15, 20)
    assert config.sample_count == 32
    assert config.solver == "heun"
    assert config.solver_step_count == 16
    assert config.case_quantiles == (0.5, 0.9, 0.95)
    assert config.summary_parity_atol == 1e-6
    assert config.summary_parity_rtol == 1e-5
    assert config.selection_roles == (
        "typical",
        "p90_hard",
        "p95_hard",
        "pre_handoff",
        "handoff_adjacent",
        "motion_transition",
        "short_delay_hard",
        "long_delay_hard",
    )


def test_quality_config_rejects_nonformal_or_duplicate_delays() -> None:
    config = load_flow_belief_quality_sample_config(_CONFIG_PATH)

    with pytest.raises(ValueError, match="display delay"):
        replace(config, display_delay_ticks=(1, 5, 5, 20))
    with pytest.raises(ValueError, match="validation"):
        replace(config, evaluation_split=ProbeSplit.TRAIN)
    with pytest.raises(ValueError, match="selection roles"):
        replace(config, selection_roles=tuple(reversed(config.selection_roles)))


def test_quality_sample_bundle_locks_shapes_and_copies_inputs() -> None:
    samples = np.zeros((1, 5, 32, 22), dtype=np.float32)
    targets = np.zeros((1, 5, 22), dtype=np.float32)
    probabilities = np.asarray([[0.01, 0.12, 0.2, 0.06, 0.001]], dtype=np.float64)
    modes = np.zeros((1, 5), dtype=np.int8)
    absorbing = np.zeros((1, 5), dtype=np.bool_)

    bundle = QualitySampleBundle(
        selections=(_selection(),),
        display_delay_ticks=np.asarray([1, 5, 10, 15, 20], dtype=np.int64),
        latency_probabilities=probabilities,
        normalized_samples=samples,
        normalized_targets=targets,
        physical_samples=samples,
        physical_targets=targets,
        interaction_mode=modes,
        absorbing=absorbing,
    )
    samples[0, 0, 0, 0] = 1.0

    assert bundle.normalized_samples.shape == (1, 5, 32, 22)
    assert bundle.normalized_samples[0, 0, 0, 0] == 0.0
    assert bundle.normalized_samples.flags.writeable is False
    assert bundle.latency_probabilities.flags.writeable is False


def test_quality_sample_bundle_rejects_wrong_identity_order_and_nonfinite_values() -> None:
    kwargs = {
        "display_delay_ticks": np.asarray([1, 5, 10, 15, 20], dtype=np.int64),
        "latency_probabilities": np.full((2, 5), 0.1, dtype=np.float64),
        "normalized_samples": np.zeros((2, 5, 32, 22), dtype=np.float32),
        "normalized_targets": np.zeros((2, 5, 22), dtype=np.float32),
        "physical_samples": np.zeros((2, 5, 32, 22), dtype=np.float32),
        "physical_targets": np.zeros((2, 5, 22), dtype=np.float32),
        "interaction_mode": np.zeros((2, 5), dtype=np.int8),
        "absorbing": np.zeros((2, 5), dtype=np.bool_),
    }
    with pytest.raises(ValueError, match="sorted unique"):
        QualitySampleBundle(selections=(_selection(1), _selection(0)), **kwargs)

    kwargs["normalized_samples"][0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        QualitySampleBundle(selections=(_selection(0), _selection(1)), **kwargs)
