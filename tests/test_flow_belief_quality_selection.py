from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from latency_meta_mdp.legacy.belief.flow.quality_config import (
    FlowBeliefQualitySampleConfig,
    load_flow_belief_quality_sample_config,
)
from latency_meta_mdp.legacy.belief.flow.quality_selection import (
    FormalFlowSummary,
    score_validation_contexts,
    select_quality_contexts,
)
from latency_meta_mdp.legacy.belief.flow.quality_types import (
    QualityContextIdentity,
    QualityContextScore,
)
from latency_meta_mdp.legacy.belief.flow.training_data import FlowBeliefNormalization
from latency_meta_mdp.legacy.vision_probe_data import ProbeSplit


class _FakeCorpus:
    def __init__(self) -> None:
        probabilities = np.full(20, 0.05, dtype=np.float64)
        target_a = np.zeros((20, 22), dtype=np.float32)
        target_b = np.zeros((20, 22), dtype=np.float32)
        target_b[:, 16] = np.linspace(0.0, 0.19, 20)
        self._samples = (
            SimpleNamespace(
                episode_id="episode-a",
                level=1,
                scene_seed=1180,
                source_tick=4,
                latency_probabilities=probabilities,
                target_states=target_a,
            ),
            SimpleNamespace(
                episode_id="episode-b",
                level=1,
                scene_seed=1181,
                source_tick=6,
                latency_probabilities=probabilities,
                target_states=target_b,
            ),
        )
        episode_a = SimpleNamespace(expert_phase=np.asarray(["approach"] * 40))
        episode_b = SimpleNamespace(expert_phase=np.asarray(["close"] * 40))
        self.records = (
            SimpleNamespace(
                tail=SimpleNamespace(episode=episode_a),
                indices=(SimpleNamespace(source_tick=4),),
                handoff=np.asarray(["none"] * 10 + ["physical"] * 30),
            ),
            SimpleNamespace(
                tail=SimpleNamespace(episode=episode_b),
                indices=(SimpleNamespace(source_tick=6),),
                handoff=np.asarray(["none"] * 8 + ["physical"] * 32),
            ),
        )
        self.sample_references = {ProbeSplit.VALIDATION: ((0, 0), (1, 0))}

    def materialize(self, split: ProbeSplit, offset: int):
        assert split is ProbeSplit.VALIDATION
        return self._samples[offset]


def _normalization() -> FlowBeliefNormalization:
    return FlowBeliefNormalization(
        proprio_mean=np.zeros(16, dtype=np.float32),
        proprio_std=np.ones(16, dtype=np.float32),
        action_mean=np.zeros(7, dtype=np.float32),
        action_std=np.ones(7, dtype=np.float32),
        target_mean=np.zeros(22, dtype=np.float32),
        target_std=np.ones(22, dtype=np.float32),
    )


def _summary() -> FormalFlowSummary:
    mean = np.zeros((2, 20, 22), dtype=np.float32)
    mean[0, :, :] = 1.0
    mean[1, :, 16] = np.linspace(0.01, 0.2, 20)
    target = np.zeros_like(mean)
    target[1, :, 16] = np.linspace(0.0, 0.19, 20)
    return FormalFlowSummary(
        sample_mean_normalized=mean,
        sample_std_normalized=np.full_like(mean, 0.2),
        target_normalized=target,
        interaction_mode=np.zeros((2, 20), dtype=np.int8),
        absorbing=np.zeros((2, 20), dtype=np.bool_),
    )


def _config() -> FlowBeliefQualitySampleConfig:
    return load_flow_belief_quality_sample_config(
        Path("configs/legacy/analysis/flow_belief_quality_samples_v1.yaml")
    )


def _score(offset: int, value: float, *, phase: str = "approach") -> QualityContextScore:
    return QualityContextScore(
        identity=QualityContextIdentity(
            level=1,
            episode_id=f"episode-{offset:02d}",
            scene_seed=1180 + offset,
            validation_offset=offset,
            source_tick=10 + offset,
        ),
        source_phase=phase,
        ranking_score=value,
        short_delay_error=value / 2.0,
        long_delay_error=value,
        handoff_distance_ticks=abs(4 - offset),
        motion_transition_score=value / 3.0,
        object_position_rmse_m=value / 1000.0,
        robot_qpos_rmse_rad=value / 100.0,
    )


def test_context_scores_use_normalized_ranking_and_physical_group_errors() -> None:
    scored = score_validation_contexts(
        corpus=_FakeCorpus(),
        summary=_summary(),
        normalization=_normalization(),
    )

    assert tuple(row.identity.validation_offset for row in scored) == (0, 1)
    assert scored[0].ranking_score == pytest.approx(1.0)
    assert scored[0].short_delay_error == pytest.approx(1.0)
    assert scored[0].long_delay_error == pytest.approx(1.0)
    assert scored[0].object_position_rmse_m == pytest.approx(1.0)
    assert scored[0].robot_qpos_rmse_rad == pytest.approx(1.0)
    assert scored[0].source_phase == "approach"
    assert scored[0].handoff_distance_ticks == 6
    assert scored[1].motion_transition_score == pytest.approx(0.0, abs=1e-7)


def test_context_scoring_rejects_summary_target_misalignment() -> None:
    summary = _summary()
    target = np.array(summary.target_normalized, copy=True)
    target[0, 0, 0] = 1.0

    with pytest.raises(ValueError, match="target alignment"):
        score_validation_contexts(
            corpus=_FakeCorpus(),
            summary=replace(summary, target_normalized=target),
            normalization=_normalization(),
        )


def test_role_selection_is_deterministic_and_merges_duplicate_roles() -> None:
    scores = tuple(_score(offset, float(offset + 1)) for offset in range(8))
    selected = select_quality_contexts(scored=scores, config=_config())
    selected_by_offset = {row.identity.validation_offset: row.roles for row in selected}

    assert tuple(row.identity for row in selected) == tuple(
        sorted(row.identity for row in selected)
    )
    assert selected_by_offset[7][0] == "p95_hard"
    assert selected_by_offset[7][-1] == "long_delay_hard"
    assert "typical" in selected_by_offset[3]
    assert "handoff_adjacent" in selected_by_offset[4]
    assert "motion_transition" in selected_by_offset[7]


def test_role_selection_uses_identity_as_the_final_tie_break() -> None:
    scores = (
        _score(1, 1.0),
        _score(0, 1.0),
        _score(2, 2.0, phase="close"),
    )
    selected = select_quality_contexts(scored=scores, config=_config())
    role_to_identity = {role: row.identity for row in selected for role in row.roles}

    assert role_to_identity["typical"].validation_offset == 0
    assert role_to_identity["pre_handoff"].validation_offset == 0
