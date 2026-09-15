"""Deterministic validation-context scoring and quality-case selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from latency_meta_mdp.legacy.belief.common.feature_corpus import FeatureBeliefCorpus
from latency_meta_mdp.legacy.belief.flow.quality_config import FlowBeliefQualitySampleConfig
from latency_meta_mdp.legacy.belief.flow.quality_types import (
    QualityContextIdentity,
    QualityContextScore,
    QualitySelection,
)
from latency_meta_mdp.legacy.belief.flow.training_data import FlowBeliefNormalization
from latency_meta_mdp.legacy.vision_probe_data import ProbeSplit

_NO_HANDOFF_DISTANCE = 2**31 - 1


def _readonly(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    copied = np.array(value, dtype=dtype, copy=True)
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True)
class FormalFlowSummary:
    sample_mean_normalized: np.ndarray
    sample_std_normalized: np.ndarray
    target_normalized: np.ndarray
    interaction_mode: np.ndarray
    absorbing: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.sample_mean_normalized)
        context_count = mean.shape[0] if mean.ndim == 3 else -1
        shapes = {
            "sample_mean_normalized": (context_count, 20, 22),
            "sample_std_normalized": (context_count, 20, 22),
            "target_normalized": (context_count, 20, 22),
            "interaction_mode": (context_count, 20),
            "absorbing": (context_count, 20),
        }
        if context_count <= 0:
            raise ValueError("formal Flow summary requires at least one context")
        for name, shape in shapes.items():
            if np.asarray(getattr(self, name)).shape != shape:
                raise ValueError(f"formal Flow summary {name} has invalid shape")
        numeric_names = (
            "sample_mean_normalized",
            "sample_std_normalized",
            "target_normalized",
        )
        if any(not np.all(np.isfinite(getattr(self, name))) for name in numeric_names):
            raise ValueError("formal Flow summary arrays must be finite")
        if np.any(np.asarray(self.sample_std_normalized) < 0.0):
            raise ValueError("formal Flow summary standard deviation must be non-negative")
        for name, dtype in (
            ("sample_mean_normalized", np.float32),
            ("sample_std_normalized", np.float32),
            ("target_normalized", np.float32),
            ("interaction_mode", np.int8),
            ("absorbing", np.bool_),
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name), dtype=dtype))

    @property
    def context_count(self) -> int:
        return int(self.sample_mean_normalized.shape[0])


def load_formal_flow_summary(
    path: Path,
    *,
    expected_context_count: int,
) -> FormalFlowSummary:
    with np.load(path, allow_pickle=False) as source:
        required = {
            "sample_mean_normalized",
            "sample_std_normalized",
            "target_normalized",
            "interaction_mode",
            "absorbing",
        }
        if set(source.files) != required:
            raise ValueError("formal Flow summary fields are invalid")
        summary = FormalFlowSummary(
            sample_mean_normalized=source["sample_mean_normalized"],
            sample_std_normalized=source["sample_std_normalized"],
            target_normalized=source["target_normalized"],
            interaction_mode=source["interaction_mode"],
            absorbing=source["absorbing"],
        )
    if summary.context_count != expected_context_count:
        raise ValueError("formal Flow summary context count is invalid")
    return summary


def _weighted_group_rmse(
    error: np.ndarray,
    probabilities: np.ndarray,
    columns: slice,
) -> float:
    selected = error[:, columns]
    return float(np.sqrt(np.sum(probabilities[:, None] * np.square(selected)) / selected.shape[-1]))


def _motion_transition_score(target_states: np.ndarray) -> float:
    xy = np.asarray(target_states, dtype=np.float64)[:, 16:18]
    if len(xy) < 3:
        return 0.0
    return float(np.max(np.linalg.norm(np.diff(xy, n=2, axis=0), axis=1)))


def score_validation_contexts(
    *,
    corpus: FeatureBeliefCorpus,
    summary: FormalFlowSummary,
    normalization: FlowBeliefNormalization,
) -> tuple[QualityContextScore, ...]:
    references = corpus.sample_references[ProbeSplit.VALIDATION]
    if summary.context_count != len(references):
        raise ValueError("formal Flow summary and validation corpus counts disagree")
    target_mean = np.asarray(normalization.target_mean, dtype=np.float64)
    target_std = np.asarray(normalization.target_std, dtype=np.float64)
    if target_mean.shape != (22,) or target_std.shape != (22,) or np.any(target_std <= 0):
        raise ValueError("Flow quality target normalization is invalid")
    scored = []
    for offset, (record_offset, index_offset) in enumerate(references):
        sample = corpus.materialize(ProbeSplit.VALIDATION, offset)
        record = corpus.records[record_offset]
        index = record.indices[index_offset]
        normalized_target = (
            np.asarray(sample.target_states, dtype=np.float64) - target_mean
        ) / target_std
        if not np.allclose(
            normalized_target,
            summary.target_normalized[offset],
            atol=1e-5,
            rtol=1e-5,
        ):
            raise ValueError("formal Flow summary target alignment failed")
        probability = np.asarray(sample.latency_probabilities, dtype=np.float64)
        normalized_error = (
            np.asarray(summary.sample_mean_normalized[offset], dtype=np.float64) - normalized_target
        )
        physical_error = normalized_error * target_std
        ranking_score = float(
            np.sqrt(np.sum(probability[:, None] * np.square(normalized_error)) / 22.0)
        )
        short_delay_error = float(np.sqrt(np.mean(np.square(normalized_error[0]))))
        long_delay_error = float(np.sqrt(np.mean(np.square(normalized_error[-1]))))
        physical_ticks = np.flatnonzero(np.asarray(record.handoff) == "physical")
        handoff_distance = (
            int(np.min(np.abs(physical_ticks - index.source_tick)))
            if len(physical_ticks)
            else _NO_HANDOFF_DISTANCE
        )
        scored.append(
            QualityContextScore(
                identity=QualityContextIdentity(
                    level=sample.level,
                    episode_id=sample.episode_id,
                    scene_seed=sample.scene_seed,
                    validation_offset=offset,
                    source_tick=sample.source_tick,
                ),
                source_phase=str(record.tail.episode.expert_phase[index.source_tick]),
                ranking_score=ranking_score,
                short_delay_error=short_delay_error,
                long_delay_error=long_delay_error,
                handoff_distance_ticks=handoff_distance,
                motion_transition_score=_motion_transition_score(sample.target_states),
                object_position_rmse_m=_weighted_group_rmse(
                    physical_error,
                    probability,
                    slice(16, 19),
                ),
                robot_qpos_rmse_rad=_weighted_group_rmse(
                    physical_error,
                    probability,
                    slice(0, 7),
                ),
            )
        )
    return tuple(sorted(scored, key=lambda row: row.identity))


def _to_selection(score: QualityContextScore, roles: tuple[str, ...]) -> QualitySelection:
    return QualitySelection(
        identity=score.identity,
        roles=roles,
        source_phase=score.source_phase,
        ranking_score=score.ranking_score,
        short_delay_error=score.short_delay_error,
        long_delay_error=score.long_delay_error,
        handoff_distance_ticks=score.handoff_distance_ticks,
        motion_transition_score=score.motion_transition_score,
        object_position_rmse_m=score.object_position_rmse_m,
        robot_qpos_rmse_rad=score.robot_qpos_rmse_rad,
    )


def select_quality_contexts(
    *,
    scored: tuple[QualityContextScore, ...],
    config: FlowBeliefQualitySampleConfig,
) -> tuple[QualitySelection, ...]:
    ordered = tuple(sorted(scored, key=lambda row: row.identity))
    if not ordered or len({row.identity for row in ordered}) != len(ordered):
        raise ValueError("quality context scores must have unique identities")
    values = np.asarray([row.ranking_score for row in ordered], dtype=np.float64)

    def nearest_quantile(quantile: float) -> QualityContextScore:
        target = float(np.quantile(values, quantile))
        return min(ordered, key=lambda row: (abs(row.ranking_score - target), row.identity))

    pre_handoff = tuple(row for row in ordered if row.source_phase in {"pregrasp", "approach"})
    if not pre_handoff:
        raise ValueError("quality selection requires a pre-handoff context")
    role_rows = {
        "typical": nearest_quantile(0.5),
        "p90_hard": nearest_quantile(0.9),
        "p95_hard": nearest_quantile(0.95),
        "pre_handoff": sorted(
            pre_handoff,
            key=lambda row: (-row.ranking_score, row.identity),
        )[0],
        "handoff_adjacent": min(
            ordered,
            key=lambda row: (
                row.handoff_distance_ticks,
                -row.ranking_score,
                row.identity,
            ),
        ),
        "motion_transition": sorted(
            ordered,
            key=lambda row: (
                -row.motion_transition_score,
                -row.ranking_score,
                row.identity,
            ),
        )[0],
        "short_delay_hard": sorted(
            ordered,
            key=lambda row: (-row.short_delay_error, row.identity),
        )[0],
        "long_delay_hard": sorted(
            ordered,
            key=lambda row: (-row.long_delay_error, row.identity),
        )[0],
    }
    roles_by_identity: dict[QualityContextIdentity, list[str]] = {}
    scores_by_identity = {row.identity: row for row in ordered}
    for role in config.selection_roles:
        roles_by_identity.setdefault(role_rows[role].identity, []).append(role)
    return tuple(
        _to_selection(scores_by_identity[identity], tuple(roles_by_identity[identity]))
        for identity in sorted(roles_by_identity)
    )
