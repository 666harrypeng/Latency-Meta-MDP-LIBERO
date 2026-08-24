"""Latency-weighted return-state clouds and action-target bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.special import ndtri

from latency_meta_mdp.belief_data import BeliefEpisodeView
from latency_meta_mdp.latency_law import TruncatedBetaLatencyLaw
from latency_meta_mdp.temporal_contract import TemporalContract

RETURN_STATE_DIM = 22
RETURN_STATE_NAMES = (
    *(f"robot_qpos_{index}" for index in range(7)),
    *(f"robot_qvel_{index}" for index in range(7)),
    "gripper_width",
    "gripper_width_velocity",
    "object_position_x",
    "object_position_y",
    "object_position_z",
    "object_linear_velocity_x",
    "object_linear_velocity_y",
    "object_linear_velocity_z",
)
_PHASES = frozenset({"pregrasp", "approach", "close", "lift"})


def _readonly(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ReturnBeliefAuditConfig:
    schema_version: int
    analysis_id: str
    central_probability_mass: float
    action_prefix_ticks: int
    state_scale_floor: float
    summary_quantiles: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.analysis_id != "return_belief_geometry_v1":
            raise ValueError("unsupported return-belief audit schema or identifier")
        if not np.isfinite(self.central_probability_mass) or not (
            0.0 < self.central_probability_mass < 1.0
        ):
            raise ValueError("central_probability_mass must be in (0, 1)")
        if (
            isinstance(self.action_prefix_ticks, bool)
            or not isinstance(self.action_prefix_ticks, int)
            or self.action_prefix_ticks <= 0
        ):
            raise ValueError("action_prefix_ticks must be a positive integer")
        if not np.isfinite(self.state_scale_floor) or self.state_scale_floor <= 0.0:
            raise ValueError("state_scale_floor must be finite and positive")
        quantiles = tuple(self.summary_quantiles)
        if (
            not quantiles
            or any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in quantiles)
            or tuple(sorted(set(quantiles))) != quantiles
        ):
            raise ValueError("summary_quantiles must be unique sorted probabilities")
        object.__setattr__(self, "summary_quantiles", quantiles)


def load_return_belief_audit_config(path: Path) -> ReturnBeliefAuditConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != set(
        ReturnBeliefAuditConfig.__dataclass_fields__
    ):
        raise ValueError("return-belief audit config fields are invalid")
    raw["summary_quantiles"] = tuple(raw["summary_quantiles"])
    return ReturnBeliefAuditConfig(**raw)


@dataclass(frozen=True)
class ReturnContext:
    episode_id: str
    level: int
    source_tick: int
    source_phase: str
    delay_ticks: np.ndarray
    probabilities: np.ndarray
    target_ticks: np.ndarray
    future_states: np.ndarray
    relative_positions: np.ndarray
    return_phases: np.ndarray
    return_contact: np.ndarray
    return_physical_handoff: np.ndarray
    action_targets: np.ndarray | None

    def __post_init__(self) -> None:
        branch_count = len(self.delay_ticks)
        if not self.episode_id or self.level not in (1, 2, 3) or self.source_tick < 0:
            raise ValueError("return context identity is invalid")
        if self.source_phase not in _PHASES:
            raise ValueError("return context source phase is invalid")
        shapes = {
            "delay_ticks": (branch_count,),
            "probabilities": (branch_count,),
            "target_ticks": (branch_count,),
            "future_states": (branch_count, RETURN_STATE_DIM),
            "relative_positions": (branch_count, 3),
            "return_phases": (branch_count,),
            "return_contact": (branch_count,),
            "return_physical_handoff": (branch_count,),
        }
        for name, shape in shapes.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"{name} does not match the return-context shape")
        if (
            not np.all(np.isfinite(self.probabilities))
            or not np.all(np.isfinite(self.future_states))
            or not np.all(np.isfinite(self.relative_positions))
            or not np.isclose(self.probabilities.sum(), 1.0, atol=1e-12, rtol=0)
            or not set(np.unique(self.return_phases)) <= _PHASES
        ):
            raise ValueError("return context contains invalid branch data")
        if self.action_targets is not None and self.action_targets.shape != (
            branch_count,
            50,
            7,
        ):
            raise ValueError("action targets must contain a complete H50 bundle")
        for name in shapes:
            object.__setattr__(self, name, _readonly(getattr(self, name)))
        if self.action_targets is not None:
            object.__setattr__(self, "action_targets", _readonly(self.action_targets))


def build_return_state_stream(episode: BeliefEpisodeView) -> np.ndarray:
    """Build the active 22D Franka-plus-ball state at every formal boundary."""

    deployment = episode.deployment
    supervision = episode.supervision
    gripper_width = deployment.gripper_qpos[:, 0] - deployment.gripper_qpos[:, 1]
    gripper_width_velocity = (
        deployment.gripper_qvel[:, 0] - deployment.gripper_qvel[:, 1]
    )
    state = np.concatenate(
        (
            deployment.robot_qpos,
            deployment.robot_qvel,
            gripper_width[:, None],
            gripper_width_velocity[:, None],
            supervision.object_pose[:, :3],
            supervision.object_velocity[:, :3],
        ),
        axis=1,
    )
    if state.shape != (episode.boundary_count, RETURN_STATE_DIM) or not np.all(
        np.isfinite(state)
    ):
        raise ValueError("episode cannot provide a finite 22D return-state stream")
    return _readonly(state, dtype=np.float64)


def build_return_contexts(
    *,
    episode: BeliefEpisodeView,
    temporal_contract: TemporalContract,
    latency_law: TruncatedBetaLatencyLaw,
) -> tuple[ReturnContext, ...]:
    """Construct exact weighted return clouds for every valid launch source."""

    if latency_law.delay_ticks != tuple(
        range(1, temporal_contract.maximum_delay_ticks + 1)
    ):
        raise ValueError("latency law and temporal contract delay support disagree")
    if round(latency_law.control_tick_seconds * 1_000_000) != (
        temporal_contract.formal_tick_us
    ):
        raise ValueError("latency law and temporal contract clocks disagree")
    state = build_return_state_stream(episode)
    interval = temporal_contract.belief_source_interval(
        episode_action_count=episode.transition_count
    )
    delays = np.asarray(latency_law.delay_ticks, dtype=np.int64)
    contexts: list[ReturnContext] = []
    for source_tick in range(interval.minimum, interval.maximum + 1):
        targets = source_tick + delays
        return_phase_indices = np.minimum(targets, episode.transition_count - 1)
        action_targets = None
        if (
            source_tick
            + temporal_contract.maximum_delay_ticks
            + temporal_contract.prediction_horizon
            <= episode.transition_count
        ):
            action_targets = np.stack(
                [
                    episode.expert_actions[
                        target_tick : target_tick
                        + temporal_contract.prediction_horizon
                    ]
                    for target_tick in targets
                ]
            )
        contexts.append(
            ReturnContext(
                episode_id=episode.episode_id,
                level=episode.level,
                source_tick=source_tick,
                source_phase=str(episode.expert_phase[source_tick]),
                delay_ticks=delays,
                probabilities=latency_law.probabilities,
                target_ticks=targets,
                future_states=state[targets],
                relative_positions=episode.supervision.relative_geometry[targets],
                return_phases=episode.expert_phase[return_phase_indices],
                return_contact=(
                    episode.supervision.left_pad_contact[targets]
                    | episode.supervision.right_pad_contact[targets]
                ),
                return_physical_handoff=(
                    episode.supervision.handoff_state[targets] == "physical"
                ),
                action_targets=action_targets,
            )
        )
    return tuple(contexts)


@dataclass(frozen=True)
class StateNormalization:
    mean: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        if (
            mean.shape != (RETURN_STATE_DIM,)
            or scale.shape != (RETURN_STATE_DIM,)
            or not np.all(np.isfinite(mean))
            or not np.all(np.isfinite(scale))
            or np.any(scale <= 0.0)
        ):
            raise ValueError("state normalization must contain finite positive 22D scales")
        object.__setattr__(self, "mean", _readonly(mean))
        object.__setattr__(self, "scale", _readonly(scale))


def fit_state_normalization(
    state_streams: tuple[np.ndarray, ...],
    *,
    floor: float,
) -> StateNormalization:
    if not state_streams:
        raise ValueError("at least one state stream is required")
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("normalization floor must be finite and positive")
    states = np.concatenate(state_streams, axis=0).astype(np.float64, copy=False)
    if states.ndim != 2 or states.shape[1] != RETURN_STATE_DIM or not np.all(
        np.isfinite(states)
    ):
        raise ValueError("state streams must contain finite 22D rows")
    return StateNormalization(
        mean=states.mean(axis=0),
        scale=np.maximum(states.std(axis=0), floor),
    )


def _weighted_rms_spread(
    values: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    mean = np.sum(probabilities[:, None] * values, axis=0)
    variance = np.sum(probabilities[:, None] * (values - mean) ** 2)
    return float(np.sqrt(variance / values.shape[1]))


def _weighted_spectrum(
    centered: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    weighted = np.sqrt(probabilities)[:, None] * centered
    _u, singular_values, right = np.linalg.svd(weighted, full_matrices=False)
    return singular_values**2, right


def _effective_rank(eigenvalues: np.ndarray) -> float:
    total = float(eigenvalues.sum())
    square_sum = float(np.square(eigenvalues).sum())
    if total <= 1e-15 or square_sum <= 1e-30:
        return 0.0
    return float(total**2 / square_sum)


def _central_slice(probabilities: np.ndarray, mass: float) -> slice:
    tail = (1.0 - mass) / 2.0
    cumulative = np.cumsum(probabilities)
    start = int(np.searchsorted(cumulative, tail, side="left"))
    stop = int(np.searchsorted(cumulative, 1.0 - tail, side="left")) + 1
    return slice(start, stop)


@dataclass(frozen=True)
class StateGeometryMetrics:
    combined_rms_spread: float
    robot_qpos_rms_spread: float
    robot_qvel_rms_spread: float
    gripper_width_spread: float
    gripper_width_velocity_spread: float
    object_position_rms_spread: float
    object_velocity_rms_spread: float
    relative_position_rms_spread: float
    pc1_explained_ratio: float
    pc12_explained_ratio: float
    effective_rank: float
    affine_residual_fraction: float
    pc1_skewness: float
    pc1_excess_kurtosis: float
    pc1_gaussian_quantile_rmse: float
    central_tortuosity: float
    phase_crossing_probability: float
    contact_probability: float
    physical_handoff_probability: float

    def __post_init__(self) -> None:
        if any(not np.isfinite(value) for value in self.__dict__.values()):
            raise ValueError("state geometry metrics must be finite")


def measure_state_geometry(
    context: ReturnContext,
    *,
    normalization: StateNormalization,
    central_probability_mass: float,
) -> StateGeometryMetrics:
    if not 0.0 < central_probability_mass < 1.0:
        raise ValueError("central_probability_mass must be in (0, 1)")
    probabilities = context.probabilities.astype(np.float64, copy=False)
    states = context.future_states.astype(np.float64, copy=False)
    normalized = (states - normalization.mean) / normalization.scale
    normalized_mean = np.sum(probabilities[:, None] * normalized, axis=0)
    normalized_centered = normalized - normalized_mean
    eigenvalues, right = _weighted_spectrum(normalized_centered, probabilities)
    total_variance = float(eigenvalues.sum())
    if total_variance <= 1e-15:
        pc1_ratio = 0.0
        pc12_ratio = 0.0
        affine_residual = 0.0
        skewness = 0.0
        excess_kurtosis = 0.0
        gaussian_quantile_rmse = 0.0
    else:
        pc1_ratio = float(eigenvalues[0] / total_variance)
        pc12_ratio = float(eigenvalues[:2].sum() / total_variance)
        affine_residual = float(1.0 - pc1_ratio)
        projection = normalized_centered @ right[0]
        standard_deviation = float(np.sqrt(eigenvalues[0]))
        if standard_deviation <= 1e-15:
            skewness = 0.0
            excess_kurtosis = 0.0
            gaussian_quantile_rmse = 0.0
        else:
            standardized = projection / standard_deviation
            skewness = float(np.sum(probabilities * standardized**3))
            excess_kurtosis = float(
                np.sum(probabilities * standardized**4) - 3.0
            )
            order = np.argsort(standardized)
            ordered_values = standardized[order]
            ordered_probabilities = probabilities[order]
            midpoint_cdf = (
                np.cumsum(ordered_probabilities) - 0.5 * ordered_probabilities
            )
            quantile_probabilities = np.linspace(0.05, 0.95, 19)
            empirical_quantiles = np.interp(
                quantile_probabilities,
                np.concatenate(([0.0], midpoint_cdf, [1.0])),
                np.concatenate(
                    ([ordered_values[0]], ordered_values, [ordered_values[-1]])
                ),
            )
            gaussian_quantile_rmse = float(
                np.sqrt(
                    np.mean(
                        (empirical_quantiles - ndtri(quantile_probabilities)) ** 2
                    )
                )
            )
    central = normalized[_central_slice(probabilities, central_probability_mass)]
    if len(central) < 2:
        tortuosity = 1.0
    else:
        path_length = float(np.linalg.norm(np.diff(central, axis=0), axis=1).sum())
        chord_length = float(np.linalg.norm(central[-1] - central[0]))
        tortuosity = (
            1.0
            if path_length <= 1e-15
            else float(path_length / max(chord_length, 1e-12))
        )
    return StateGeometryMetrics(
        combined_rms_spread=_weighted_rms_spread(normalized, probabilities),
        robot_qpos_rms_spread=_weighted_rms_spread(states[:, :7], probabilities),
        robot_qvel_rms_spread=_weighted_rms_spread(states[:, 7:14], probabilities),
        gripper_width_spread=_weighted_rms_spread(states[:, 14:15], probabilities),
        gripper_width_velocity_spread=_weighted_rms_spread(
            states[:, 15:16], probabilities
        ),
        object_position_rms_spread=_weighted_rms_spread(
            states[:, 16:19], probabilities
        ),
        object_velocity_rms_spread=_weighted_rms_spread(
            states[:, 19:22], probabilities
        ),
        relative_position_rms_spread=_weighted_rms_spread(
            context.relative_positions, probabilities
        ),
        pc1_explained_ratio=pc1_ratio,
        pc12_explained_ratio=pc12_ratio,
        effective_rank=_effective_rank(eigenvalues),
        affine_residual_fraction=affine_residual,
        pc1_skewness=skewness,
        pc1_excess_kurtosis=excess_kurtosis,
        pc1_gaussian_quantile_rmse=gaussian_quantile_rmse,
        central_tortuosity=tortuosity,
        phase_crossing_probability=float(
            probabilities[context.return_phases != context.source_phase].sum()
        ),
        contact_probability=float(probabilities[context.return_contact].sum()),
        physical_handoff_probability=float(
            probabilities[context.return_physical_handoff].sum()
        ),
    )


@dataclass(frozen=True)
class ActionCompatibilityMetrics:
    translation_rms_deviation: float
    rotation_rms_deviation: float
    gripper_disagreement_probability: float
    prefix_opposite_direction_rate: float
    effective_rank: float

    def __post_init__(self) -> None:
        if any(not np.isfinite(value) for value in self.__dict__.values()):
            raise ValueError("action compatibility metrics must be finite")


def _gripper_disagreement(
    gripper_actions: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    disagreements = []
    for step in gripper_actions.T:
        masses = [probabilities[step == value].sum() for value in np.unique(step)]
        disagreements.append(1.0 - float(np.square(masses).sum()))
    return float(np.mean(disagreements))


def _prefix_opposite_direction_rate(
    translation: np.ndarray,
    probabilities: np.ndarray,
    prefix_ticks: int,
) -> float:
    vectors = translation[:, :prefix_ticks].reshape(len(translation), -1)
    conflict_weight = 0.0
    comparable_weight = 0.0
    for left in range(len(vectors)):
        left_norm = float(np.linalg.norm(vectors[left]))
        for right_index in range(left + 1, len(vectors)):
            right_norm = float(np.linalg.norm(vectors[right_index]))
            if left_norm <= 1e-12 or right_norm <= 1e-12:
                continue
            pair_weight = float(2.0 * probabilities[left] * probabilities[right_index])
            comparable_weight += pair_weight
            if float(np.dot(vectors[left], vectors[right_index])) < 0.0:
                conflict_weight += pair_weight
    return 0.0 if comparable_weight <= 1e-15 else conflict_weight / comparable_weight


def measure_action_compatibility(
    context: ReturnContext,
    *,
    prefix_ticks: int,
) -> ActionCompatibilityMetrics | None:
    actions = context.action_targets
    if actions is None:
        return None
    if (
        isinstance(prefix_ticks, bool)
        or not isinstance(prefix_ticks, int)
        or not 0 < prefix_ticks <= actions.shape[1]
    ):
        raise ValueError("prefix_ticks must fit inside the action horizon")
    probabilities = context.probabilities.astype(np.float64, copy=False)
    mean = np.sum(probabilities[:, None, None] * actions, axis=0)
    centered = actions - mean
    flattened = centered.reshape(len(actions), -1)
    eigenvalues, _right = _weighted_spectrum(flattened, probabilities)
    return ActionCompatibilityMetrics(
        translation_rms_deviation=float(
            np.sqrt(
                np.sum(probabilities[:, None, None] * centered[:, :, :3] ** 2)
                / 50
            )
        ),
        rotation_rms_deviation=float(
            np.sqrt(
                np.sum(probabilities[:, None, None] * centered[:, :, 3:6] ** 2)
                / 50
            )
        ),
        gripper_disagreement_probability=_gripper_disagreement(
            actions[:, :, 6], probabilities
        ),
        prefix_opposite_direction_rate=_prefix_opposite_direction_rate(
            actions[:, :, :3], probabilities, prefix_ticks
        ),
        effective_rank=_effective_rank(eigenvalues),
    )
