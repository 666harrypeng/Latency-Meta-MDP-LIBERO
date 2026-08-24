"""Feature-backed causal samples for predictive return-belief training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from latency_meta_mdp.return_belief_geometry import RETURN_STATE_DIM


def _readonly(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class FeatureBeliefSample:
    episode_id: str
    level: int
    scene_seed: int
    source_tick: int
    vision_history: np.ndarray
    robot_proprio_history: np.ndarray
    remaining_actions: np.ndarray
    latency_probabilities: np.ndarray
    target_delay_ticks: np.ndarray
    target_states: np.ndarray
    target_interaction_mode: np.ndarray
    target_absorbing: np.ndarray

    def __post_init__(self) -> None:
        if (
            not self.episode_id
            or self.level not in (1, 2, 3)
            or isinstance(self.scene_seed, bool)
            or not isinstance(self.scene_seed, int)
            or isinstance(self.source_tick, bool)
            or not isinstance(self.source_tick, int)
            or self.source_tick < 0
        ):
            raise ValueError("feature belief sample identity is invalid")
        shapes = {
            "vision_history": (6, 2, 196, 384),
            "robot_proprio_history": (6, 16),
            "remaining_actions": (25, 7),
            "latency_probabilities": (20,),
            "target_delay_ticks": (20,),
            "target_states": (20, RETURN_STATE_DIM),
            "target_interaction_mode": (20,),
            "target_absorbing": (20,),
        }
        for name, expected in shapes.items():
            if np.asarray(getattr(self, name)).shape != expected:
                raise ValueError(f"{name} does not match the feature belief contract")
        if (
            self.vision_history.dtype != np.float16
            or not np.all(np.isfinite(self.vision_history))
            or not np.all(np.isfinite(self.robot_proprio_history))
            or not np.all(np.isfinite(self.remaining_actions))
            or not np.all(np.isfinite(self.latency_probabilities))
            or not np.all(np.isfinite(self.target_states))
            or not np.isclose(self.latency_probabilities.sum(), 1.0, atol=1e-12, rtol=0)
            or not np.array_equal(self.target_delay_ticks, np.arange(1, 21))
        ):
            raise ValueError("feature belief sample contains invalid numeric values")
        for name, dtype in (
            ("vision_history", np.float16),
            ("robot_proprio_history", np.float32),
            ("remaining_actions", np.float32),
            ("latency_probabilities", np.float64),
            ("target_delay_ticks", np.int64),
            ("target_states", np.float32),
            ("target_interaction_mode", np.int8),
            ("target_absorbing", np.bool_),
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name), dtype=dtype))


@dataclass(frozen=True)
class FeatureDelayQueries:
    delay_ticks: np.ndarray
    probabilities: np.ndarray
    target_states: np.ndarray
    interaction_mode: np.ndarray
    absorbing: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.delay_ticks)
        expected = {
            "delay_ticks": (count,),
            "probabilities": (count,),
            "target_states": (count, RETURN_STATE_DIM),
            "interaction_mode": (count,),
            "absorbing": (count,),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"{name} does not match decoder query contract")
            object.__setattr__(self, name, _readonly(value))


def _queries_from_rows(sample: FeatureBeliefSample, rows: np.ndarray) -> FeatureDelayQueries:
    return FeatureDelayQueries(
        delay_ticks=sample.target_delay_ticks[rows],
        probabilities=sample.latency_probabilities[rows],
        target_states=sample.target_states[rows],
        interaction_mode=sample.target_interaction_mode[rows],
        absorbing=sample.target_absorbing[rows],
    )


def sample_feature_delay_queries(
    *,
    sample: FeatureBeliefSample,
    rng: np.random.Generator,
    query_count: int,
) -> FeatureDelayQueries:
    if isinstance(query_count, bool) or not isinstance(query_count, int) or query_count <= 0:
        raise ValueError("query_count must be a positive integer")
    rows = rng.choice(
        len(sample.target_delay_ticks),
        size=query_count,
        replace=True,
        p=sample.latency_probabilities,
    )
    return _queries_from_rows(sample, np.asarray(rows, dtype=np.int64))


def exhaustive_feature_delay_queries(sample: FeatureBeliefSample) -> FeatureDelayQueries:
    return _queries_from_rows(
        sample,
        np.arange(len(sample.target_delay_ticks), dtype=np.int64),
    )
