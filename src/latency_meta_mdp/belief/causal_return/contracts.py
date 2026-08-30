"""Typed contracts for causal-return information-state estimation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

_PHASES = frozenset({"pregrasp", "approach", "close", "lift"})
_HISTORY_TIME_MS = np.asarray([-100, -80, -60, -40, -20, 0], dtype=np.int64)


def _readonly(value, *, dtype=None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class InformationStateConfig:
    schema_version: int
    config_id: str
    history_sample_count: int
    formal_tick_ms: int
    camera_count: int
    patch_token_count: int
    vision_feature_dim: int
    robot_state_dim: int
    object_state_dim: int
    model_dim: int
    state_token_count: int
    spatial_pool_query_count: int
    temporal_fusion_layer_count: int
    attention_head_count: int
    dropout: float
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    max_epochs: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    log_scale_min: float
    log_scale_max: float
    random_seed: int

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.config_id != "causal_return_information_state":
            raise ValueError("unsupported information-state config identity")
        dimensions = (
            self.history_sample_count,
            self.formal_tick_ms,
            self.camera_count,
            self.patch_token_count,
            self.vision_feature_dim,
            self.robot_state_dim,
            self.object_state_dim,
            self.model_dim,
            self.state_token_count,
            self.spatial_pool_query_count,
            self.temporal_fusion_layer_count,
            self.attention_head_count,
            self.batch_size,
            self.max_epochs,
            self.early_stopping_patience,
            self.random_seed,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in dimensions
        ):
            raise ValueError("information-state integer fields must be positive")
        if (
            self.history_sample_count != 6
            or self.formal_tick_ms != 20
            or self.camera_count != 2
            or self.patch_token_count != 196
            or self.vision_feature_dim != 384
            or self.robot_state_dim != 16
            or self.object_state_dim != 6
            or self.model_dim != 192
            or self.state_token_count != 4
            or self.spatial_pool_query_count != 4
            or self.attention_head_count != 6
            or self.model_dim % self.attention_head_count
        ):
            raise ValueError("information-state architecture contract is invalid")
        real_values = (
            self.dropout,
            self.learning_rate,
            self.weight_decay,
            self.gradient_clip_norm,
            self.early_stopping_min_delta,
            self.log_scale_min,
            self.log_scale_max,
        )
        if any(not np.isfinite(value) for value in real_values):
            raise ValueError("information-state real fields must be finite")
        if (
            not 0.0 <= self.dropout < 1.0
            or self.learning_rate <= 0.0
            or self.weight_decay < 0.0
            or self.gradient_clip_norm <= 0.0
            or self.early_stopping_min_delta < 0.0
            or self.log_scale_min >= self.log_scale_max
        ):
            raise ValueError("information-state optimization fields are invalid")


def load_information_state_config(path: Path) -> InformationStateConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != set(InformationStateConfig.__dataclass_fields__):
        raise ValueError("information-state config fields are invalid")
    return InformationStateConfig(**raw)


@dataclass(frozen=True)
class InformationStateSample:
    episode_id: str
    level: int
    scene_seed: int
    source_tick: int
    source_phase: str
    motion_curvature: float
    motion_transition_distance_ticks: int
    vision_history: np.ndarray
    robot_history: np.ndarray
    history_time_ms: np.ndarray
    history_valid_mask: np.ndarray
    object_state_target: np.ndarray

    def __post_init__(self) -> None:
        if (
            not self.episode_id
            or self.level not in (1, 2, 3)
            or isinstance(self.scene_seed, bool)
            or not isinstance(self.scene_seed, int)
            or self.scene_seed < 0
            or isinstance(self.source_tick, bool)
            or not isinstance(self.source_tick, int)
            or self.source_tick < 25
            or self.source_phase not in _PHASES
            or not np.isfinite(self.motion_curvature)
            or self.motion_curvature < 0.0
            or isinstance(self.motion_transition_distance_ticks, bool)
            or not isinstance(self.motion_transition_distance_ticks, int)
            or self.motion_transition_distance_ticks < -1
        ):
            raise ValueError("information-state sample identity is invalid")
        vision = np.asarray(self.vision_history)
        robot = np.asarray(self.robot_history)
        times = np.asarray(self.history_time_ms)
        valid = np.asarray(self.history_valid_mask)
        target = np.asarray(self.object_state_target)
        if vision.shape != (6, 2, 196, 384) or vision.dtype != np.float16:
            raise ValueError("information-state vision history is invalid")
        if robot.shape != (6, 16):
            raise ValueError("information-state robot history is invalid")
        if times.shape != (6,) or not np.array_equal(times, _HISTORY_TIME_MS):
            raise ValueError("information-state timestamps are invalid")
        if valid.shape != (6,) or valid.dtype != np.bool_ or not np.all(valid):
            raise ValueError("information-state valid mask must contain six real samples")
        if target.shape != (6,):
            raise ValueError("information-state object target is invalid")
        if any(not np.all(np.isfinite(value)) for value in (vision, robot, target)):
            raise ValueError("information-state numeric values must be finite")
        object.__setattr__(self, "vision_history", _readonly(vision, dtype=np.float16))
        object.__setattr__(self, "robot_history", _readonly(robot, dtype=np.float32))
        object.__setattr__(self, "history_time_ms", _readonly(times, dtype=np.int64))
        object.__setattr__(self, "history_valid_mask", _readonly(valid, dtype=np.bool_))
        object.__setattr__(self, "object_state_target", _readonly(target, dtype=np.float32))

    def model_inputs(self) -> dict[str, np.ndarray]:
        return {
            "vision_history": self.vision_history,
            "robot_history": self.robot_history,
            "history_time_ms": self.history_time_ms,
            "history_valid_mask": self.history_valid_mask,
        }


@dataclass(frozen=True)
class CurrentStateEstimate:
    robot_state: torch.Tensor
    object_state_mean: torch.Tensor
    object_state_log_scale: torch.Tensor
    state_tokens: torch.Tensor

    def __post_init__(self) -> None:
        batch = self.robot_state.shape[0]
        expected = {
            "robot_state": (batch, 16),
            "object_state_mean": (batch, 6),
            "object_state_log_scale": (batch, 6),
            "state_tokens": (batch, 4, 192),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"current-state estimate {name} has invalid shape")
            if not torch.all(torch.isfinite(value)):
                raise ValueError(f"current-state estimate {name} must be finite")
