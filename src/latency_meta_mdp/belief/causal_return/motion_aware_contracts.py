"""Typed contracts for motion-aware causal-return history encoding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

_CONFIG_ID = "causal_return_motion_aware_history"
_STATE_SCHEMA_ID = "franka_ball_position_velocity"
_PHASES = frozenset({"pregrasp", "approach", "close", "lift"})


def _readonly(value, *, dtype) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class MotionAwareHistoryConfig:
    """Validated configuration for one level-specific history encoder."""

    schema_version: int
    config_id: str
    history_sample_count: int
    formal_tick_us: int
    camera_count: int
    patch_token_count: int
    vision_feature_dim: int
    robot_state_dim: int
    state_schema_id: str
    object_position_dim: int
    object_velocity_dim: int
    history_dim: int
    spatiotemporal_block_count: int
    attention_head_count: int
    typed_query_count: int
    robot_gru_layer_count: int
    dropout: float
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    max_epochs: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    huber_delta: float
    random_seed: int

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.config_id != _CONFIG_ID
            or self.state_schema_id != _STATE_SCHEMA_ID
        ):
            raise ValueError("unsupported motion-aware history config identity")
        integer_fields = (
            self.history_sample_count,
            self.formal_tick_us,
            self.camera_count,
            self.patch_token_count,
            self.vision_feature_dim,
            self.robot_state_dim,
            self.object_position_dim,
            self.object_velocity_dim,
            self.history_dim,
            self.spatiotemporal_block_count,
            self.attention_head_count,
            self.typed_query_count,
            self.robot_gru_layer_count,
            self.batch_size,
            self.max_epochs,
            self.early_stopping_patience,
            self.random_seed,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in integer_fields
        ):
            raise ValueError("motion-aware history integer fields must be positive integers")
        if (
            self.history_sample_count != 6
            or self.formal_tick_us != 20_000
            or self.camera_count != 2
            or self.patch_token_count != 196
            or self.vision_feature_dim != 384
            or self.robot_state_dim != 16
            or self.object_position_dim != 3
            or self.object_velocity_dim != 3
            or self.history_dim != 192
            or self.spatiotemporal_block_count != 3
            or self.attention_head_count != 8
            or self.typed_query_count != 2
            or self.robot_gru_layer_count != 2
            or self.history_dim % self.attention_head_count
        ):
            raise ValueError("motion-aware history architecture contract is invalid")
        real_fields = (
            self.dropout,
            self.learning_rate,
            self.weight_decay,
            self.gradient_clip_norm,
            self.early_stopping_min_delta,
            self.huber_delta,
        )
        if any(isinstance(value, bool) or not np.isfinite(value) for value in real_fields):
            raise ValueError("motion-aware history real fields must be finite numbers")
        if (
            not 0.0 <= self.dropout < 1.0
            or self.learning_rate <= 0.0
            or self.weight_decay < 0.0
            or self.gradient_clip_norm <= 0.0
            or self.early_stopping_min_delta < 0.0
            or self.huber_delta <= 0.0
        ):
            raise ValueError("motion-aware history optimization fields are invalid")


def load_motion_aware_history_config(path: Path) -> MotionAwareHistoryConfig:
    """Load an exact versioned motion-aware history config."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = set(MotionAwareHistoryConfig.__dataclass_fields__)
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("motion-aware history config fields are invalid")
    return MotionAwareHistoryConfig(**raw)


@dataclass(frozen=True)
class MotionAwareHistorySample:
    """One K6 deployment history with privileged current-state supervision."""

    episode_id: str
    level: int
    scene_seed: int
    source_tick: int
    source_phase: str
    transition_offset_ticks: int
    transition_offset_valid: bool
    vision_history: np.ndarray
    robot_history: np.ndarray
    history_valid_mask: np.ndarray
    object_position_target: np.ndarray
    object_velocity_target: np.ndarray

    def __post_init__(self) -> None:
        if (
            not isinstance(self.episode_id, str)
            or not self.episode_id
            or self.level not in (1, 2, 3)
            or isinstance(self.scene_seed, bool)
            or not isinstance(self.scene_seed, int)
            or self.scene_seed < 0
            or isinstance(self.source_tick, bool)
            or not isinstance(self.source_tick, int)
            or self.source_tick < 25
            or self.source_phase not in _PHASES
            or not isinstance(self.transition_offset_valid, bool)
            or isinstance(self.transition_offset_ticks, bool)
            or not isinstance(self.transition_offset_ticks, int)
            or self.transition_offset_valid != (self.level == 3)
            or (not self.transition_offset_valid and self.transition_offset_ticks != 0)
        ):
            raise ValueError(
                "motion-aware history sample identity or transition metadata is invalid"
            )
        vision = np.asarray(self.vision_history)
        robot = np.asarray(self.robot_history)
        valid = np.asarray(self.history_valid_mask)
        position = np.asarray(self.object_position_target)
        velocity = np.asarray(self.object_velocity_target)
        if vision.shape != (6, 2, 196, 384) or vision.dtype != np.float16:
            raise ValueError("motion-aware vision history has invalid shape or dtype")
        if robot.shape != (6, 16) or robot.dtype != np.float32:
            raise ValueError("motion-aware robot history has invalid shape or dtype")
        if valid.shape != (6,) or valid.dtype != np.bool_ or not np.all(valid):
            raise ValueError("motion-aware history validity mask must contain six real samples")
        if (
            position.shape != (3,)
            or velocity.shape != (3,)
            or position.dtype != np.float32
            or velocity.dtype != np.float32
        ):
            raise ValueError("motion-aware object targets have invalid shapes or dtypes")
        if any(not np.all(np.isfinite(value)) for value in (vision, robot, position, velocity)):
            raise ValueError("motion-aware history numeric values must be finite")
        object.__setattr__(self, "vision_history", _readonly(vision, dtype=np.float16))
        object.__setattr__(self, "robot_history", _readonly(robot, dtype=np.float32))
        object.__setattr__(self, "history_valid_mask", _readonly(valid, dtype=np.bool_))
        object.__setattr__(
            self,
            "object_position_target",
            _readonly(position, dtype=np.float32),
        )
        object.__setattr__(
            self,
            "object_velocity_target",
            _readonly(velocity, dtype=np.float32),
        )

    def deployment_inputs(self) -> dict[str, np.ndarray]:
        """Return only deployment-observable arrays in physical units."""

        return {
            "vision_history": self.vision_history,
            "robot_history": self.robot_history,
            "history_valid_mask": self.history_valid_mask,
        }


@dataclass(frozen=True)
class MotionAwareHistoryEstimate:
    """Normalized physical readouts and typed motion-aware context tokens."""

    history_context_tokens: torch.Tensor
    object_position_normalized: torch.Tensor
    object_velocity_normalized: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.history_context_tokens, torch.Tensor):
            raise ValueError("motion-aware history context must be a tensor with a valid shape")
        if self.history_context_tokens.ndim != 3:
            raise ValueError("motion-aware history context has an invalid shape")
        batch = self.history_context_tokens.shape[0]
        expected = {
            "history_context_tokens": (batch, 2, 192),
            "object_position_normalized": (batch, 3),
            "object_velocity_normalized": (batch, 3),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"motion-aware estimate {name} has an invalid shape")
            if not torch.all(torch.isfinite(value)):
                raise ValueError(f"motion-aware estimate {name} must be finite")
