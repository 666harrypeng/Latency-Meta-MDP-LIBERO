"""Consumer-independent tensor contracts for Action-Conditioned JEPA Belief."""

from __future__ import annotations

from dataclasses import dataclass

import torch

_HISTORY_TICKS = 6
_EXECUTED_CONTROL_TICKS = 5
_MAXIMUM_DELAY_TICKS = 20
_CAMERA_COUNT = 2
_PATCH_TOKEN_COUNT = 196
_VISUAL_FEATURE_DIM = 384
_PROPRIO_DIM = 16
_ACTION_DIM = 7


def _require_tensor(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(value.shape) != shape or value.dtype != dtype:
        raise ValueError(f"{name} must have shape {shape} and dtype {dtype}")
    return value


def _require_same_device(*tensors: torch.Tensor) -> torch.device:
    devices = {tensor.device for tensor in tensors}
    if len(devices) != 1:
        raise ValueError("all tensors in one Belief contract must use the same device")
    return tensors[0].device


def _validate_finite(*tensors: torch.Tensor) -> None:
    if any(not bool(torch.isfinite(tensor).all()) for tensor in tensors):
        raise ValueError("Belief tensors must contain only finite values")


@dataclass(frozen=True)
class JepaLaunchSupportContract:
    """Availability interval of a K6 predictor inside an H50/E25 client chunk."""

    minimum_cursor: int
    maximum_cursor: int
    prediction_horizon: int
    maximum_delay_ticks: int

    def __post_init__(self) -> None:
        values = (
            self.minimum_cursor,
            self.maximum_cursor,
            self.prediction_horizon,
            self.maximum_delay_ticks,
        )
        if any(type(value) is not int or value <= 0 for value in values):
            raise ValueError("JEPA launch-support values must be positive integers")
        if not self.minimum_cursor <= self.maximum_cursor < self.prediction_horizon:
            raise ValueError("JEPA launch-support cursor interval is invalid")
        if self.remaining_controls_at_latest_cursor < self.maximum_delay_ticks:
            raise ValueError("latest launch cursor does not retain D20 buffer coverage")

    @property
    def remaining_controls_at_latest_cursor(self) -> int:
        return self.prediction_horizon - self.maximum_cursor

    def is_belief_ready(self, cursor: int) -> bool:
        if type(cursor) is not int:
            raise TypeError("cursor must be an integer")
        return self.minimum_cursor <= cursor <= self.maximum_cursor

    def require_supported_launch_cursor(self, cursor: int) -> None:
        if not self.is_belief_ready(cursor):
            raise ValueError("cursor is outside the JEPA Belief availability interval")


@dataclass(frozen=True)
class LaunchContextBatch:
    """Deployment-visible K6 history and D20 executable control prefix."""

    vision_history: torch.Tensor
    proprio_history: torch.Tensor
    executed_controls: torch.Tensor
    executable_controls: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.vision_history, torch.Tensor) or self.vision_history.ndim != 5:
            raise ValueError("vision_history must be a batched five-dimensional tensor")
        batch_size = self.vision_history.shape[0]
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("LaunchContextBatch requires a non-empty batch")
        vision = _require_tensor(
            self.vision_history,
            name="vision_history",
            shape=(
                batch_size,
                _HISTORY_TICKS,
                _CAMERA_COUNT,
                _PATCH_TOKEN_COUNT,
                _VISUAL_FEATURE_DIM,
            ),
            dtype=torch.float16,
        )
        proprio = _require_tensor(
            self.proprio_history,
            name="proprio_history",
            shape=(batch_size, _HISTORY_TICKS, _PROPRIO_DIM),
            dtype=torch.float32,
        )
        executed = _require_tensor(
            self.executed_controls,
            name="executed_controls",
            shape=(batch_size, _EXECUTED_CONTROL_TICKS, _ACTION_DIM),
            dtype=torch.float32,
        )
        executable = _require_tensor(
            self.executable_controls,
            name="executable_controls",
            shape=(batch_size, _MAXIMUM_DELAY_TICKS, _ACTION_DIM),
            dtype=torch.float32,
        )
        _require_same_device(vision, proprio, executed, executable)

    @property
    def batch_size(self) -> int:
        return self.vision_history.shape[0]

    @property
    def device(self) -> torch.device:
        return self.vision_history.device

    @property
    def proprio_space(self) -> str:
        return "normalized"

    @property
    def control_space(self) -> str:
        return "controller_native"

    def validate_finite(self) -> None:
        _validate_finite(
            self.vision_history,
            self.proprio_history,
            self.executed_controls,
            self.executable_controls,
        )


@dataclass(frozen=True)
class FutureLatentRollout:
    """All twenty fixed-delay futures produced by one stationary predictor rollout."""

    future_visual_latents: torch.Tensor
    future_proprio: torch.Tensor

    def __post_init__(self) -> None:
        if (
            not isinstance(self.future_visual_latents, torch.Tensor)
            or self.future_visual_latents.ndim != 5
        ):
            raise ValueError("future_visual_latents must be a batched five-dimensional tensor")
        batch_size = self.future_visual_latents.shape[0]
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("FutureLatentRollout requires a non-empty batch")
        visual = _require_tensor(
            self.future_visual_latents,
            name="future_visual_latents",
            shape=(
                batch_size,
                _MAXIMUM_DELAY_TICKS,
                _CAMERA_COUNT,
                _PATCH_TOKEN_COUNT,
                _VISUAL_FEATURE_DIM,
            ),
            dtype=torch.float16,
        )
        proprio = _require_tensor(
            self.future_proprio,
            name="future_proprio",
            shape=(batch_size, _MAXIMUM_DELAY_TICKS, _PROPRIO_DIM),
            dtype=torch.float32,
        )
        _require_same_device(visual, proprio)

    @property
    def batch_size(self) -> int:
        return self.future_visual_latents.shape[0]

    @property
    def device(self) -> torch.device:
        return self.future_visual_latents.device

    @property
    def future_proprio_units(self) -> str:
        return "physical_si"

    def validate_finite(self) -> None:
        _validate_finite(self.future_visual_latents, self.future_proprio)


@dataclass(frozen=True)
class ReturnLatentBeliefBatch:
    """Latency-weighted, consumer-independent return-state latent Belief."""

    delay_ticks: torch.Tensor
    delay_probabilities: torch.Tensor
    future_visual_latents: torch.Tensor
    future_proprio: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.delay_probabilities, torch.Tensor):
            raise TypeError("delay_probabilities must be a torch.Tensor")
        if self.delay_probabilities.ndim != 2:
            raise ValueError("delay_probabilities must have shape [B,20]")
        batch_size = self.delay_probabilities.shape[0]
        if batch_size <= 0:
            raise ValueError("ReturnLatentBeliefBatch requires a non-empty batch")
        delays = _require_tensor(
            self.delay_ticks,
            name="delay_ticks",
            shape=(_MAXIMUM_DELAY_TICKS,),
            dtype=torch.int64,
        )
        probabilities = _require_tensor(
            self.delay_probabilities,
            name="delay_probabilities",
            shape=(batch_size, _MAXIMUM_DELAY_TICKS),
            dtype=torch.float32,
        )
        visual = _require_tensor(
            self.future_visual_latents,
            name="future_visual_latents",
            shape=(
                batch_size,
                _MAXIMUM_DELAY_TICKS,
                _CAMERA_COUNT,
                _PATCH_TOKEN_COUNT,
                _VISUAL_FEATURE_DIM,
            ),
            dtype=torch.float16,
        )
        proprio = _require_tensor(
            self.future_proprio,
            name="future_proprio",
            shape=(batch_size, _MAXIMUM_DELAY_TICKS, _PROPRIO_DIM),
            dtype=torch.float32,
        )
        _require_same_device(delays, probabilities, visual, proprio)
        expected_delays = torch.arange(
            1,
            _MAXIMUM_DELAY_TICKS + 1,
            dtype=torch.int64,
            device=delays.device,
        )
        if not torch.equal(delays, expected_delays):
            raise ValueError("delay_ticks must be the canonical sequence 1..20")
        probability_valid = (
            bool(torch.isfinite(probabilities).all())
            and bool((probabilities >= 0).all())
            and bool(
                torch.allclose(
                    probabilities.sum(dim=1),
                    torch.ones(batch_size, dtype=torch.float32, device=probabilities.device),
                    atol=1e-6,
                    rtol=0.0,
                )
            )
        )
        if not probability_valid:
            raise ValueError("delay probabilities must be finite, nonnegative, and sum to one")

    @property
    def batch_size(self) -> int:
        return self.delay_probabilities.shape[0]

    @property
    def maximum_delay_ticks(self) -> int:
        return _MAXIMUM_DELAY_TICKS

    @property
    def future_proprio_units(self) -> str:
        return "physical_si"

    def validate_finite(self) -> None:
        _validate_finite(
            self.delay_probabilities,
            self.future_visual_latents,
            self.future_proprio,
        )
