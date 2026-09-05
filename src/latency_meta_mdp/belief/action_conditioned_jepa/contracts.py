"""Consumer-independent tensor contracts for Action-Conditioned JEPA Belief."""

from __future__ import annotations

from dataclasses import dataclass

import torch

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


def _validate_native_delay_ticks(delays: torch.Tensor) -> int:
    if delays.ndim != 1 or delays.numel() <= 0:
        raise ValueError("native delay ticks must be a non-empty vector")
    if delays.dtype != torch.int64:
        raise ValueError("native delay ticks must use int64")
    stride = int(delays[0].item())
    if stride <= 0:
        raise ValueError("native delay stride must be positive")
    expected = torch.arange(
        stride,
        _MAXIMUM_DELAY_TICKS + 1,
        stride,
        dtype=torch.int64,
        device=delays.device,
    )
    if not torch.equal(delays, expected) or int(delays[-1].item()) != _MAXIMUM_DELAY_TICKS:
        raise ValueError("native delay ticks must be a uniform grid ending at D20")
    return stride


@dataclass(frozen=True)
class JepaLaunchSupportContract:
    """Independent history-readiness and H50/E25 launch-cursor constraints."""

    required_history_ticks: int
    latest_launch_cursor: int
    prediction_horizon: int
    maximum_delay_ticks: int

    def __post_init__(self) -> None:
        values = (
            self.required_history_ticks,
            self.latest_launch_cursor,
            self.prediction_horizon,
            self.maximum_delay_ticks,
        )
        if any(type(value) is not int or value <= 0 for value in values):
            raise ValueError("JEPA launch-support values must be positive integers")
        if self.latest_launch_cursor >= self.prediction_horizon:
            raise ValueError("JEPA latest launch cursor must precede the chunk horizon")
        if self.remaining_controls_at_latest_cursor < self.maximum_delay_ticks:
            raise ValueError("latest launch cursor does not retain D20 buffer coverage")

    @property
    def remaining_controls_at_latest_cursor(self) -> int:
        return self.prediction_horizon - self.latest_launch_cursor

    def is_belief_ready(self, *, available_history_ticks: int) -> bool:
        if type(available_history_ticks) is not int or available_history_ticks < 0:
            raise TypeError("available_history_ticks must be a nonnegative integer")
        return available_history_ticks >= self.required_history_ticks

    def is_supported_launch_cursor(self, cursor: int) -> bool:
        if type(cursor) is not int:
            raise TypeError("cursor must be an integer")
        return 0 <= cursor <= self.latest_launch_cursor

    def require_supported_launch(self, *, cursor: int, available_history_ticks: int) -> None:
        if not self.is_belief_ready(available_history_ticks=available_history_ticks):
            raise ValueError("available observation history is insufficient for JEPA Belief")
        if not self.is_supported_launch_cursor(cursor):
            raise ValueError("cursor is outside the shield-supported launch interval")


@dataclass(frozen=True)
class LaunchContextBatch:
    """Deployment-visible tensors for the dense 20 ms reference implementation."""

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
        history_count = int(self.vision_history.shape[1])
        if history_count < 2:
            raise ValueError("vision_history requires at least two observations")
        if not isinstance(self.executed_controls, torch.Tensor) or self.executed_controls.ndim != 4:
            raise ValueError("executed_controls must have shape [B,W-1,f,7]")
        if (
            not isinstance(self.executable_controls, torch.Tensor)
            or self.executable_controls.ndim != 4
        ):
            raise ValueError("executable_controls must have shape [B,H,f,7]")
        stride = int(self.executed_controls.shape[2])
        native_steps = int(self.executable_controls.shape[1])
        if stride <= 0 or native_steps * stride != _MAXIMUM_DELAY_TICKS:
            raise ValueError("macro controls must cover exactly D20 source ticks")
        vision = _require_tensor(
            self.vision_history,
            name="vision_history",
            shape=(
                batch_size,
                history_count,
                _CAMERA_COUNT,
                _PATCH_TOKEN_COUNT,
                _VISUAL_FEATURE_DIM,
            ),
            dtype=torch.float16,
        )
        proprio = _require_tensor(
            self.proprio_history,
            name="proprio_history",
            shape=(batch_size, history_count, _PROPRIO_DIM),
            dtype=torch.float32,
        )
        executed = _require_tensor(
            self.executed_controls,
            name="executed_controls",
            shape=(batch_size, history_count - 1, stride, _ACTION_DIM),
            dtype=torch.float32,
        )
        executable = _require_tensor(
            self.executable_controls,
            name="executable_controls",
            shape=(batch_size, native_steps, stride, _ACTION_DIM),
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
    def history_observation_count(self) -> int:
        return int(self.vision_history.shape[1])

    @property
    def model_stride_ticks(self) -> int:
        return int(self.executed_controls.shape[2])

    @property
    def native_rollout_steps(self) -> int:
        return int(self.executable_controls.shape[1])

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
    """All model-native futures through D20 from one stationary predictor rollout."""

    native_delay_ticks: torch.Tensor
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
        if not isinstance(self.native_delay_ticks, torch.Tensor):
            raise TypeError("native_delay_ticks must be a torch.Tensor")
        stride = _validate_native_delay_ticks(self.native_delay_ticks)
        native_steps = int(self.native_delay_ticks.numel())
        visual = _require_tensor(
            self.future_visual_latents,
            name="future_visual_latents",
            shape=(
                batch_size,
                native_steps,
                _CAMERA_COUNT,
                _PATCH_TOKEN_COUNT,
                _VISUAL_FEATURE_DIM,
            ),
            dtype=torch.float16,
        )
        proprio = _require_tensor(
            self.future_proprio,
            name="future_proprio",
            shape=(batch_size, native_steps, _PROPRIO_DIM),
            dtype=torch.float32,
        )
        _require_same_device(self.native_delay_ticks, visual, proprio)
        if native_steps * stride != _MAXIMUM_DELAY_TICKS:
            raise ValueError("native rollout must end at D20")

    @property
    def batch_size(self) -> int:
        return self.future_visual_latents.shape[0]

    @property
    def device(self) -> torch.device:
        return self.future_visual_latents.device

    @property
    def native_delay_count(self) -> int:
        return int(self.native_delay_ticks.numel())

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
            raise ValueError("delay_probabilities must have shape [B,H]")
        batch_size = self.delay_probabilities.shape[0]
        if batch_size <= 0:
            raise ValueError("ReturnLatentBeliefBatch requires a non-empty batch")
        if not isinstance(self.delay_ticks, torch.Tensor):
            raise TypeError("delay_ticks must be a torch.Tensor")
        _validate_native_delay_ticks(self.delay_ticks)
        native_steps = int(self.delay_ticks.numel())
        delays = self.delay_ticks
        probabilities = _require_tensor(
            self.delay_probabilities,
            name="delay_probabilities",
            shape=(batch_size, native_steps),
            dtype=torch.float32,
        )
        visual = _require_tensor(
            self.future_visual_latents,
            name="future_visual_latents",
            shape=(
                batch_size,
                native_steps,
                _CAMERA_COUNT,
                _PATCH_TOKEN_COUNT,
                _VISUAL_FEATURE_DIM,
            ),
            dtype=torch.float16,
        )
        proprio = _require_tensor(
            self.future_proprio,
            name="future_proprio",
            shape=(batch_size, native_steps, _PROPRIO_DIM),
            dtype=torch.float32,
        )
        _require_same_device(delays, probabilities, visual, proprio)
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
        return int(self.delay_ticks[-1].item())

    @property
    def native_delay_count(self) -> int:
        return int(self.delay_ticks.numel())

    @property
    def future_proprio_units(self) -> str:
        return "physical_si"

    def validate_finite(self) -> None:
        _validate_finite(
            self.delay_probabilities,
            self.future_visual_latents,
            self.future_proprio,
        )
