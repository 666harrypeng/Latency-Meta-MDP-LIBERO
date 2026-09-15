"""Online 50 Hz history sampled for any admitted JEPA temporal configuration."""

from __future__ import annotations

from collections import deque

import torch

from latency_meta_mdp.belief.jepa.config import JepaTemporalSampling
from latency_meta_mdp.belief.jepa.contracts import (
    ForecastQuery,
    LaunchContextBatch,
)
from latency_meta_mdp.belief.jepa.corpus import (
    JepaProprioNormalization,
)


class JepaRuntimeHistory:
    """One-client 50 Hz ring buffer sampled at a config-bound model stride."""

    def __init__(
        self,
        proprio_normalization: JepaProprioNormalization,
        *,
        temporal_sampling: JepaTemporalSampling,
    ) -> None:
        if not isinstance(proprio_normalization, JepaProprioNormalization):
            raise TypeError("proprio_normalization must be JepaProprioNormalization")
        if not isinstance(temporal_sampling, JepaTemporalSampling):
            raise TypeError("temporal_sampling must be JepaTemporalSampling")
        self.normalization = proprio_normalization
        self.temporal_sampling = temporal_sampling
        source_boundaries = temporal_sampling.history_span_ticks + 1
        self._vision: deque[torch.Tensor] = deque(maxlen=source_boundaries)
        self._proprio: deque[torch.Tensor] = deque(maxlen=source_boundaries)
        self._controls: deque[torch.Tensor] = deque(maxlen=temporal_sampling.history_span_ticks)
        self._device: torch.device | None = None
        self._mean = torch.tensor(proprio_normalization.mean, dtype=torch.float32)
        self._scale = torch.tensor(proprio_normalization.scale, dtype=torch.float32)
        self._last_tick: int | None = None
        self._episode_id: str | None = None

    def reset(self) -> None:
        """Discard all observations/controls before starting a different episode."""
        self._vision.clear()
        self._proprio.clear()
        self._controls.clear()
        self._last_tick = None
        self._episode_id = None
        self._device = None

    @property
    def ready(self) -> bool:
        required_boundaries = self.temporal_sampling.history_span_ticks + 1
        return (
            len(self._vision) == required_boundaries
            and len(self._proprio) == required_boundaries
            and len(self._controls) == self.temporal_sampling.history_span_ticks
        )

    def append_boundary(
        self,
        *,
        vision_features: torch.Tensor,
        proprio: torch.Tensor,
        executed_control_from_previous: torch.Tensor | None,
        formal_tick: int | None = None,
        episode_id: str | None = None,
    ) -> None:
        clocked = formal_tick is not None or episode_id is not None
        if clocked:
            if (
                type(formal_tick) is not int
                or formal_tick < 0
                or not isinstance(episode_id, str)
                or not episode_id
            ):
                raise ValueError("clocked history requires a valid tick and episode identity")
            if self._vision:
                if episode_id != self._episode_id:
                    raise ValueError("runtime history episode changed without reset")
                if self._last_tick is None or formal_tick != self._last_tick + 1:
                    raise ValueError("runtime history ticks must be consecutive")
        elif self._last_tick is not None:
            raise ValueError("clocked runtime history requires consecutive timestamped inputs")
        if not isinstance(vision_features, torch.Tensor) or not isinstance(proprio, torch.Tensor):
            raise TypeError("runtime vision and proprio must be torch tensors")
        if tuple(vision_features.shape) != (2, 196, 384) or vision_features.dtype != torch.float16:
            raise ValueError("runtime vision_features must be float16[2,196,384]")
        if tuple(proprio.shape) != (16,) or proprio.dtype != torch.float32:
            raise ValueError("runtime proprio must be physical float32[16]")
        if vision_features.device != proprio.device:
            raise ValueError("runtime boundary tensors must use one device")
        first = not self._vision
        if first and executed_control_from_previous is not None:
            raise ValueError("the first runtime boundary cannot have a previous control")
        if not first and executed_control_from_previous is None:
            raise ValueError("subsequent runtime boundaries require the executed previous control")
        if self._device is None:
            self._device = vision_features.device
            self._mean = self._mean.to(self._device)
            self._scale = self._scale.to(self._device)
        elif vision_features.device != self._device:
            raise ValueError("runtime history cannot change device")
        if not bool(torch.isfinite(vision_features).all()) or not bool(
            torch.isfinite(proprio).all()
        ):
            raise ValueError("runtime boundary tensors must be finite")
        if executed_control_from_previous is not None:
            control = executed_control_from_previous
            if (
                not isinstance(control, torch.Tensor)
                or tuple(control.shape) != (7,)
                or control.dtype != torch.float32
                or control.device != self._device
                or not bool(torch.isfinite(control).all())
                or bool((control < -1.0).any())
                or bool((control > 1.0).any())
            ):
                raise ValueError("runtime previous control must be controller-native float32[7]")
            self._controls.append(control.detach().clone())
        normalized = (proprio - self._mean) / self._scale
        self._vision.append(vision_features.detach().clone())
        self._proprio.append(normalized.detach().clone())
        if clocked:
            self._last_tick = formal_tick
            self._episode_id = episode_id

    def build_forecast_query(
        self, executable_controls: torch.Tensor, *, query_ticks: int
    ) -> ForecastQuery:
        if not self.ready or self._last_tick is None:
            raise RuntimeError("Direct query requires complete, timestamped real history")
        if (
            self.temporal_sampling.model_stride_ticks != 4
            or self.temporal_sampling.history_observation_count != 3
        ):
            raise ValueError("Direct query requires stride4 W3 history")
        if type(query_ticks) is not int or not 0 <= query_ticks <= 20:
            raise ValueError("query ticks must be an integer in0..20")
        context = self.build_launch_context(executable_controls)
        mask = torch.arange(20, device=self._device)[None] < query_ticks
        query = ForecastQuery(
            vision_history=context.vision_history,
            proprio_history=context.proprio_history,
            executed_controls=context.executed_controls,
            executable_controls=torch.where(mask[..., None], executable_controls[None], 0),
            control_mask=mask,
            query_ticks=torch.tensor([query_ticks], device=self._device),
            source_ticks=torch.tensor([self._last_tick], device=self._device),
        )
        query.validate_finite()
        return query

    def build_launch_context(self, executable_controls: torch.Tensor) -> LaunchContextBatch:
        if not self.ready:
            raise RuntimeError("JEPA Belief requires the complete real history before launch")
        if (
            not isinstance(executable_controls, torch.Tensor)
            or tuple(executable_controls.shape) != (20, 7)
            or executable_controls.dtype != torch.float32
            or executable_controls.device != self._device
            or not bool(torch.isfinite(executable_controls).all())
            or bool((executable_controls < -1.0).any())
            or bool((executable_controls > 1.0).any())
        ):
            raise ValueError("runtime executable controls must be controller-native float32[20,7]")
        stride = self.temporal_sampling.model_stride_ticks
        history_count = self.temporal_sampling.history_observation_count
        vision = tuple(self._vision)[::stride]
        proprio = tuple(self._proprio)[::stride]
        if len(vision) != history_count or len(proprio) != history_count:
            raise RuntimeError("runtime history sampling disagrees with the temporal contract")
        past_controls = torch.stack(tuple(self._controls)).reshape(
            history_count - 1,
            stride,
            7,
        )
        future_controls = (
            executable_controls.detach()
            .clone()
            .reshape(
                self.temporal_sampling.native_rollout_steps,
                stride,
                7,
            )
        )
        return LaunchContextBatch(
            vision_history=torch.stack(vision).unsqueeze(0),
            proprio_history=torch.stack(proprio).unsqueeze(0),
            executed_controls=past_controls.unsqueeze(0),
            executable_controls=future_controls.unsqueeze(0),
        )
