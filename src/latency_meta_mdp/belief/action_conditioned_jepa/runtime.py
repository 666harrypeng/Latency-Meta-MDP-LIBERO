"""Online K6 history aligned exactly with the offline JEPA data contract."""

from __future__ import annotations

from collections import deque

import torch

from latency_meta_mdp.belief.action_conditioned_jepa.contracts import LaunchContextBatch
from latency_meta_mdp.belief.action_conditioned_jepa.data_adapter import (
    JepaProprioNormalization,
)


class JepaRuntimeHistory:
    """One-client ring buffer of six real boundaries and five executed controls."""

    def __init__(self, proprio_normalization: JepaProprioNormalization) -> None:
        if not isinstance(proprio_normalization, JepaProprioNormalization):
            raise TypeError("proprio_normalization must be JepaProprioNormalization")
        self.normalization = proprio_normalization
        self._vision: deque[torch.Tensor] = deque(maxlen=6)
        self._proprio: deque[torch.Tensor] = deque(maxlen=6)
        self._controls: deque[torch.Tensor] = deque(maxlen=5)
        self._device: torch.device | None = None
        self._mean = torch.tensor(proprio_normalization.mean, dtype=torch.float32)
        self._scale = torch.tensor(proprio_normalization.scale, dtype=torch.float32)

    @property
    def ready(self) -> bool:
        return len(self._vision) == 6 and len(self._proprio) == 6 and len(self._controls) == 5

    def append_boundary(
        self,
        *,
        vision_features: torch.Tensor,
        proprio: torch.Tensor,
        executed_control_from_previous: torch.Tensor | None,
    ) -> None:
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

    def build_launch_context(self, executable_controls: torch.Tensor) -> LaunchContextBatch:
        if not self.ready:
            raise RuntimeError("JEPA Belief requires six real boundaries before launch")
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
        return LaunchContextBatch(
            vision_history=torch.stack(tuple(self._vision)).unsqueeze(0),
            proprio_history=torch.stack(tuple(self._proprio)).unsqueeze(0),
            executed_controls=torch.stack(tuple(self._controls)).unsqueeze(0),
            executable_controls=executable_controls.detach().clone().unsqueeze(0),
        )
