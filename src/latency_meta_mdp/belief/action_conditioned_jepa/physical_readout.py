"""Frozen-latent physical readout used only to qualify JEPA futures."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from latency_meta_mdp.belief.action_conditioned_jepa.data_adapter import JepaEpisodeRecord
from latency_meta_mdp.belief.action_conditioned_jepa.temporal_signal_audit import (
    TemporalSignalEpisode,
)
from latency_meta_mdp.belief.action_conditioned_jepa.upstream_adapter import (
    load_upstream_reference,
    verify_upstream_checkout,
)


@dataclass(frozen=True)
class ObjectStateNormalization:
    mean: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float32)
        scale = np.asarray(self.scale, dtype=np.float32)
        if (
            mean.shape != (6,)
            or scale.shape != (6,)
            or not np.all(np.isfinite(mean))
            or not np.all(np.isfinite(scale))
            or np.any(scale <= 0)
        ):
            raise ValueError("object-state normalization must contain finite 6D mean/scale")
        mean.setflags(write=False)
        scale.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or value.shape[-1] != 6:
            raise ValueError("object state tensor must end in six dimensions")
        mean = torch.tensor(self.mean, device=value.device, dtype=value.dtype)
        scale = torch.tensor(self.scale, device=value.device, dtype=value.dtype)
        return (value - mean) / scale

    def denormalize(self, value: torch.Tensor) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or value.shape[-1] != 6:
            raise ValueError("object state tensor must end in six dimensions")
        mean = torch.tensor(self.mean, device=value.device, dtype=value.dtype)
        scale = torch.tensor(self.scale, device=value.device, dtype=value.dtype)
        return value * scale + mean


def compute_object_state_normalization(
    episodes: tuple[TemporalSignalEpisode, ...],
) -> ObjectStateNormalization:
    if (
        type(episodes) is not tuple
        or not episodes
        or any(not isinstance(value, TemporalSignalEpisode) for value in episodes)
    ):
        raise ValueError("episodes must contain temporal signal episodes")
    states = np.concatenate(
        tuple(
            np.concatenate((episode.object_position, episode.object_linear_velocity), axis=1)
            for episode in episodes
        ),
        axis=0,
    ).astype(np.float64)
    scale = states.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return ObjectStateNormalization(
        mean=states.mean(axis=0).astype(np.float32),
        scale=scale.astype(np.float32),
    )


class ObjectStateReadoutDataset(torch.utils.data.Dataset):
    """Each source boundary once: full dual-view GT latent to normalized object state."""

    def __init__(
        self,
        *,
        records: tuple[JepaEpisodeRecord, ...],
        episodes: Mapping[str, TemporalSignalEpisode],
        normalization: ObjectStateNormalization,
    ) -> None:
        if (
            type(records) is not tuple
            or not records
            or any(not isinstance(value, JepaEpisodeRecord) for value in records)
            or len({value.episode_id for value in records}) != len(records)
        ):
            raise ValueError("records must contain unique JEPA episodes")
        if not isinstance(episodes, Mapping) or set(episodes) != {
            value.episode_id for value in records
        }:
            raise ValueError("signal episodes must exactly cover the JEPA records")
        if not isinstance(normalization, ObjectStateNormalization):
            raise TypeError("normalization must be ObjectStateNormalization")
        self.records = records
        self.episodes = episodes
        self.normalization = normalization
        self.indices = tuple(
            (record_index, tick)
            for record_index, record in enumerate(records)
            for tick in range(record.terminal_tick + 1)
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        record_index, tick = self.indices[item]
        record = self.records[record_index]
        episode = self.episodes[record.episode_id]
        latent = torch.from_numpy(np.array(record.cache.features[tick], copy=True)).to(
            torch.float16
        )
        state = torch.from_numpy(
            np.concatenate(
                (
                    episode.object_position[tick],
                    episode.object_linear_velocity[tick],
                )
            ).astype(np.float32)
        )
        return latent, self.normalization.normalize(state)


class DualViewObjectStateReadout(torch.nn.Module):
    """JEPA-WM StateReadoutViT over both full-patch views, followed by view fusion."""

    def __init__(self, *, project_root: Path) -> None:
        super().__init__()
        root = Path(project_root).resolve()
        reference = load_upstream_reference(
            root / "configs/belief/action_conditioned_jepa/upstream_reference.yaml"
        )
        checkout = verify_upstream_checkout(reference=reference, project_root=root)
        module = importlib.import_module("app.plan_common.models.state_decoder")
        module_path = Path(module.__file__).resolve()
        try:
            module_path.relative_to(checkout.root)
        except ValueError as error:
            raise ValueError(
                "JEPA-WM StateReadoutViT resolved outside the verified checkout"
            ) from error
        self.view_readout = module.StateReadoutViT(
            grid_size=14,
            embed_dim=384,
            decoder_embed_dim=384,
            depth=6,
            num_heads=16,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            state_dim=6,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            use_rope=False,
            use_camera_embed=False,
        )
        self.view_fusion = torch.nn.Linear(12, 6)

    def forward(self, future_visual_latents: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(future_visual_latents, torch.Tensor)
            or future_visual_latents.ndim != 5
            or tuple(future_visual_latents.shape[2:]) != (2, 196, 384)
            or not bool(torch.isfinite(future_visual_latents).all())
        ):
            raise ValueError("future visual latents must have shape [B,D,2,196,384]")
        batch_size, delay_count = future_visual_latents.shape[:2]
        features = (
            future_visual_latents.detach()
            .reshape(batch_size * delay_count * 2, 1, 1, 14, 14, 384)
            .to(dtype=self.view_readout.decoder_embed.weight.dtype)
        )
        per_view = self.view_readout(features, None).reshape(
            batch_size,
            delay_count,
            12,
        )
        return self.view_fusion(per_view)
