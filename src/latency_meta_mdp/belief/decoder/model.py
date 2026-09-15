"""Independent RGB reconstruction of frozen, full-grid DINO/JEPA visual features."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class VisualDecoderConfig:
    width: int = 384
    depth: int = 4
    heads: int = 6
    mlp_ratio: int = 4

    def __post_init__(self):
        values = (self.width, self.depth, self.heads, self.mlp_ratio)
        if any(type(x) is not int or x <= 0 for x in values):
            raise ValueError("decoder dimensions must be positive integers")
        if self.width < 384 or self.width % self.heads:
            raise ValueError("decoder must preserve feature width and have divisible heads")


def unpatchify_rgb(patches: torch.Tensor) -> torch.Tensor:
    """Invert row-major 14x14 patches, each containing 16x16 RGB pixels."""
    if patches.shape[-2:] != (196, 768):
        raise ValueError("RGB patch predictions must end in [196,768]")
    leading = patches.shape[:-2]
    n = len(leading)
    pixels = patches.reshape(*leading, 14, 14, 16, 16, 3)
    return pixels.permute(*range(n), n + 4, n, n + 2, n + 1, n + 3).reshape(*leading, 3, 224, 224)


class DualViewVisualDecoder(nn.Module):
    """Shared per-view decoder; no spatial resampling and no gradients into Belief.

    Input ends in [2,196,384]; arbitrary batch/anchor dimensions are retained.
    Output ends in [2,3,224,224], continuous float32 RGB in [0,1].
    """

    def __init__(self, config: VisualDecoderConfig = VisualDecoderConfig()):
        super().__init__()
        self.config = config
        self.input_norm = nn.LayerNorm(384)
        self.input_projection = nn.Linear(384, config.width)
        self.position = nn.Parameter(torch.empty(196, config.width))
        self.camera = nn.Parameter(torch.empty(2, config.width))
        nn.init.normal_(self.position, std=0.02)
        nn.init.normal_(self.camera, std=0.02)
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=config.width,
                nhead=config.heads,
                dim_feedforward=config.width * config.mlp_ratio,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(config.depth)
        )
        self.output_norm = nn.LayerNorm(config.width)
        self.rgb_head = nn.Linear(config.width, 768)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.shape[-3:] != (2, 196, 384) or not latents.is_floating_point():
            raise ValueError("visual decoder input must end in floating [2,196,384]")
        leading = latents.shape[:-3]
        z = latents.detach().float().reshape(-1, 2, 196, 384)
        z = self.input_projection(self.input_norm(z))
        z = z + self.position[None, None] + self.camera[None, :, None]
        z = z.reshape(-1, 196, self.config.width)
        for block in self.blocks:
            z = block(z)
        patches = self.rgb_head(self.output_norm(z)).float().sigmoid()
        return unpatchify_rgb(patches).reshape(*leading, 2, 3, 224, 224)


def visual_reconstruction_loss(
    predicted: torch.Tensor, target: torch.Tensor, *, edge_weight: float = 0.1
) -> dict[str, torch.Tensor]:
    if predicted.shape != target.shape or predicted.shape[-3:] != (3, 224, 224):
        raise ValueError("reconstruction requires matched RGB224 tensors")
    if edge_weight < 0:
        raise ValueError("edge weight must be nonnegative")
    predicted, target = predicted.float(), target.detach().float()
    pixel = F.l1_loss(predicted, target)
    edge = 0.5 * sum(
        F.l1_loss(torch.diff(predicted, dim=axis), torch.diff(target, dim=axis))
        for axis in (-2, -1)
    )
    return {"pixel": pixel, "edge": edge, "total": pixel + edge_weight * edge}
