"""One-pass queried JEPA endpoint, reusing the pinned dual-view AdaLN backbone."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save
from torch import nn

from latency_meta_mdp.belief.jepa.backbone import ActionConditionedJepaPredictor
from latency_meta_mdp.belief.jepa.config import ActionConditionedJepaConfig
from latency_meta_mdp.belief.jepa.contracts import (
    ForecastQuery,
    FutureLatentPrediction,
)
from latency_meta_mdp.belief.jepa.corpus import JepaProprioNormalization

DIRECT_ARCHITECTURE_ID = "jepa_direct_q20_history_stride4_w3_v1"


class DirectJepaPredictor(nn.Module):
    """Full spatial tokens, ordered masked controls, and a discrete horizon embedding.

    The legacy config describes the reused trunk only, not this model's future grid or
    loss. Historical blocks retain their observed four-control transition condition;
    the current block receives the complete requested control prefix and query horizon.
    No AR API, PMF, target image or absolute episode time enters the learned mapping.
    """

    architecture_id = DIRECT_ARCHITECTURE_ID

    def __init__(
        self,
        *,
        backbone_config: ActionConditionedJepaConfig,
        proprio_normalization: JepaProprioNormalization,
        project_root: Path,
    ) -> None:
        super().__init__()
        if (backbone_config.history_ticks, backbone_config.model_stride_ticks) != (3, 4):
            raise ValueError("direct predictor requires three observations with stride4 history")
        self.trunk = ActionConditionedJepaPredictor(
            config=backbone_config,
            proprio_normalization=proprio_normalization,
            project_root=project_root,
        )
        # Flattening preserves each control's position; the explicit mask distinguishes
        # a real zero/hold command from padding. Mask before projection (0 * NaN is NaN).
        self.control_prefix_encoder = nn.Linear(20 * 7 + 20, self.trunk.width)
        self.query_embedding = nn.Embedding(21, self.trunk.width)
        nn.init.normal_(self.query_embedding.weight, std=0.02)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, query: ForecastQuery) -> tuple[torch.Tensor, torch.Tensor]:
        """Training outputs: DINO features and normalized future 16D proprio."""
        if not isinstance(query, ForecastQuery):
            raise TypeError("direct predictor requires ForecastQuery")
        if query.vision_history.device != self.trunk.proprio_mean.device:
            raise ValueError("query and predictor must use the same device")
        current_visual = query.vision_history[:, -1].float()
        current_proprio = query.proprio_history[:, -1]
        if bool((query.query_ticks == 0).all()):
            return current_visual, current_proprio
        tokens = self.trunk._build_tokens(
            vision_history=query.vision_history,
            proprio_history=query.proprio_history,
        ).flatten(1, 2)
        dtype = self.control_prefix_encoder.weight.dtype
        controls = query.executable_controls.masked_fill(~query.control_mask[..., None], 0)
        prefix = torch.cat((controls.flatten(1), query.control_mask.to(dtype)), dim=1)
        future_condition = self.control_prefix_encoder(prefix.to(dtype)) + self.query_embedding(
            query.query_ticks
        )
        past_condition = self.trunk.backbone.action_encoder(
            query.executed_controls.flatten(2).to(dtype)
        )
        conditioning = torch.cat((past_condition, future_condition[:, None]), dim=1)
        for block in self.trunk.backbone.predictor_blocks:
            tokens = block(
                tokens,
                conditioning,
                coordinates=self.trunk.coordinates,
                attn_mask=self.trunk.attention_mask,
                tokens_per_time=self.trunk.tokens_per_time,
            )
        endpoint = self.trunk.backbone.predictor_norm(tokens)[:, -self.trunk.tokens_per_time :]
        visual = (
            self.trunk.backbone.predictor_proj(endpoint[:, :-1]).reshape(-1, 2, 196, 384).float()
        )
        proprio = self.trunk.proprio_head(endpoint[:, -1]).float()
        zero = query.query_ticks == 0
        return (
            torch.where(zero[:, None, None, None], current_visual, visual),
            torch.where(zero[:, None], current_proprio, proprio),
        )

    @torch.no_grad()
    def predict_at(self, query: ForecastQuery) -> FutureLatentPrediction:
        visual, proprio = self(query)
        return FutureLatentPrediction(
            source_ticks=query.source_ticks.clone(),
            target_ticks=query.source_ticks + query.query_ticks,
            visual_latents=visual.half(),
            proprio=proprio * self.trunk.proprio_scale + self.trunk.proprio_mean,
        )


def save_direct_prediction_weights(model: DirectJepaPredictor, path: Path) -> None:
    """Immutable inference weights with an explicit incompatible-architecture guard."""
    metadata = {"architecture_id": model.architecture_id, "level": str(model.trunk.config.level)}
    if model.trunk.config.task_id is not None:
        metadata = {
            "architecture_id": model.architecture_id,
            "task_id": model.trunk.config.task_id,
            "action_contract_id": model.trunk.config.action_contract.contract_id,
        }
    tensors = {
        key: tensor.detach().cpu().contiguous() for key, tensor in model.state_dict().items()
    }
    payload = save(tensors, metadata=metadata)
    with Path(path).open("xb") as stream:
        stream.write(payload)


def load_direct_prediction_weights(model: DirectJepaPredictor, path: Path) -> None:
    with safe_open(str(path), framework="pt", device="cpu") as stream:
        metadata = stream.metadata() or {}
    if metadata.get("architecture_id") != model.architecture_id:
        raise ValueError("incompatible direct predictor architecture")
    if model.trunk.config.task_id is None:
        if metadata.get("level") != str(model.trunk.config.level) or "task_id" in metadata:
            raise ValueError("direct predictor checkpoint level mismatch")
    elif (
        metadata.get("task_id") != model.trunk.config.task_id
        or metadata.get("action_contract_id") != model.trunk.config.action_contract.contract_id
        or "level" in metadata
    ):
        raise ValueError("direct predictor checkpoint task/controller mismatch")
    state = load_file(str(path))
    for name in ("proprio_mean", "proprio_scale"):
        if not torch.equal(state[f"trunk.{name}"], getattr(model.trunk, name).cpu()):
            raise ValueError("direct predictor checkpoint normalization mismatch")
    model.load_state_dict(state, strict=True)
