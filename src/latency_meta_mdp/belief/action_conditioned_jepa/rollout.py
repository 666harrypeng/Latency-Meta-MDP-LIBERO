"""Dual-view, action-conditioned JEPA predictor and stationary AR20 rollout."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from latency_meta_mdp.belief.action_conditioned_jepa.config import (
    ActionConditionedJepaConfig,
)
from latency_meta_mdp.belief.action_conditioned_jepa.contracts import (
    FutureLatentRollout,
    LaunchContextBatch,
)
from latency_meta_mdp.belief.action_conditioned_jepa.data_adapter import (
    JepaProprioNormalization,
)
from latency_meta_mdp.belief.action_conditioned_jepa.upstream_adapter import (
    UpstreamPrimitiveBundle,
    adapt_upstream_block_for_coordinates,
    build_dual_view_upstream_types,
    load_upstream_primitives,
)


def build_spatiotemporal_coordinates(
    *,
    history_ticks: int,
    patch_grid: tuple[int, int],
    view_count: int,
    include_proprio: bool,
) -> torch.Tensor:
    """Return explicit time/y/x coordinates with spatial positions restarted per view."""

    height, width = patch_grid
    values = (history_ticks, height, width, view_count)
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("coordinate dimensions must be positive integers")
    spatial = torch.stack(
        torch.meshgrid(
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)
    per_time = []
    for time in range(history_ticks):
        view_coordinates = [
            torch.cat(
                (
                    torch.full((height * width, 1), float(time)),
                    spatial,
                ),
                dim=1,
            )
            for _ in range(view_count)
        ]
        if include_proprio:
            view_coordinates.append(torch.tensor([[float(time), 0.0, 0.0]]))
        per_time.append(torch.cat(view_coordinates, dim=0))
    return torch.stack(per_time, dim=0)


def build_temporal_block_causal_mask(
    *,
    history_ticks: int,
    tokens_per_time: int,
    temporal_window: int,
) -> torch.Tensor:
    """Allow every token to read its same-time block and bounded past blocks only."""

    values = (history_ticks, tokens_per_time, temporal_window)
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("temporal mask dimensions must be positive integers")
    if temporal_window > history_ticks:
        raise ValueError("temporal window cannot exceed available history")
    token_times = torch.arange(history_ticks).repeat_interleave(tokens_per_time)
    query_time = token_times[:, None]
    key_time = token_times[None, :]
    return (key_time <= query_time) & (key_time >= query_time - temporal_window + 1)


class ActionConditionedJepaPredictor(nn.Module):
    """Pinned JEPA-WM AdaLN predictor adapted only at the dual-view token boundary."""

    rope_coordinate_names = ("time", "y", "x")
    additive_time_embedding = None
    additive_spatial_embedding = None

    def __init__(
        self,
        *,
        config: ActionConditionedJepaConfig,
        proprio_normalization: JepaProprioNormalization,
        project_root: Path,
        primitives: UpstreamPrimitiveBundle | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(config, ActionConditionedJepaConfig):
            raise TypeError("config must be an ActionConditionedJepaConfig")
        if not isinstance(proprio_normalization, JepaProprioNormalization):
            raise TypeError("proprio_normalization must be JepaProprioNormalization")
        if proprio_normalization.level != config.level:
            raise ValueError("predictor and proprio normalization levels disagree")
        if primitives is None:
            primitives = load_upstream_primitives(
                reference=config.upstream_reference,
                project_root=Path(project_root),
            )
        if not isinstance(primitives, UpstreamPrimitiveBundle):
            raise TypeError("primitives must be an UpstreamPrimitiveBundle")
        self.config = config
        self.width = config.predictor_width
        self.tokens_per_time = 2 * config.patch_token_count + 1
        self.backbone = primitives.vision_transformer_adaln_type(
            img_size=(224, 224),
            patch_size=16,
            num_frames=config.history_ticks,
            tubelet_size=1,
            embed_dim=config.visual_feature_dim,
            predictor_embed_dim=config.predictor_width,
            depth=config.predictor_depth,
            num_heads=config.predictor_heads,
            mlp_ratio=config.mlp_ratio,
            qkv_bias=config.qkv_bias,
            drop_rate=config.dropout,
            attn_drop_rate=config.attention_dropout,
            drop_path_rate=config.drop_path,
            norm_layer=partial(nn.LayerNorm, eps=config.layer_norm_epsilon),
            init_std=0.02,
            use_silu=False,
            is_causal=False,
            use_activation_checkpointing=False,
            local_window=(-1, -1, -1),
            use_rope=True,
            action_dim=config.action_dim,
            proprio_dim=config.proprio_dim,
            use_proprio=True,
            act_mlp=False,
            prop_mlp=False,
            init_scale_factor_adaln=config.adaln_init_scale_factor,
            proprio_encoding="token",
            proprio_emb_dim=0,
            proprio_encoder_inpred=True,
            proprio_tokens=1,
            action_encoder_inpred=True,
        )
        types = build_dual_view_upstream_types(primitives)
        for block in self.backbone.predictor_blocks:
            adapt_upstream_block_for_coordinates(block, types=types)
        self.backbone.attn_mask = None

        self.view_embedding = nn.Embedding(2, self.width)
        self.proprio_head = nn.Linear(self.width, config.proprio_dim, bias=True)
        with torch.no_grad():
            self.view_embedding.weight[0].zero_()
            primitives.trunc_normal(self.view_embedding.weight[1:], std=0.02)
            primitives.trunc_normal(self.proprio_head.weight, std=0.02)
            self.proprio_head.bias.zero_()
        self.register_buffer(
            "proprio_mean",
            torch.from_numpy(np.array(proprio_normalization.mean, copy=True)),
        )
        self.register_buffer(
            "proprio_scale",
            torch.from_numpy(np.array(proprio_normalization.scale, copy=True)),
        )
        self.register_buffer(
            "coordinates",
            build_spatiotemporal_coordinates(
                history_ticks=config.history_ticks,
                patch_grid=config.vision_encoder.patch_grid,
                view_count=2,
                include_proprio=True,
            ),
            persistent=False,
        )
        self.register_buffer(
            "attention_mask",
            build_temporal_block_causal_mask(
                history_ticks=config.history_ticks,
                tokens_per_time=self.tokens_per_time,
                temporal_window=config.temporal_attention_window,
            ),
            persistent=False,
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _model_device(self) -> torch.device:
        return self.backbone.predictor_embed.weight.device

    def _validate_step_inputs(
        self,
        *,
        vision_history: torch.Tensor,
        proprio_history: torch.Tensor,
        executed_controls: torch.Tensor,
        outgoing_control: torch.Tensor,
    ) -> int:
        if not all(
            isinstance(value, torch.Tensor)
            for value in (
                vision_history,
                proprio_history,
                executed_controls,
                outgoing_control,
            )
        ):
            raise TypeError("predict_next inputs must be torch tensors")
        batch_size = vision_history.shape[0]
        expected = (
            tuple(vision_history.shape) == (batch_size, 6, 2, 196, 384),
            tuple(proprio_history.shape) == (batch_size, 6, 16),
            tuple(executed_controls.shape) == (batch_size, 5, 7),
            tuple(outgoing_control.shape) == (batch_size, 7),
        )
        if batch_size <= 0 or not all(expected):
            raise ValueError("predict_next tensor shapes do not match K6/D20 semantics")
        devices = {
            vision_history.device,
            proprio_history.device,
            executed_controls.device,
            outgoing_control.device,
            self._model_device(),
        }
        if len(devices) != 1:
            raise ValueError("predict_next inputs and model must use one device")
        return batch_size

    def _build_tokens(
        self,
        *,
        vision_history: torch.Tensor,
        proprio_history: torch.Tensor,
    ) -> torch.Tensor:
        model_dtype = self.backbone.predictor_embed.weight.dtype
        vision = self.backbone.predictor_embed(vision_history.to(dtype=model_dtype))
        view_identity = self.view_embedding.weight.to(dtype=vision.dtype)
        vision = vision + view_identity[None, None, :, None, :]
        vision = vision.flatten(2, 3)
        proprio = self.backbone.proprio_encoder(proprio_history.to(dtype=model_dtype)).unsqueeze(2)
        return torch.cat((vision, proprio), dim=2)

    def build_context_tokens(
        self,
        context: LaunchContextBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(context, LaunchContextBatch):
            raise TypeError("context must be a LaunchContextBatch")
        if context.device != self._model_device():
            raise ValueError("context and predictor must use one device")
        return (
            self._build_tokens(
                vision_history=context.vision_history,
                proprio_history=context.proprio_history,
            ),
            self.coordinates,
        )

    def _encode_step(
        self,
        *,
        vision_history: torch.Tensor,
        proprio_history: torch.Tensor,
        executed_controls: torch.Tensor,
        outgoing_control: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = self._validate_step_inputs(
            vision_history=vision_history,
            proprio_history=proprio_history,
            executed_controls=executed_controls,
            outgoing_control=outgoing_control,
        )
        tokens = self._build_tokens(
            vision_history=vision_history,
            proprio_history=proprio_history,
        ).flatten(1, 2)
        controls = torch.cat((executed_controls, outgoing_control.unsqueeze(1)), dim=1)
        conditioning = self.backbone.action_encoder(
            controls.to(dtype=self.backbone.action_encoder.weight.dtype)
        )
        for block in self.backbone.predictor_blocks:
            tokens = block(
                tokens,
                conditioning,
                coordinates=self.coordinates,
                attn_mask=self.attention_mask,
                tokens_per_time=self.tokens_per_time,
            )
        return self.backbone.predictor_norm(tokens).reshape(
            batch_size,
            self.config.history_ticks,
            self.tokens_per_time,
            self.width,
        )

    def encode_context(self, context: LaunchContextBatch) -> torch.Tensor:
        if not isinstance(context, LaunchContextBatch):
            raise TypeError("context must be a LaunchContextBatch")
        return self._encode_step(
            vision_history=context.vision_history,
            proprio_history=context.proprio_history,
            executed_controls=context.executed_controls,
            outgoing_control=context.executable_controls[:, 0],
        )

    def forward_single_view_compat(
        self,
        vision_history: torch.Tensor,
        controls: torch.Tensor,
    ) -> torch.Tensor:
        """Run the project blocks on the exact upstream single-view/no-proprio topology."""

        batch_size = vision_history.shape[0]
        if tuple(vision_history.shape) != (
            batch_size,
            self.config.history_ticks,
            1,
            14,
            14,
            self.config.visual_feature_dim,
        ) or tuple(controls.shape) != (
            batch_size,
            self.config.history_ticks,
            self.config.action_dim,
        ):
            raise ValueError("single-view compatibility tensors have invalid shapes")
        devices = {
            vision_history.device,
            controls.device,
            self._model_device(),
        }
        if len(devices) != 1:
            raise ValueError("single-view compatibility tensors must share the model device")
        model_dtype = self.backbone.predictor_embed.weight.dtype
        tokens = self.backbone.predictor_embed(vision_history.to(dtype=model_dtype)).flatten(2, 4)
        tokens = tokens + self.view_embedding.weight[0].to(dtype=tokens.dtype)
        coordinates = build_spatiotemporal_coordinates(
            history_ticks=self.config.history_ticks,
            patch_grid=(14, 14),
            view_count=1,
            include_proprio=False,
        ).to(device=tokens.device)
        attention_mask = build_temporal_block_causal_mask(
            history_ticks=self.config.history_ticks,
            tokens_per_time=196,
            temporal_window=self.config.history_ticks,
        ).to(device=tokens.device)
        conditioning = self.backbone.action_encoder(
            controls.to(dtype=self.backbone.action_encoder.weight.dtype)
        )
        tokens = tokens.flatten(1, 2)
        for block in self.backbone.predictor_blocks:
            tokens = block(
                tokens,
                conditioning,
                coordinates=coordinates,
                attn_mask=attention_mask,
                tokens_per_time=196,
            )
        tokens = self.backbone.predictor_norm(tokens).reshape(
            batch_size,
            self.config.history_ticks,
            196,
            self.width,
        )
        return self.backbone.predictor_proj(tokens)

    def predict_next(
        self,
        *,
        vision_history: torch.Tensor,
        proprio_history: torch.Tensor,
        executed_controls: torch.Tensor,
        outgoing_control: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self._encode_step(
            vision_history=vision_history,
            proprio_history=proprio_history,
            executed_controls=executed_controls,
            outgoing_control=outgoing_control,
        )[:, -1]
        visual_hidden = hidden[:, : 2 * self.config.patch_token_count].reshape(
            hidden.shape[0],
            2,
            self.config.patch_token_count,
            self.width,
        )
        visual = self.backbone.predictor_proj(visual_hidden)
        proprio = self.proprio_head(hidden[:, -1])
        return visual, proprio

    @staticmethod
    def _advance_context(
        *,
        vision_history: torch.Tensor,
        proprio_history: torch.Tensor,
        executed_controls: torch.Tensor,
        outgoing_control: torch.Tensor,
        next_visual: torch.Tensor,
        next_proprio: torch.Tensor,
        detach_prediction: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if detach_prediction:
            next_visual = next_visual.detach()
            next_proprio = next_proprio.detach()
        return (
            torch.cat((vision_history[:, 1:], next_visual.unsqueeze(1)), dim=1),
            torch.cat((proprio_history[:, 1:], next_proprio.unsqueeze(1)), dim=1),
            torch.cat((executed_controls[:, 1:], outgoing_control.unsqueeze(1)), dim=1),
        )

    def rollout_endpoint_for_loss(
        self,
        context: LaunchContextBatch,
        horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if type(horizon) is not int or not 1 <= horizon <= self.config.maximum_delay_ticks:
            raise ValueError("rollout horizon must lie in 1..20")
        vision = context.vision_history
        proprio = context.proprio_history
        executed = context.executed_controls
        for step in range(horizon):
            outgoing = context.executable_controls[:, step]
            next_visual, next_proprio = self.predict_next(
                vision_history=vision,
                proprio_history=proprio,
                executed_controls=executed,
                outgoing_control=outgoing,
            )
            if step + 1 == horizon:
                return next_visual, next_proprio
            vision, proprio, executed = self._advance_context(
                vision_history=vision,
                proprio_history=proprio,
                executed_controls=executed,
                outgoing_control=outgoing,
                next_visual=next_visual,
                next_proprio=next_proprio,
                detach_prediction=True,
            )
        raise AssertionError("unreachable rollout horizon")

    @torch.no_grad()
    def rollout_d20(self, context: LaunchContextBatch) -> FutureLatentRollout:
        if not isinstance(context, LaunchContextBatch):
            raise TypeError("context must be a LaunchContextBatch")
        vision = context.vision_history
        proprio = context.proprio_history
        executed = context.executed_controls
        visual_futures = []
        proprio_futures = []
        with torch.autocast(
            device_type=context.device.type,
            dtype=torch.bfloat16,
            enabled=context.device.type == "cuda",
        ):
            for step in range(self.config.maximum_delay_ticks):
                outgoing = context.executable_controls[:, step]
                next_visual, next_proprio = self.predict_next(
                    vision_history=vision,
                    proprio_history=proprio,
                    executed_controls=executed,
                    outgoing_control=outgoing,
                )
                visual_futures.append(next_visual)
                proprio_futures.append(next_proprio)
                vision, proprio, executed = self._advance_context(
                    vision_history=vision,
                    proprio_history=proprio,
                    executed_controls=executed,
                    outgoing_control=outgoing,
                    next_visual=next_visual,
                    next_proprio=next_proprio,
                    detach_prediction=True,
                )
        visual = torch.stack(visual_futures, dim=1).to(torch.float16)
        normalized_proprio = torch.stack(proprio_futures, dim=1).to(torch.float32)
        physical_proprio = (
            normalized_proprio * self.proprio_scale[None, None] + self.proprio_mean[None, None]
        ).to(torch.float32)
        return FutureLatentRollout(
            future_visual_latents=visual,
            future_proprio=physical_proprio,
        )
