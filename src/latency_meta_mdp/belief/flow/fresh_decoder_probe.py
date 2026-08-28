"""Frozen-Encoder token caching for Decoder-independent sufficiency probes."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from latency_meta_mdp.belief.flow.config import FlowBeliefConfig
from latency_meta_mdp.belief.flow.model import FlowBeliefModel
from latency_meta_mdp.belief.flow.training_data import FlowBeliefNormalization
from latency_meta_mdp.belief.flow.vector_field import ConditionalStateVectorField
from latency_meta_mdp.vision_probe_data import ProbeSplit


def _readonly(value: Any, *, dtype: Any) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class CachedBeliefSplit:
    belief_tokens: np.ndarray
    latency_probabilities: np.ndarray
    target_states: np.ndarray
    interaction_mode: np.ndarray
    absorbing: np.ndarray
    episode_ids: tuple[str, ...]
    source_ticks: np.ndarray

    def __post_init__(self) -> None:
        tokens = np.asarray(self.belief_tokens)
        if tokens.ndim != 3 or tokens.shape[1:] != (8, 192):
            raise ValueError("cached belief tokens have invalid shape")
        count = len(tokens)
        expected = {
            "latency_probabilities": (count, 20),
            "target_states": (count, 20, 22),
            "interaction_mode": (count, 20),
            "absorbing": (count, 20),
            "source_ticks": (count,),
        }
        for name, shape in expected.items():
            if np.asarray(getattr(self, name)).shape != shape:
                raise ValueError(f"cached belief {name} has invalid shape")
        episode_ids = tuple(self.episode_ids)
        if len(episode_ids) != count or any(not value for value in episode_ids):
            raise ValueError("cached belief episode identities are invalid")
        if any(
            not np.all(np.isfinite(value))
            for value in (tokens, self.latency_probabilities, self.target_states)
        ):
            raise ValueError("cached belief arrays must be finite")
        probabilities = np.asarray(self.latency_probabilities, dtype=np.float64)
        if np.any(probabilities < 0.0) or not np.allclose(
            probabilities.sum(axis=1),
            1.0,
            atol=1e-12,
            rtol=0,
        ):
            raise ValueError("cached belief latency probabilities are invalid")
        if np.any(np.asarray(self.source_ticks) < 0):
            raise ValueError("cached belief source ticks are invalid")
        object.__setattr__(self, "episode_ids", episode_ids)
        for name, dtype in (
            ("belief_tokens", np.float32),
            ("latency_probabilities", np.float64),
            ("target_states", np.float32),
            ("interaction_mode", np.int8),
            ("absorbing", np.bool_),
            ("source_ticks", np.int64),
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name), dtype=dtype))


class FrozenEncoderFreshDecoder(nn.Module):
    """Copy a trained Encoder exactly and attach a newly initialized vector field."""

    def __init__(
        self,
        *,
        encoder: nn.Module,
        fresh_decoder: ConditionalStateVectorField,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.fresh_decoder = fresh_decoder
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    @classmethod
    def from_source_model(
        cls,
        *,
        source_model: FlowBeliefModel,
        config: FlowBeliefConfig,
        decoder_seed: int,
    ) -> FrozenEncoderFreshDecoder:
        if not isinstance(source_model, FlowBeliefModel) or not isinstance(
            config, FlowBeliefConfig
        ):
            raise TypeError("fresh Decoder probe requires typed Flow model and config")
        if isinstance(decoder_seed, bool) or not isinstance(decoder_seed, int) or decoder_seed < 0:
            raise ValueError("fresh Decoder seed must be a non-negative integer")
        encoder = copy.deepcopy(source_model.encoder)
        device = next(encoder.parameters()).device
        cuda_devices = (
            [device.index or torch.cuda.current_device()] if device.type == "cuda" else []
        )
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(decoder_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(decoder_seed)
            fresh_decoder = ConditionalStateVectorField(config).to(device)
        return cls(encoder=encoder, fresh_decoder=fresh_decoder)

    def train(self, mode: bool = True) -> FrozenEncoderFreshDecoder:
        super().train(mode)
        self.encoder.eval()
        return self

    def encode(
        self,
        *,
        vision_history: torch.Tensor,
        proprio_history: torch.Tensor,
        remaining_actions: torch.Tensor,
        latency_probabilities: torch.Tensor,
    ) -> torch.Tensor:
        with torch.inference_mode():
            return self.encoder(
                vision_history=vision_history,
                proprio_history=proprio_history,
                remaining_actions=remaining_actions,
                latency_probabilities=latency_probabilities,
            )

    def decode(
        self,
        *,
        noisy_state: torch.Tensor,
        flow_time: torch.Tensor,
        belief_tokens: torch.Tensor,
        delay_ticks: torch.Tensor,
    ) -> torch.Tensor:
        return self.fresh_decoder(
            noisy_state=noisy_state,
            flow_time=flow_time,
            belief_tokens=belief_tokens,
            delay_ticks=delay_ticks,
        )


def cache_frozen_belief_splits(
    *,
    corpus: Any,
    probe: FrozenEncoderFreshDecoder,
    normalization: FlowBeliefNormalization,
    splits: tuple[ProbeSplit, ...],
    batch_size: int,
    device: str,
    context_limits: Mapping[ProbeSplit, int] | None = None,
) -> dict[ProbeSplit, CachedBeliefSplit]:
    if (
        not splits
        or splits != tuple(dict.fromkeys(splits))
        or any(not isinstance(split, ProbeSplit) for split in splits)
    ):
        raise ValueError("cached belief splits must be unique typed values")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("cached belief batch size must be positive")
    limits = dict(context_limits or {})
    if any(
        not isinstance(split, ProbeSplit)
        or isinstance(limit, bool)
        or not isinstance(limit, int)
        or limit <= 0
        for split, limit in limits.items()
    ):
        raise ValueError("cached belief context limits are invalid")
    result = {}
    for split in splits:
        references = corpus.sample_references[split]
        selected_count = min(len(references), limits.get(split, len(references)))
        token_batches = []
        probabilities = []
        targets = []
        modes = []
        absorbing = []
        episode_ids = []
        source_ticks = []
        for start in range(0, selected_count, batch_size):
            contexts = [
                corpus.materialize(split, offset)
                for offset in range(start, min(start + batch_size, selected_count))
            ]
            vision = np.stack([context.vision_history for context in contexts])
            proprio = np.stack(
                [
                    (context.robot_proprio_history - normalization.proprio_mean)
                    / normalization.proprio_std
                    for context in contexts
                ]
            ).astype(np.float32)
            actions = np.stack(
                [
                    (context.remaining_actions - normalization.action_mean)
                    / normalization.action_std
                    for context in contexts
                ]
            ).astype(np.float32)
            latency = np.stack([context.latency_probabilities for context in contexts]).astype(
                np.float32
            )
            with torch.inference_mode():
                belief = probe.encode(
                    vision_history=torch.from_numpy(vision).to(device),
                    proprio_history=torch.from_numpy(proprio).to(device),
                    remaining_actions=torch.from_numpy(actions).to(device),
                    latency_probabilities=torch.from_numpy(latency).to(device),
                )
            token_batches.append(belief.detach().cpu().numpy())
            probabilities.extend(context.latency_probabilities for context in contexts)
            targets.extend(
                (context.target_states - normalization.target_mean) / normalization.target_std
                for context in contexts
            )
            modes.extend(context.target_interaction_mode for context in contexts)
            absorbing.extend(context.target_absorbing for context in contexts)
            episode_ids.extend(context.episode_id for context in contexts)
            source_ticks.extend(context.source_tick for context in contexts)
        if token_batches:
            tokens = np.concatenate(token_batches, axis=0)
            probability_array = np.stack(probabilities)
            target_array = np.stack(targets)
            mode_array = np.stack(modes)
            absorbing_array = np.stack(absorbing)
        else:
            tokens = np.empty((0, 8, 192), dtype=np.float32)
            probability_array = np.empty((0, 20), dtype=np.float64)
            target_array = np.empty((0, 20, 22), dtype=np.float32)
            mode_array = np.empty((0, 20), dtype=np.int8)
            absorbing_array = np.empty((0, 20), dtype=np.bool_)
        result[split] = CachedBeliefSplit(
            belief_tokens=tokens,
            latency_probabilities=probability_array,
            target_states=target_array,
            interaction_mode=mode_array,
            absorbing=absorbing_array,
            episode_ids=tuple(episode_ids),
            source_ticks=np.asarray(source_ticks, dtype=np.int64),
        )
    return result
