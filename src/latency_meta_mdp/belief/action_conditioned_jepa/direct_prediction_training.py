"""Direct endpoint supervision; all targets are real frozen-feature/state observations."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from latency_meta_mdp.belief.action_conditioned_jepa.direct_prediction import DIRECT_ARCHITECTURE_ID


@dataclass(frozen=True)
class DirectTrainingConfig:
    """Candidate budget in sampled real pairs, not equivalent legacy AR epochs."""

    architecture_id: str = DIRECT_ARCHITECTURE_ID
    max_epochs: int = 75
    examples_per_epoch: int = 44800
    global_batch_size: int = 256
    learning_rate: float = 5e-4
    weight_decay_start: float = 1e-7
    weight_decay_final: float = 1e-6
    gradient_clip_norm: float = 1.0
    seed: int = 27

    def __post_init__(self) -> None:
        if self.architecture_id != DIRECT_ARCHITECTURE_ID:
            raise ValueError("incompatible direct training architecture")
        for name in ("max_epochs", "examples_per_epoch", "global_batch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.examples_per_epoch % 20 or self.examples_per_epoch % self.global_batch_size:
            raise ValueError("epoch exposure must balance twenty queries and whole optimizer steps")
        for name in (
            "learning_rate",
            "weight_decay_start",
            "weight_decay_final",
            "gradient_clip_norm",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not 0 < value < float("inf"):
                raise ValueError(f"{name} must be positive and finite")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")

    @property
    def optimizer_steps_per_epoch(self) -> int:
        return self.examples_per_epoch // self.global_batch_size


def direct_prediction_loss(
    prediction: tuple[torch.Tensor, torch.Tensor],
    target_visual: torch.Tensor,
    target_proprio: torch.Tensor,
    *,
    proprio_mean: torch.Tensor,
    proprio_scale: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    visual, normalized_proprio = prediction
    if visual.shape != target_visual.shape or normalized_proprio.shape != target_proprio.shape:
        raise ValueError("direct prediction and target shapes must agree")
    normalized_target = (target_proprio.detach().float() - proprio_mean) / proprio_scale
    visual_loss = (visual.float() - target_visual.detach().float()).square().mean()
    proprio_loss = (normalized_proprio.float() - normalized_target).square().mean()
    return visual_loss + proprio_loss, {
        "visual_mse": visual_loss.detach(),
        "normalized_proprio_mse": proprio_loss.detach(),
    }


def train_direct_epoch(
    model, loader, optimizer, *, config: DirectTrainingConfig, device: torch.device, on_update=None
) -> dict:
    """Accumulate complete logical batches; return measured exposure and losses.

    Dataset sampling/epoch seeds and checkpoint lifetime are owned by the caller.
    No validation data, synthetic terminal states or AR targets are accessed here.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    seen = np.zeros(21, dtype=np.int64)
    updates, group_metrics = [], []
    in_group = 0
    started = time.perf_counter()
    update_started = started
    for sample in loader:
        batch = sample.query.query_ticks.numel()
        if config.global_batch_size % batch or in_group + batch > config.global_batch_size:
            raise ValueError("microbatches must partition complete logical batches")
        seen += np.bincount(sample.query.query_ticks.numpy(), minlength=21)
        query = sample.query.to(device)
        query.validate_finite()
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            loss, metrics = direct_prediction_loss(
                model(query),
                sample.target_visual.to(device),
                sample.target_proprio.to(device),
                proprio_mean=model.trunk.proprio_mean,
                proprio_scale=model.trunk.proprio_scale,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite direct training loss")
        fraction = batch / config.global_batch_size
        (fraction * loss).backward()
        group_metrics.append({key: value.item() * fraction for key, value in metrics.items()})
        in_group += batch
        if in_group == config.global_batch_size:
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_norm, error_if_nonfinite=True
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            now = time.perf_counter()
            row = {
                "optimizer_step_in_epoch": len(updates) + 1,
                "gradient_norm_before_clip": float(norm),
                "seconds": now - update_started,
                **{key: sum(item[key] for item in group_metrics) for key in group_metrics[0]},
            }
            updates.append(row)
            if on_update is not None:
                on_update(row)
            group_metrics.clear()
            in_group = 0
            update_started = now
    if in_group or not updates:
        raise ValueError("epoch must finish with a nonempty whole number of logical batches")
    return {
        "examples_seen": int(seen.sum()),
        "seen_by_query": seen.tolist(),
        "updates": updates,
        "seconds": time.perf_counter() - started,
    }
