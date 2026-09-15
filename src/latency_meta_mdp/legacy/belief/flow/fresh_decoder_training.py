"""Training for a new vector field over cached frozen belief tokens."""

from __future__ import annotations

import json
import math
import os
import shutil
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file as save_safetensors

from latency_meta_mdp.io.artifacts import sha256_file
from latency_meta_mdp.legacy.belief.flow.config import FlowBeliefConfig
from latency_meta_mdp.legacy.belief.flow.fresh_decoder_probe import CachedBeliefSplit
from latency_meta_mdp.legacy.belief.flow.model import build_flow_matching_batch
from latency_meta_mdp.legacy.belief.flow.vector_field import ConditionalStateVectorField
from latency_meta_mdp.legacy.vision_probe_data import ProbeSplit


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _validation_loss(
    *,
    decoder: ConditionalStateVectorField,
    cached: CachedBeliefSplit,
    config: FlowBeliefConfig,
    level: int,
    device: str,
) -> float:
    decoder.eval()
    delay_ticks = np.repeat(
        np.arange(1, 21, dtype=np.int64),
        config.validation_flow_draw_count,
    )
    total = 0.0
    count = 0
    with torch.inference_mode():
        for start in range(0, len(cached.belief_tokens), config.batch_size):
            stop = min(start + config.batch_size, len(cached.belief_tokens))
            context_count = stop - start
            targets = np.repeat(
                cached.target_states[start:stop],
                config.validation_flow_draw_count,
                axis=1,
            ).astype(np.float32)
            weights = np.repeat(
                cached.latency_probabilities[start:stop] / config.validation_flow_draw_count,
                config.validation_flow_draw_count,
                axis=1,
            ).astype(np.float32)
            rng = np.random.default_rng(config.evaluation_seed + level * 1_000_000 + start)
            noise = rng.standard_normal(targets.shape, dtype=np.float32)
            flow_time = rng.random(targets.shape[:2], dtype=np.float32)
            path = build_flow_matching_batch(
                target_state=torch.from_numpy(targets).to(device),
                noise=torch.from_numpy(noise).to(device),
                flow_time=torch.from_numpy(flow_time).to(device),
            )
            prediction = decoder(
                noisy_state=path.noisy_state,
                flow_time=path.flow_time,
                belief_tokens=torch.from_numpy(
                    np.array(cached.belief_tokens[start:stop], copy=True)
                ).to(device),
                delay_ticks=torch.from_numpy(
                    np.broadcast_to(delay_ticks, (context_count, len(delay_ticks))).copy()
                ).to(device),
            )
            query_mse = torch.mean(
                torch.square(prediction - path.target_velocity),
                dim=-1,
            )
            loss = torch.mean(torch.sum(torch.from_numpy(weights).to(device) * query_mse, dim=-1))
            total += float(loss.item()) * context_count
            count += context_count
    return total / count


def train_fresh_vector_field(
    *,
    cached: dict[ProbeSplit, CachedBeliefSplit],
    config: FlowBeliefConfig,
    level: int,
    decoder_seed: int,
    output_dir: Path,
    device: str,
) -> Path:
    if set(cached) < {ProbeSplit.TRAIN, ProbeSplit.VALIDATION}:
        raise ValueError("fresh Decoder training requires training and validation splits")
    train_split = cached[ProbeSplit.TRAIN]
    validation_split = cached[ProbeSplit.VALIDATION]
    if len(train_split.belief_tokens) == 0 or len(validation_split.belief_tokens) == 0:
        raise ValueError("fresh Decoder training and validation splits must be non-empty")
    if level not in (1, 2, 3):
        raise ValueError("fresh Decoder level must be 1, 2, or 3")
    if isinstance(decoder_seed, bool) or not isinstance(decoder_seed, int) or decoder_seed < 0:
        raise ValueError("fresh Decoder seed must be a non-negative integer")
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"fresh Decoder output already exists: {target}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for fresh Decoder training but is unavailable")
    cuda_devices = [torch.cuda.current_device()] if device.startswith("cuda") else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(decoder_seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(decoder_seed)
        decoder = ConditionalStateVectorField(config).to(device)
    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_loss = math.inf
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    history = []
    context_count = len(train_split.belief_tokens)
    for epoch in range(1, config.max_epochs + 1):
        decoder.train()
        epoch_rng = np.random.default_rng(
            config.random_seed + decoder_seed + level * 1_000_000 + epoch * 100_000
        )
        order = epoch_rng.permutation(context_count)
        total = 0.0
        count = 0
        for start in range(0, context_count, config.batch_size):
            indices = order[start : start + config.batch_size]
            query_rows = np.stack(
                [
                    epoch_rng.choice(
                        20,
                        size=config.sampled_delay_query_count,
                        replace=True,
                        p=train_split.latency_probabilities[index],
                    )
                    for index in indices
                ]
            )
            targets = np.stack(
                [
                    train_split.target_states[index, rows]
                    for index, rows in zip(indices, query_rows, strict=True)
                ]
            ).astype(np.float32)
            path = build_flow_matching_batch(
                target_state=torch.from_numpy(targets).to(device),
                noise=torch.from_numpy(
                    epoch_rng.standard_normal(targets.shape, dtype=np.float32)
                ).to(device),
                flow_time=torch.from_numpy(
                    epoch_rng.random(targets.shape[:2], dtype=np.float32)
                ).to(device),
            )
            optimizer.zero_grad(set_to_none=True)
            prediction = decoder(
                noisy_state=path.noisy_state,
                flow_time=path.flow_time,
                belief_tokens=torch.from_numpy(
                    np.array(train_split.belief_tokens[indices], copy=True)
                ).to(device),
                delay_ticks=torch.from_numpy(query_rows + 1).to(device),
            )
            loss = torch.mean(torch.square(prediction - path.target_velocity))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                decoder.parameters(),
                config.gradient_clip_norm,
            )
            optimizer.step()
            total += float(loss.item()) * len(indices)
            count += len(indices)
        training_loss = total / count
        validation_loss = _validation_loss(
            decoder=decoder,
            cached=validation_split,
            config=config,
            level=level,
            device=device,
        )
        history.append(
            {
                "epoch": epoch,
                "training_sampled_flow_mse": training_loss,
                "validation_fixed_weighted_flow_mse": validation_loss,
            }
        )
        improved = validation_loss < best_loss - config.early_stopping_min_delta
        if improved:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in decoder.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        print(
            f"[fresh-decoder][L{level}][seed={decoder_seed}] "
            f"epoch={epoch:03d}/{config.max_epochs:03d} "
            f"train_mse={training_loss:.6f} validation_mse={validation_loss:.6f} "
            f"best={best_loss:.6f} stale={stale_epochs}/{config.early_stopping_patience}",
            flush=True,
        )
        if stale_epochs >= config.early_stopping_patience:
            break
    if best_state is None:
        raise RuntimeError("fresh Decoder training produced no finite checkpoint")
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f"{target.name}.building-{uuid.uuid4().hex}"
    building.mkdir()
    try:
        save_safetensors(
            {name: value.contiguous() for name, value in best_state.items()},
            building / "vector_field.safetensors",
        )
        _write_json(
            building / "metrics.json",
            {"validation_fixed_flow_mse": best_loss},
        )
        _write_json(building / "training_history.json", history)
        artifacts = {
            name: sha256_file(building / name)
            for name in (
                "vector_field.safetensors",
                "metrics.json",
                "training_history.json",
            )
        }
        _write_json(
            building / "manifest.json",
            {
                "schema_version": 1,
                "format_id": "flow_belief_fresh_decoder_seed_v1",
                "level": level,
                "decoder_seed": decoder_seed,
                "config": asdict(config),
                "initialization": "random_vector_field_only",
                "encoder_trainable": False,
                "decoder_parameter_count": sum(
                    parameter.numel() for parameter in decoder.parameters()
                ),
                "training_context_count": len(train_split.belief_tokens),
                "validation_context_count": len(validation_split.belief_tokens),
                "epochs_completed": len(history),
                "best_epoch": best_epoch,
                "best_validation_flow_mse": best_loss,
                "artifacts": artifacts,
            },
        )
        os.rename(building, target)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return target / "manifest.json"
