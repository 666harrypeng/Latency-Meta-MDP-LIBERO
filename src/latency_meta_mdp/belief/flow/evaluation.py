"""Deterministic sample-based evaluation for trained Flow Belief checkpoints."""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors

from latency_meta_mdp.artifacts import sha256_file
from latency_meta_mdp.belief.common.feature_corpus import FeatureBeliefCorpus
from latency_meta_mdp.belief.flow.config import FlowBeliefConfig
from latency_meta_mdp.belief.flow.metrics import (
    sample_distribution_metrics,
    sample_mean_physical_metrics,
)
from latency_meta_mdp.belief.flow.model import FlowBeliefModel
from latency_meta_mdp.belief.flow.sampler import sample_flow_belief
from latency_meta_mdp.belief.flow.training_data import FlowBeliefNormalization
from latency_meta_mdp.belief_training_data import InteractionMode
from latency_meta_mdp.vision_probe_data import ProbeSplit


def _load_normalization(path: Path) -> FlowBeliefNormalization:
    with np.load(path, allow_pickle=False) as source:
        return FlowBeliefNormalization(
            proprio_mean=np.array(source["proprio_mean"], copy=True),
            proprio_std=np.array(source["proprio_std"], copy=True),
            action_mean=np.array(source["action_mean"], copy=True),
            action_std=np.array(source["action_std"], copy=True),
            target_mean=np.array(source["target_mean"], copy=True),
            target_std=np.array(source["target_std"], copy=True),
        )


def _bundle(
    *,
    samples: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    normalization: FlowBeliefNormalization,
) -> dict[str, Any]:
    return {
        "distribution": sample_distribution_metrics(
            samples=samples,
            target=target,
            weights=weights,
        ),
        "physical": sample_mean_physical_metrics(
            normalized_samples=samples,
            normalized_target=target,
            weights=weights,
            target_mean=normalization.target_mean,
            target_std=normalization.target_std,
        ),
    }


def _subset_bundle(
    *,
    samples: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
    normalization: FlowBeliefNormalization,
) -> dict[str, Any] | None:
    masked = weights * mask
    mass = masked.sum(axis=1)
    valid = mass > 0.0
    if not np.any(valid):
        return None
    normalized = masked[valid] / mass[valid, None]
    result = _bundle(
        samples=samples[valid],
        target=target[valid],
        weights=normalized,
        normalization=normalization,
    )
    result["retained_probability_mass_mean"] = float(np.mean(mass[valid]))
    return result


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def evaluate_level_flow_belief(
    *,
    corpus: FeatureBeliefCorpus,
    config: FlowBeliefConfig,
    checkpoint_dir: Path,
    output_dir: Path,
    device: str,
    context_limit: int | None = None,
    sample_count: int | None = None,
    step_count: int | None = None,
    evaluation_split: ProbeSplit = ProbeSplit.HOLDOUT,
) -> Path:
    checkpoint = checkpoint_dir.resolve()
    target_dir = output_dir.resolve()
    if target_dir.exists():
        raise FileExistsError(f"Flow evaluation output already exists: {target_dir}")
    model_path = checkpoint / "model.safetensors"
    normalization_path = checkpoint / "normalization.npz"
    if not model_path.is_file() or not normalization_path.is_file():
        raise FileNotFoundError("Flow evaluation checkpoint artifacts are missing")
    selected_sample_count = sample_count or config.evaluation_sample_count
    selected_step_count = step_count or config.solver_step_count
    if selected_sample_count <= 0 or selected_step_count <= 0:
        raise ValueError("Flow evaluation sample and step counts must be positive")
    references = corpus.sample_references[evaluation_split]
    selected_count = (
        len(references)
        if context_limit is None
        else min(context_limit, len(references))
    )
    if selected_count <= 0:
        raise ValueError("Flow evaluation requires at least one holdout context")
    normalization = _load_normalization(normalization_path)
    model = FlowBeliefModel(config).to(device)
    model.load_state_dict(load_safetensors(model_path), strict=True)
    model.eval()
    all_samples = []
    all_targets = []
    all_modes = []
    all_absorbing = []
    sampling_seconds = 0.0
    delay_ticks = np.arange(1, 21, dtype=np.int64)
    for batch_start in range(0, selected_count, config.batch_size):
        batch_stop = min(batch_start + config.batch_size, selected_count)
        contexts = [
            corpus.materialize(evaluation_split, offset)
            for offset in range(batch_start, batch_stop)
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
        normalized_target = np.stack(
            [
                (context.target_states - normalization.target_mean)
                / normalization.target_std
                for context in contexts
            ]
        ).astype(np.float32)
        modes = np.stack([context.target_interaction_mode for context in contexts])
        absorbing = np.stack([context.target_absorbing for context in contexts])
        noises = []
        for offset in range(batch_start, batch_stop):
            rng = np.random.default_rng(
                config.evaluation_seed + corpus.level * 1_000_000 + offset
            )
            noises.append(
                rng.standard_normal(
                    (20, selected_sample_count, 22),
                    dtype=np.float32,
                )
            )
        with torch.inference_mode():
            belief = model.encoder(
                vision_history=torch.from_numpy(vision).to(device),
                proprio_history=torch.from_numpy(proprio).to(device),
                remaining_actions=torch.from_numpy(actions).to(device),
                latency_probabilities=torch.from_numpy(latency).to(device),
            )
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            started = time.perf_counter()
            sampled = sample_flow_belief(
                vector_field=model.vector_field,
                belief_tokens=belief,
                delay_ticks=torch.from_numpy(
                    np.broadcast_to(delay_ticks, (len(contexts), 20)).copy()
                ).to(device),
                noise=torch.from_numpy(np.stack(noises)).to(device),
                solver=config.solver,
                step_count=selected_step_count,
            )
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            sampling_seconds += time.perf_counter() - started
        all_samples.append(sampled.cpu().numpy())
        all_targets.append(normalized_target)
        all_modes.append(modes)
        all_absorbing.append(absorbing)
    samples_array = np.concatenate(all_samples, axis=0)
    target_array = np.concatenate(all_targets, axis=0)
    mode_array = np.concatenate(all_modes, axis=0)
    absorbing_array = np.concatenate(all_absorbing, axis=0).astype(bool)
    weights = np.broadcast_to(
        np.asarray(corpus.latency_law.probabilities, dtype=np.float64),
        (selected_count, 20),
    ).copy()
    metrics: dict[str, Any] = {
        "overall": _bundle(
            samples=samples_array,
            target=target_array,
            weights=weights,
            normalization=normalization,
        ),
        "per_delay": {},
        "sampling": {
            "wall_seconds": sampling_seconds,
            "seconds_per_context": sampling_seconds / selected_count,
            "sample_count": selected_sample_count,
            "solver": config.solver,
            "solver_step_count": selected_step_count,
            "evaluation_split": evaluation_split.value,
        },
    }
    for delay_index, delay in enumerate(delay_ticks):
        metrics["per_delay"][str(int(delay))] = _bundle(
            samples=samples_array[:, delay_index : delay_index + 1],
            target=target_array[:, delay_index : delay_index + 1],
            weights=np.ones((selected_count, 1), dtype=np.float64),
            normalization=normalization,
        )
    subsets = {
        "pre_handoff": (
            ~absorbing_array
            & np.isin(mode_array, (InteractionMode.FREE, InteractionMode.CONTACT))
        ),
        "post_handoff": (
            ~absorbing_array & (mode_array == InteractionMode.GRASPED)
        ),
        "absorbing": absorbing_array,
    }
    for name, mask in subsets.items():
        bundle = _subset_bundle(
            samples=samples_array,
            target=target_array,
            weights=weights,
            mask=mask,
            normalization=normalization,
        )
        if bundle is not None:
            metrics[name] = bundle
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    building = target_dir.parent / f"{target_dir.name}.building-{uuid.uuid4().hex}"
    building.mkdir()
    try:
        _write_json(building / "metrics.json", metrics)
        np.savez(
            building / "summary_arrays.npz",
            sample_mean_normalized=samples_array.mean(axis=2),
            sample_std_normalized=samples_array.std(axis=2),
            target_normalized=target_array,
            interaction_mode=mode_array,
            absorbing=absorbing_array,
        )
        artifacts = {
            name: sha256_file(building / name)
            for name in ("metrics.json", "summary_arrays.npz")
        }
        manifest = {
            "schema_version": 1,
            "format_id": "level_flow_belief_evaluation_v1",
            "level": corpus.level,
            "context_count": selected_count,
            "sample_count": selected_sample_count,
            "solver": config.solver,
            "solver_step_count": selected_step_count,
            "checkpoint_model_sha256": sha256_file(model_path),
            "checkpoint_normalization_sha256": sha256_file(normalization_path),
            "artifacts": artifacts,
        }
        _write_json(building / "manifest.json", manifest)
        if target_dir.exists():
            raise FileExistsError(f"Flow evaluation output already exists: {target_dir}")
        os.rename(building, target_dir)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return target_dir / "manifest.json"
