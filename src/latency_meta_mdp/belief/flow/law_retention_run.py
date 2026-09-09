"""Atomic frozen-Encoder latency-law retention diagnostics."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors

from latency_meta_mdp.artifacts import collect_implementation_provenance, sha256_file
from latency_meta_mdp.belief.common.feature_corpus import (
    FeatureBeliefCorpus,
    load_level_feature_belief_corpus,
)
from latency_meta_mdp.belief.flow.config import load_flow_belief_config
from latency_meta_mdp.belief.flow.encoder import FlowBeliefEncoder
from latency_meta_mdp.belief.flow.law_retention import (
    LAW_VARIANT_NAMES,
    LinearLawReadoutFit,
    evaluate_law_reconstruction,
    fit_linear_law_readout,
    law_invariant_tokens,
    select_evenly_spaced_offsets,
)
from latency_meta_mdp.belief.flow.law_retention_config import (
    load_law_retention_probe_config,
)
from latency_meta_mdp.belief.flow.multilaw_config import (
    load_multilaw_flow_training_config,
)
from latency_meta_mdp.belief.flow.multilaw_evaluation_config import (
    load_multilaw_flow_evaluation_config,
)
from latency_meta_mdp.belief.flow.training_data import FlowBeliefNormalization
from latency_meta_mdp.latency_law_family import load_episode_latency_law_family
from latency_meta_mdp.vision_encoder import load_vision_encoder_spec
from latency_meta_mdp.vision_probe_data import ProbeSplit


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


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


def _law_variants(*, family: Any, evaluation_spec: Any) -> dict[str, np.ndarray]:
    variants = {}
    for name, spec in evaluation_spec.shifted_laws.items():
        law = family.build_shifted_law(
            name=name,
            mean_logit_offset=spec.mean_logit_offset,
            log_concentration_offset=spec.log_concentration_offset,
            uniform_floor=spec.uniform_floor,
        )
        variants[f"shifted_{name}"] = law.probabilities
    if tuple(variants) != ("shifted_fast", "shifted_slow", "shifted_wide"):
        raise RuntimeError("law-retention shifted variants are inconsistent")
    return variants


def _cache_counterfactual_tokens(
    *,
    corpus: FeatureBeliefCorpus,
    split: ProbeSplit,
    context_limit: int,
    encoder: FlowBeliefEncoder,
    normalization: FlowBeliefNormalization,
    fixed_variants: dict[str, np.ndarray],
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], np.ndarray]:
    offsets = select_evenly_spaced_offsets(
        total=len(corpus.sample_references[split]),
        limit=context_limit,
    )
    token_batches = []
    target_batches = []
    episode_ids = []
    source_ticks = []
    encoder.eval()
    for start in range(0, len(offsets), batch_size):
        selected = offsets[start : start + batch_size]
        contexts = [corpus.materialize(split, offset) for offset in selected]
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
        nominal = np.broadcast_to(
            corpus.latency_law.probabilities,
            (len(contexts), 20),
        )
        variant_probabilities = np.stack(
            (
                nominal,
                np.stack([context.latency_probabilities for context in contexts]),
                np.broadcast_to(fixed_variants["shifted_fast"], nominal.shape),
                np.broadcast_to(fixed_variants["shifted_slow"], nominal.shape),
                np.broadcast_to(fixed_variants["shifted_wide"], nominal.shape),
            ),
            axis=1,
        ).astype(np.float32)
        variants = []
        with torch.inference_mode():
            for variant_index in range(len(LAW_VARIANT_NAMES)):
                belief = encoder(
                    vision_history=torch.from_numpy(vision).to(device),
                    proprio_history=torch.from_numpy(proprio).to(device),
                    remaining_actions=torch.from_numpy(actions).to(device),
                    latency_probabilities=torch.from_numpy(
                        variant_probabilities[:, variant_index]
                    ).to(device),
                )
                variants.append(belief.cpu().numpy())
        token_batches.append(np.stack(variants, axis=1))
        target_batches.append(variant_probabilities)
        episode_ids.extend(context.episode_id for context in contexts)
        source_ticks.extend(context.source_tick for context in contexts)
    return (
        np.concatenate(token_batches, axis=0),
        np.concatenate(target_batches, axis=0),
        tuple(episode_ids),
        np.asarray(source_ticks, dtype=np.int64),
    )


def _global_prior_predictions(
    *,
    training_targets: np.ndarray,
    validation_targets: np.ndarray,
) -> np.ndarray:
    prior = np.asarray(training_targets, dtype=np.float64).mean(axis=(0, 1))
    prior /= prior.sum()
    return np.broadcast_to(prior, validation_targets.shape).copy()


def _fit_payload(fit: LinearLawReadoutFit) -> dict[str, Any]:
    return {
        "best_epoch": fit.best_epoch,
        "best_validation_cross_entropy": fit.best_validation_cross_entropy,
        "epochs_completed": fit.epochs_completed,
    }


def _counterfactual_sensitivity(tokens: np.ndarray) -> dict[str, float]:
    values = np.asarray(tokens, dtype=np.float64)
    nominal = values[:, 0]
    difference = values[:, 1:] - nominal[:, None]
    law_response = np.sqrt(np.mean(np.square(difference), axis=(-1, -2)))
    centered_context = nominal - nominal.mean(axis=0, keepdims=True)
    context_variation = np.sqrt(np.mean(np.square(centered_context), axis=(-1, -2)))
    denominator = float(np.mean(context_variation))
    return {
        "counterfactual_token_rms_mean": float(np.mean(law_response)),
        "nominal_context_token_rms_mean": denominator,
        "law_to_context_rms_ratio": float(np.mean(law_response) / max(denominator, 1e-12)),
    }


def run_law_retention_probe(
    *,
    project_root: Path,
    source_bulk_manifest: Path,
    cache_run_manifest: Path,
    multilaw_run_manifest: Path,
    level: int,
    vision_config_path: Path,
    temporal_config_path: Path,
    nominal_law_path: Path,
    family_config_path: Path,
    flow_config_path: Path,
    multilaw_config_path: Path,
    split_config_path: Path,
    evaluation_config_path: Path,
    probe_config_path: Path,
    output_dir: Path,
    device: str,
) -> Path:
    if level not in (1, 2, 3):
        raise ValueError("law-retention level must be 1, 2, or 3")
    inputs = {
        "source_bulk_manifest": source_bulk_manifest.resolve(),
        "cache_run_manifest": cache_run_manifest.resolve(),
        "multilaw_run_manifest": multilaw_run_manifest.resolve(),
        "vision_config": vision_config_path.resolve(),
        "temporal_config": temporal_config_path.resolve(),
        "nominal_law": nominal_law_path.resolve(),
        "family_config": family_config_path.resolve(),
        "flow_config": flow_config_path.resolve(),
        "multilaw_config": multilaw_config_path.resolve(),
        "split_config": split_config_path.resolve(),
        "evaluation_config": evaluation_config_path.resolve(),
        "probe_config": probe_config_path.resolve(),
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(f"law-retention input does not exist: {path}")
    training_run = _load_json(inputs["multilaw_run_manifest"])
    if (
        training_run.get("format_id") != "flow_belief_multilaw_run_v2"
        or training_run.get("eligible") is not True
        or training_run.get("levels") != [level]
    ):
        raise ValueError("law-retention probe requires one eligible level run")
    expected_hashes = training_run.get("input_sha256")
    mapping = {
        "source_bulk_manifest": "source_bulk_manifest",
        "cache_run_manifest": "cache_run_manifest",
        "vision_config": "vision_config",
        "temporal_config": "temporal_config",
        "nominal_law": "nominal_law",
        "family_config": "family_config",
        "flow_config": "flow_config",
        "multilaw_config": "multilaw_config",
        "split_config": "split_config",
    }
    if not isinstance(expected_hashes, dict) or any(
        expected_hashes.get(expected_name) != sha256_file(inputs[input_name])
        for input_name, expected_name in mapping.items()
    ):
        raise ValueError("law-retention inputs do not match training")

    flow_config = load_flow_belief_config(inputs["flow_config"])
    multilaw = load_multilaw_flow_training_config(inputs["multilaw_config"])
    evaluation_spec = load_multilaw_flow_evaluation_config(inputs["evaluation_config"])
    probe_config = load_law_retention_probe_config(inputs["probe_config"])
    family = load_episode_latency_law_family(inputs["family_config"])
    vision_spec = load_vision_encoder_spec(inputs["vision_config"])
    if training_run.get("latency_law_family_id") != multilaw.latency_law_family_id:
        raise ValueError("law-retention family identifier is inconsistent")
    corpus = load_level_feature_belief_corpus(
        project_root=project_root,
        source_bulk_manifest=inputs["source_bulk_manifest"],
        cache_run_manifest=inputs["cache_run_manifest"],
        expected_spec=vision_spec,
        temporal_config_path=inputs["temporal_config"],
        latency_law_path=inputs["nominal_law"],
        latency_law_family_path=inputs["family_config"],
        split_plan_path=inputs["split_config"],
        level=level,
    )
    checkpoint = inputs["multilaw_run_manifest"].parent / f"L{level}"
    encoder_path = checkpoint / "encoder.safetensors"
    normalization_path = checkpoint / "normalization.npz"
    if not encoder_path.is_file() or not normalization_path.is_file():
        raise FileNotFoundError("law-retention checkpoint artifacts are missing")
    normalization = _load_normalization(normalization_path)
    encoder = FlowBeliefEncoder(flow_config).to(device)
    encoder.load_state_dict(load_safetensors(encoder_path), strict=True)
    encoder.requires_grad_(False)
    fixed_variants = _law_variants(family=family, evaluation_spec=evaluation_spec)
    training_tokens, training_targets, training_episodes, training_ticks = (
        _cache_counterfactual_tokens(
            corpus=corpus,
            split=ProbeSplit.TRAIN,
            context_limit=probe_config.training_context_limit,
            encoder=encoder,
            normalization=normalization,
            fixed_variants=fixed_variants,
            batch_size=probe_config.encoder_batch_size,
            device=device,
        )
    )
    validation_tokens, validation_targets, validation_episodes, validation_ticks = (
        _cache_counterfactual_tokens(
            corpus=corpus,
            split=ProbeSplit.VALIDATION,
            context_limit=probe_config.validation_context_limit,
            encoder=encoder,
            normalization=normalization,
            fixed_variants=fixed_variants,
            batch_size=probe_config.encoder_batch_size,
            device=device,
        )
    )
    primary = fit_linear_law_readout(
        training_tokens=training_tokens,
        training_targets=training_targets,
        validation_tokens=validation_tokens,
        validation_targets=validation_targets,
        config=probe_config,
        device=device,
        seed=probe_config.random_seed + level,
    )
    invariant = fit_linear_law_readout(
        training_tokens=law_invariant_tokens(training_tokens),
        training_targets=training_targets,
        validation_tokens=law_invariant_tokens(validation_tokens),
        validation_targets=validation_targets,
        config=probe_config,
        device=device,
        seed=probe_config.random_seed + 100 + level,
    )
    primary_metrics = evaluate_law_reconstruction(
        predicted=primary.validation_probabilities,
        target=validation_targets,
        variant_names=LAW_VARIANT_NAMES,
    )
    invariant_metrics = evaluate_law_reconstruction(
        predicted=invariant.validation_probabilities,
        target=validation_targets,
        variant_names=LAW_VARIANT_NAMES,
    )
    global_metrics = evaluate_law_reconstruction(
        predicted=_global_prior_predictions(
            training_targets=training_targets,
            validation_targets=validation_targets,
        ),
        target=validation_targets,
        variant_names=LAW_VARIANT_NAMES,
    )
    baseline_jsd = min(
        invariant_metrics["jensen_shannon_divergence_mean"],
        global_metrics["jensen_shannon_divergence_mean"],
    )
    relative_improvement = 1.0 - (
        primary_metrics["jensen_shannon_divergence_mean"] / max(baseline_jsd, 1e-12)
    )
    gate_checks = {
        "effective_mean_mae": (
            primary_metrics["effective_mean_mae_ms"]
            <= probe_config.effective_mean_mae_ms_max
        ),
        "baseline_relative_improvement": (
            relative_improvement >= probe_config.baseline_relative_improvement_min
        ),
        "ordering_accuracy": (
            primary_metrics["fast_nominal_slow_ordering_accuracy"]
            >= probe_config.ordering_accuracy_min
        ),
    }
    gate_passed = all(gate_checks.values())
    metrics = {
        "level": level,
        "variant_names": list(LAW_VARIANT_NAMES),
        "primary": primary_metrics,
        "law_invariant_baseline": invariant_metrics,
        "global_prior_baseline": global_metrics,
        "primary_fit": _fit_payload(primary),
        "law_invariant_fit": _fit_payload(invariant),
        "counterfactual_sensitivity": _counterfactual_sensitivity(validation_tokens),
        "baseline_relative_jsd_improvement": relative_improvement,
        "gate_checks": gate_checks,
        "gate_passed": gate_passed,
    }

    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"law-retention output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f"{target.name}.building-{uuid.uuid4().hex}"
    building.mkdir()
    provenance = collect_implementation_provenance(project_root)
    try:
        save_safetensors(
            {
                "weight": torch.from_numpy(primary.weight),
                "bias": torch.from_numpy(primary.bias),
            },
            building / "primary_readout.safetensors",
        )
        np.savez(
            building / "readout_normalization.npz",
            input_mean=primary.input_mean,
            input_std=primary.input_std,
        )
        np.savez(
            building / "validation_predictions.npz",
            predicted_probabilities=primary.validation_probabilities,
            target_probabilities=validation_targets,
            episode_ids=np.asarray(validation_episodes),
            source_ticks=validation_ticks,
        )
        _write_json(building / "metrics.json", metrics)
        _write_json(building / "training_history.json", list(primary.training_history))
        artifacts = {
            path.name: sha256_file(path)
            for path in sorted(building.iterdir())
            if path.is_file()
        }
        _write_json(
            building / "manifest.json",
            {
                "schema_version": 1,
                "format_id": "flow_belief_law_retention_probe_v1",
                "eligible": not provenance.dirty,
                "blockers": [] if not provenance.dirty else ["implementation_worktree_dirty"],
                "implementation_revision": provenance.revision,
                "implementation_source_sha256": provenance.source_sha256,
                "implementation_dirty": provenance.dirty,
                "level": level,
                "checkpoint_encoder_sha256": sha256_file(encoder_path),
                "training_context_count": len(training_tokens),
                "validation_context_count": len(validation_tokens),
                "training_episode_count": len(set(training_episodes)),
                "validation_episode_count": len(set(validation_episodes)),
                "latency_law_family_id": multilaw.latency_law_family_id,
                "gate_passed": gate_passed,
                "input_sha256": {name: sha256_file(path) for name, path in inputs.items()},
                "artifacts": artifacts,
            },
        )
        os.rename(building, target)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return target / "manifest.json"
