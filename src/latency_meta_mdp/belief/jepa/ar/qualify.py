"""One-shot final qualification for a completed stride-4 L3 admission run."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from latency_meta_mdp.belief.jepa.ar.qualification import (
    evaluate_latency_mixture_invariants,
    evaluate_stride4_qualification_batch,
    summarize_stride4_qualification,
)
from latency_meta_mdp.belief.jepa.ar.temporal_view import (
    TemporalJepaCorpus,
    build_shared_temporal_indices,
    collate_temporal_jepa_evaluation_samples,
)
from latency_meta_mdp.belief.jepa.ar.train import (
    load_jepa_admission_training_checkpoint,
    load_jepa_admission_training_config,
    load_jepa_formal_validation_records,
)
from latency_meta_mdp.belief.jepa.ar.training import (
    build_upstream_aligned_optimizer,
)
from latency_meta_mdp.belief.jepa.backbone import (
    ActionConditionedJepaPredictor,
)
from latency_meta_mdp.belief.jepa.config import (
    load_action_conditioned_jepa_config,
    load_jepa_temporal_sampling,
)
from latency_meta_mdp.belief.jepa.corpus import (
    load_jepa_proprio_normalization,
    load_verified_jepa_inputs,
)
from latency_meta_mdp.belief.jepa.diagnostics.history_signal import (
    load_temporal_signal_episodes,
)
from latency_meta_mdp.belief.jepa.diagnostics.readout import (
    DualViewObjectStateReadout,
    ObjectStateNormalization,
)
from latency_meta_mdp.data.collection.artifacts import (
    _fsync_directory,
    _write_file_fsynced,
)
from latency_meta_mdp.io.artifacts import sha256_file
from latency_meta_mdp.runtime.latency_law import load_latency_law
from latency_meta_mdp.runtime.latency_law_family import load_episode_latency_law_family

_FORMAT_ID = "action_conditioned_jepa_l3_qualification_v1"
_TEMPORAL_CONFIG_ID = "stride4_80ms_history_160ms"


@dataclass(frozen=True)
class CompletedJepaAdmissionRun:
    run_root: Path
    checkpoint_dir: Path
    model_seed: int
    existing_formal_validation: dict[str, Any]
    manifest: dict[str, Any]


def load_completed_l3_admission_run(
    run_root: Path,
    *,
    expected_model_seed: int,
) -> CompletedJepaAdmissionRun:
    root = Path(run_root).resolve()
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("completed L3 admission run manifest is invalid") from error
    if type(manifest) is not dict:
        raise ValueError("completed L3 admission run manifest is invalid")
    preflight = manifest.get("preflight")
    progress = manifest.get("progress")
    formal = manifest.get("final_formal_validation")
    checkpoint = root / "checkpoints/epoch-075"
    if (
        manifest.get("format_id") != "action_conditioned_jepa_l3_admission_run_v1"
        or manifest.get("qualification_only") is not False
        or type(preflight) is not dict
        or type(progress) is not dict
        or progress.get("completed_epochs") != 75
        or progress.get("model_seed") != expected_model_seed
        or preflight.get("model_seed") != expected_model_seed
        or progress.get("temporal_config_id") != _TEMPORAL_CONFIG_ID
        or preflight.get("temporal_config_id") != _TEMPORAL_CONFIG_ID
        or manifest.get("formal_validation_opened") is not True
        or type(formal) is not dict
        or set(formal) != {"native", "deployed_d20"}
        or not (checkpoint / "manifest.json").is_file()
    ):
        raise ValueError("final qualification requires a completed epoch-75 admission run")
    return CompletedJepaAdmissionRun(
        run_root=root,
        checkpoint_dir=checkpoint,
        model_seed=expected_model_seed,
        existing_formal_validation=formal,
        manifest=manifest,
    )


def write_l3_qualification_report(
    *,
    output_path: Path,
    report: dict[str, Any],
) -> Path:
    target = Path(output_path).absolute()
    if target.exists():
        raise FileExistsError(target)
    if type(report) is not dict or report.get("format_id") != _FORMAT_ID:
        raise ValueError("L3 qualification report format is invalid")
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_file_fsynced(
        target,
        (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
    )
    _fsync_directory(target.parent)
    return target


def _paths(root: Path, *, model_seed: int) -> dict[str, Path]:
    source_id = "panda-ball-structured-source-quota-formal-100x4-v1"
    return {
        "model_config": root / "configs/models/jepa/model.yaml",
        "level_config": root / "configs/models/jepa/l3.yaml",
        "temporal_config": (root / "configs/models/jepa/stride4_80ms_history_160ms.yaml"),
        "training_config": root / "configs/legacy/training/l3_admission.yaml",
        "source_manifest": root / "outputs/source_corpus" / source_id / "manifest.json",
        "cache_manifest": (
            root
            / "outputs/derived/vision_features"
            / "dinov3-vits16-structured-source-100x4-v1/manifest.json"
        ),
        "split_manifest": (
            root
            / "outputs/derived/source_splits"
            / source_id
            / "train80-validation20-seed20260903-v1.json"
        ),
        "selection_manifest": (
            root
            / "outputs/derived/action_conditioned_jepa/temporal_selection"
            / "l3-trainpool-four-configs-v1/manifest.json"
        ),
        "run_root": (
            root
            / "outputs/training/action_conditioned_jepa/l3-final-admission"
            / _TEMPORAL_CONFIG_ID
            / f"seed-{model_seed}"
        ),
        "readout_root": (
            root / "outputs/training/action_conditioned_jepa/l3-gt-latent-object-readout"
        ),
        "nominal_law": root / "configs/runtime/latency/truncated_beta_8_65_400ms_v1.yaml",
        "law_family": root / "configs/runtime/latency/truncated_beta_family_8_65_400ms_v1.yaml",
    }


def _candidate_samplings(root: Path) -> tuple[Any, ...]:
    config_root = root / "configs/models/jepa"
    return tuple(
        load_jepa_temporal_sampling(config_root / name)
        for name in (
            "dense_20ms_history_100ms.yaml",
            "stride2_40ms_history_120ms.yaml",
            "stride4_80ms_history_160ms.yaml",
            "stride5_100ms_history_200ms.yaml",
        )
    )


def _load_readout(
    *,
    root: Path,
    artifact_root: Path,
    device: torch.device,
) -> tuple[DualViewObjectStateReadout, ObjectStateNormalization, dict[str, Any]]:
    try:
        summary = json.loads((artifact_root / "training_summary.json").read_text(encoding="utf-8"))
        normalization_payload = json.loads(
            (artifact_root / "normalization.json").read_text(encoding="utf-8")
        )
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("GT-latent object readout artifact is invalid") from error
    if (
        type(summary) is not dict
        or type(normalization_payload) is not dict
        or (
            summary.get("format_id") != "l3_gt_latent_object_readout_v1"
            or summary.get("train_episode_count") != 320
            or type(summary.get("history")) is not list
            or len(summary["history"]) != 40
            or set(normalization_payload) != {"mean", "scale"}
        )
    ):
        raise ValueError("GT-latent object readout contract is incompatible")
    normalization = ObjectStateNormalization(
        mean=np.asarray(normalization_payload["mean"], dtype=np.float32),
        scale=np.asarray(normalization_payload["scale"], dtype=np.float32),
    )
    readout = DualViewObjectStateReadout(project_root=root).to(device)
    readout.load_state_dict(
        load_file(str(artifact_root / "model.safetensors"), device=str(device)),
        strict=True,
    )
    readout.eval()
    return readout, normalization, summary


def _representative_laws(
    *,
    nominal_path: Path,
    family_path: Path,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    nominal = load_latency_law(nominal_path)
    family = load_episode_latency_law_family(family_path)
    floor = family.uniform_floor_max / 2.0
    laws = {
        "nominal": nominal.probabilities,
        "uniform": np.full(20, 0.05, dtype=np.float64),
        "fast_one_sigma": family.build_shifted_law(
            name="fast-one-sigma",
            mean_logit_offset=-family.mean_logit_std,
            log_concentration_offset=0.0,
            uniform_floor=floor,
        ).probabilities,
        "slow_one_sigma": family.build_shifted_law(
            name="slow-one-sigma",
            mean_logit_offset=family.mean_logit_std,
            log_concentration_offset=0.0,
            uniform_floor=floor,
        ).probabilities,
    }
    return {
        name: torch.from_numpy(np.array(probabilities, copy=True))
        .to(device=device, dtype=torch.float32)
        .unsqueeze(0)
        .repeat(batch_size, 1)
        for name, probabilities in laws.items()
    }


def execute_l3_final_qualification(
    *,
    project_root: Path,
    model_seed: int,
    device: str,
    batch_size: int,
    output_path: Path,
) -> Path:
    root = Path(project_root).resolve()
    target = Path(output_path).absolute()
    if target.exists():
        raise FileExistsError(target)
    if type(model_seed) is not int or model_seed not in (7, 17, 27):
        raise ValueError("model_seed must be 7, 17, or 27")
    if type(device) is not str or re.fullmatch(r"cuda:[0-9]+", device) is None:
        raise ValueError("device must identify one CUDA device")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    target_device = torch.device(device)
    paths = _paths(root, model_seed=model_seed)
    completed = load_completed_l3_admission_run(
        paths["run_root"],
        expected_model_seed=model_seed,
    )
    config = load_action_conditioned_jepa_config(
        model_path=paths["model_config"],
        level_path=paths["level_config"],
        temporal_sampling_path=paths["temporal_config"],
    )
    training = load_jepa_admission_training_config(paths["training_config"])
    inputs = load_verified_jepa_inputs(
        source_root=paths["source_manifest"].parent,
        cache_run_manifest=paths["cache_manifest"],
        split_manifest_path=paths["split_manifest"],
        config=config,
    )
    level_ids = set(inputs.source.episode_ids(level=3))
    validation_ids = tuple(sorted(level_ids.intersection(inputs.split.validation_episode_ids)))
    normalization = load_jepa_proprio_normalization(
        completed.run_root / "proprio_normalization.json"
    )
    model = ActionConditionedJepaPredictor(
        config=config,
        proprio_normalization=normalization,
        project_root=root,
    ).to(target_device)
    optimizer = build_upstream_aligned_optimizer(model=model, config=training)
    input_hashes = {
        name: sha256_file(paths[name])
        for name in (
            "source_manifest",
            "cache_manifest",
            "split_manifest",
            "selection_manifest",
            "model_config",
            "temporal_config",
            "training_config",
        )
    }
    progress = load_jepa_admission_training_checkpoint(
        output_dir=completed.checkpoint_dir,
        model=model,
        optimizer=optimizer,
        expected_training_config=training,
        expected_input_sha256=input_hashes,
        expected_model_seed=model_seed,
    )
    del optimizer
    records = load_jepa_formal_validation_records(
        inputs=inputs,
        episode_ids=validation_ids,
        progress=progress,
        config=training,
    )
    indices = build_shared_temporal_indices(
        records=records,
        samplings=_candidate_samplings(root),
    )
    corpus = TemporalJepaCorpus(
        records=records,
        indices=indices,
        episode_ids=validation_ids,
        partition="development",
        sampling=config.temporal_sampling,
        normalization=normalization,
    )
    signal_episodes = load_temporal_signal_episodes(
        inputs=inputs,
        episode_ids=validation_ids,
        level=3,
        split="validation",
    )
    signals = {value.record.episode_id: value for value in signal_episodes}
    readout, object_normalization, readout_summary = _load_readout(
        root=root,
        artifact_root=paths["readout_root"],
        device=target_device,
    )
    loader = DataLoader(
        corpus,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_temporal_jepa_evaluation_samples,
    )
    model.eval()
    batches = []
    first_batch = None
    for value in loader:
        value = value.to(target_device, non_blocking=True)
        if first_batch is None:
            first_batch = value
        batches.append(
            evaluate_stride4_qualification_batch(
                model=model,
                readout=readout,
                object_normalization=object_normalization,
                batch=value,
                signal_episodes=signals,
            )
        )
    if first_batch is None:
        raise RuntimeError("formal qualification corpus is empty")
    representative_rollout = model.rollout_native(first_batch.context)
    j5 = evaluate_latency_mixture_invariants(
        rollout=representative_rollout,
        probabilities_by_law=_representative_laws(
            nominal_path=paths["nominal_law"],
            family_path=paths["law_family"],
            batch_size=first_batch.batch_size,
            device=target_device,
        ),
        sampling=config.temporal_sampling,
    )
    qualification = summarize_stride4_qualification(
        tuple(batches),
        signal_episodes=signals,
    )
    report = {
        "format_id": _FORMAT_ID,
        "model_seed": model_seed,
        "temporal_config_id": _TEMPORAL_CONFIG_ID,
        "checkpoint_epoch": 75,
        "formal_validation": {
            "master_count": len({value.logical_master_task_index for value in records}),
            "episode_count": len(records),
            "context_count": len(corpus),
            "opened_by_completed_training_run": True,
        },
        "j1_and_deployed_d20": completed.existing_formal_validation,
        **qualification,
        "j5_latency_mixture": j5,
        "readout": {
            "format_id": readout_summary["format_id"],
            "train_episode_count": readout_summary["train_episode_count"],
            "train_boundary_count": readout_summary["train_boundary_count"],
            "final_train_loss": readout_summary["final_train_loss"],
        },
        "pending_gates": [
            "J4 same-source action-effect",
            "J6a final-checkpoint client runtime",
        ],
        "admission_ready": False,
    }
    return write_l3_qualification_report(output_path=target, report=report)
