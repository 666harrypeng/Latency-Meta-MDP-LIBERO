"""Preflight and execution boundary for one temporal-selection fold job."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from latency_meta_mdp.belief.jepa.ar.metrics import (
    evaluate_deployed_temporal_batch,
    evaluate_temporal_batch,
    select_monitor_indices,
    summarize_deployed_temporal_evaluation,
    summarize_temporal_evaluation,
)
from latency_meta_mdp.belief.jepa.ar.selection_config import (
    load_temporal_selection_artifact,
)
from latency_meta_mdp.belief.jepa.ar.temporal_view import (
    TemporalJepaCorpus,
    TemporalJepaDeployedEvaluationCorpus,
    collate_temporal_jepa_deployed_evaluation_samples,
    collate_temporal_jepa_evaluation_samples,
    collate_temporal_jepa_samples,
)
from latency_meta_mdp.belief.jepa.ar.training import (
    JepaTrainingProgress,
    build_epoch_microbatch_indices,
    build_upstream_aligned_optimizer,
    load_completed_temporal_jepa_history,
    load_temporal_jepa_training_checkpoint,
    load_temporal_jepa_training_config,
    publish_rolling_temporal_jepa_checkpoint,
    run_temporal_jepa_epoch,
)
from latency_meta_mdp.belief.jepa.backbone import (
    ActionConditionedJepaPredictor,
)
from latency_meta_mdp.belief.jepa.config import (
    load_action_conditioned_jepa_config,
)
from latency_meta_mdp.belief.jepa.corpus import (
    compute_jepa_proprio_normalization,
    load_jepa_proprio_normalization,
    load_verified_jepa_inputs,
    load_verified_jepa_record,
    write_jepa_proprio_normalization,
)
from latency_meta_mdp.belief.jepa.tracking import (
    build_wandb_run_spec,
    credential_environment_status,
    initialize_wandb_run,
    load_jepa_wandb_config,
)
from latency_meta_mdp.data.collection.artifacts import (
    _fsync_directory,
    _write_file_fsynced,
)
from latency_meta_mdp.io.artifacts import sha256_file


@dataclass(frozen=True)
class TemporalFoldPreflight:
    temporal_config_id: str
    fold_index: int
    model_seed: int
    fit_episode_count: int
    development_episode_count: int
    fit_context_count: int
    development_context_count: int
    logical_global_batch_size: int
    microbatch_size: int
    gradient_accumulation_steps: int
    optimizer_steps_per_epoch: int
    max_epochs: int
    total_optimizer_steps: int
    num_workers: int
    device: str
    output_dir: str
    wandb_api_key: str
    wandb_run_id: str

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


def should_emit_temporal_training_progress(
    *,
    epoch_step: int,
    optimizer_steps_per_epoch: int,
    period: int,
) -> bool:
    if (
        type(epoch_step) is not int
        or type(optimizer_steps_per_epoch) is not int
        or type(period) is not int
        or optimizer_steps_per_epoch <= 0
        or period <= 0
        or not 1 <= epoch_step <= optimizer_steps_per_epoch
    ):
        raise ValueError("console progress cadence is invalid")
    return epoch_step == 1 or epoch_step == optimizer_steps_per_epoch or epoch_step % period == 0


def format_temporal_training_progress(
    *,
    temporal_config_id: str,
    fold_index: int,
    model_seed: int,
    epoch: int,
    max_epochs: int,
    epoch_step: int,
    optimizer_steps_per_epoch: int,
    global_optimizer_step: int,
    total_optimizer_steps: int,
    examples_seen: int,
    total_loss: float,
    gradient_norm: float,
    learning_rate: float,
    weight_decay: float,
    elapsed_seconds: float,
) -> str:
    return (
        "[action-conditioned-jepa] "
        f"config={temporal_config_id} fold={fold_index} seed={model_seed} "
        f"epoch={epoch}/{max_epochs} epoch_step={epoch_step}/{optimizer_steps_per_epoch} "
        f"global_step={global_optimizer_step}/{total_optimizer_steps} "
        f"examples={examples_seen} loss={total_loss:.6f} "
        f"grad_norm={gradient_norm:.6f} lr={learning_rate:g} wd={weight_decay:g} "
        f"elapsed={elapsed_seconds:.1f}s"
    )


def build_temporal_training_loader(
    *,
    dataset,
    batches: tuple[tuple[int, ...], ...],
    num_workers: int,
    collate_fn: Callable,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_sampler=batches,
        num_workers=num_workers,
        pin_memory=num_workers > 0,
        persistent_workers=False,
        collate_fn=collate_fn,
    )


def build_temporal_evaluation_loader(
    *,
    dataset,
    batch_size: int,
    collate_fn: Callable,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        collate_fn=collate_fn,
    )


def build_temporal_fold_preflight(
    *,
    project_root: Path,
    temporal_config_path: Path,
    fold_index: int,
    model_seed: int,
    microbatch_size: int,
    num_workers: int,
    device: str,
    output_dir: Path,
    resume: bool = False,
) -> TemporalFoldPreflight:
    root = Path(project_root).resolve()
    temporal_path = Path(temporal_config_path)
    if not temporal_path.is_absolute():
        temporal_path = root / temporal_path
    config = load_action_conditioned_jepa_config(
        model_path=root / "configs/models/jepa/model.yaml",
        level_path=root / "configs/models/jepa/l3.yaml",
        temporal_sampling_path=temporal_path,
    )
    training = load_temporal_jepa_training_config(
        root / "configs/legacy/training/temporal_selection_l3.yaml"
    )
    selection = load_temporal_selection_artifact(
        root
        / "outputs/derived/action_conditioned_jepa/temporal_selection"
        / "l3-trainpool-four-configs-v1"
    )
    if type(fold_index) is not int or not 0 <= fold_index < len(selection.folds):
        raise ValueError("fold_index is outside the temporal selection")
    if model_seed not in (training.selection_seed, *training.confirmation_seeds):
        raise ValueError("model_seed is outside the training protocol")
    if (
        type(microbatch_size) is not int
        or microbatch_size <= 0
        or training.logical_global_batch_size % microbatch_size
    ):
        raise ValueError("microbatch_size must divide the logical global batch")
    if type(num_workers) is not int or num_workers < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    if type(device) is not str or re.fullmatch(r"cuda:[0-9]+", device) is None:
        raise ValueError("device must identify one CUDA device")
    target = Path(output_dir).absolute()
    if target.exists() and not resume:
        raise FileExistsError(target)
    if not target.exists() and resume:
        raise FileNotFoundError(target)
    fold = selection.folds[fold_index]
    fit_ids = set(fold.fit_episode_ids)
    development_ids = set(fold.development_episode_ids)
    fit_contexts = sum(index.episode_id in fit_ids for index in selection.indices)
    development_contexts = sum(index.episode_id in development_ids for index in selection.indices)
    fold_step_counts = []
    for candidate_fold in selection.folds:
        candidate_fit = set(candidate_fold.fit_episode_ids)
        count = sum(index.episode_id in candidate_fit for index in selection.indices)
        fold_step_counts.append(count // training.logical_global_batch_size)
    optimizer_steps_per_epoch = min(fold_step_counts)
    wandb_config = load_jepa_wandb_config(root / "configs/training/belief/wandb.yaml")
    wandb_spec = build_wandb_run_spec(
        config=wandb_config,
        selection_id=selection.manifest["selection_id"],
        level=3,
        temporal_config_id=config.temporal_sampling.config_id,
        fold_index=fold_index,
        model_seed=model_seed,
    )
    return TemporalFoldPreflight(
        temporal_config_id=config.temporal_sampling.config_id,
        fold_index=fold_index,
        model_seed=model_seed,
        fit_episode_count=len(fold.fit_episode_ids),
        development_episode_count=len(fold.development_episode_ids),
        fit_context_count=fit_contexts,
        development_context_count=development_contexts,
        logical_global_batch_size=training.logical_global_batch_size,
        microbatch_size=microbatch_size,
        gradient_accumulation_steps=training.logical_global_batch_size // microbatch_size,
        optimizer_steps_per_epoch=optimizer_steps_per_epoch,
        max_epochs=training.max_epochs,
        total_optimizer_steps=optimizer_steps_per_epoch * training.max_epochs,
        num_workers=num_workers,
        device=device,
        output_dir=str(target),
        wandb_api_key=credential_environment_status()["WANDB_API_KEY"],
        wandb_run_id=wandb_spec.run_id,
    )


def _canonical_paths(root: Path) -> dict[str, Path]:
    return {
        "model_config": root / "configs/models/jepa/model.yaml",
        "level_config": root / "configs/models/jepa/l3.yaml",
        "training_config": (root / "configs/legacy/training/temporal_selection_l3.yaml"),
        "wandb_config": root / "configs/training/belief/wandb.yaml",
        "source_manifest": (
            root
            / "outputs/source_corpus/panda-ball-structured-source-quota-formal-100x4-v1"
            / "manifest.json"
        ),
        "cache_manifest": (
            root
            / "outputs/derived/vision_features"
            / "dinov3-vits16-structured-source-100x4-v1/manifest.json"
        ),
        "split_manifest": (
            root
            / "outputs/derived/source_splits"
            / "panda-ball-structured-source-quota-formal-100x4-v1"
            / "train80-validation20-seed20260903-v1.json"
        ),
        "selection_manifest": (
            root
            / "outputs/derived/action_conditioned_jepa/temporal_selection"
            / "l3-trainpool-four-configs-v1/manifest.json"
        ),
    }


def _load_fold_records(*, inputs, episode_ids: tuple[str, ...]) -> tuple:
    return tuple(
        load_verified_jepa_record(
            inputs,
            episode_id=episode_id,
            level=3,
            split="train",
        )
        for episode_id in episode_ids
    )


def _evaluate_dataset(
    *,
    model: torch.nn.Module,
    dataset,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    loader = build_temporal_evaluation_loader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_temporal_jepa_evaluation_samples,
    )
    model.eval()
    metrics = []
    for batch in loader:
        metrics.append(
            evaluate_temporal_batch(
                model=model,
                batch=batch.to(device, non_blocking=True),
            )
        )
    return summarize_temporal_evaluation(tuple(metrics))


def _evaluate_deployed_dataset(
    *,
    model: torch.nn.Module,
    dataset,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    loader = build_temporal_evaluation_loader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_temporal_jepa_deployed_evaluation_samples,
    )
    model.eval()
    metrics = []
    for batch in loader:
        metrics.append(
            evaluate_deployed_temporal_batch(
                model=model,
                batch=batch.to(device, non_blocking=True),
            )
        )
    return summarize_deployed_temporal_evaluation(tuple(metrics))


def _write_epoch_metrics(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_file_fsynced(
        path,
        (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
    )
    _fsync_directory(path.parent)


def execute_temporal_fold_job(
    *,
    project_root: Path,
    temporal_config_path: Path,
    fold_index: int,
    model_seed: int,
    microbatch_size: int,
    num_workers: int,
    device: str,
    output_dir: Path,
    resume: bool = False,
    qualification_max_epochs: int | None = None,
    enable_wandb: bool = True,
) -> Path:
    preflight = build_temporal_fold_preflight(
        project_root=project_root,
        temporal_config_path=temporal_config_path,
        fold_index=fold_index,
        model_seed=model_seed,
        microbatch_size=microbatch_size,
        num_workers=num_workers,
        device=device,
        output_dir=output_dir,
        resume=resume,
    )
    if qualification_max_epochs is not None:
        if type(qualification_max_epochs) is not int or qualification_max_epochs <= 0:
            raise ValueError("qualification_max_epochs must be a positive integer")
        if qualification_max_epochs > preflight.max_epochs:
            raise ValueError("qualification cannot exceed the formal epoch budget")
    elif not enable_wandb:
        raise ValueError("formal training requires W&B tracking")
    if enable_wandb and preflight.wandb_api_key != "SET":
        raise RuntimeError("WANDB_API_KEY is UNSET")

    target_epochs = preflight.max_epochs
    if qualification_max_epochs is not None:
        target_epochs = min(target_epochs, qualification_max_epochs)
    print(
        "[action-conditioned-jepa] stage=preflight status=complete "
        f"config={preflight.temporal_config_id} fold={preflight.fold_index} "
        f"seed={preflight.model_seed} device={preflight.device} "
        f"microbatch={preflight.microbatch_size} "
        f"accumulation={preflight.gradient_accumulation_steps} "
        f"epochs={target_epochs} steps_per_epoch={preflight.optimizer_steps_per_epoch}",
        flush=True,
    )
    print(
        "[action-conditioned-jepa] stage=input_verification status=start",
        flush=True,
    )

    root = Path(project_root).resolve()
    target = Path(output_dir).absolute()
    temporal_path = Path(temporal_config_path)
    if not temporal_path.is_absolute():
        temporal_path = root / temporal_path
    paths = _canonical_paths(root)
    config = load_action_conditioned_jepa_config(
        model_path=paths["model_config"],
        level_path=paths["level_config"],
        temporal_sampling_path=temporal_path,
    )
    training = load_temporal_jepa_training_config(paths["training_config"])
    wandb_config = load_jepa_wandb_config(paths["wandb_config"])
    selection = load_temporal_selection_artifact(paths["selection_manifest"].parent)
    fold = selection.folds[fold_index]
    inputs = load_verified_jepa_inputs(
        source_root=paths["source_manifest"].parent,
        cache_run_manifest=paths["cache_manifest"],
        split_manifest_path=paths["split_manifest"],
        config=config,
    )
    fit_records = _load_fold_records(inputs=inputs, episode_ids=fold.fit_episode_ids)
    development_records = _load_fold_records(
        inputs=inputs,
        episode_ids=fold.development_episode_ids,
    )
    normalization_path = target / "proprio_normalization.json"
    if resume:
        normalization = load_jepa_proprio_normalization(normalization_path)
        if normalization.episode_ids != fold.fit_episode_ids:
            raise ValueError("resume normalization does not match fold-fit episodes")
    else:
        normalization = compute_jepa_proprio_normalization(
            records=fit_records,
            source_manifest_sha256=inputs.source_manifest_sha256,
            split_manifest_sha256=inputs.split_manifest_sha256,
        )
        target.mkdir(parents=True, exist_ok=False)
        write_jepa_proprio_normalization(normalization_path, normalization)
    fit_corpus = TemporalJepaCorpus(
        records=fit_records,
        indices=selection.indices,
        episode_ids=fold.fit_episode_ids,
        partition="fit",
        sampling=config.temporal_sampling,
        normalization=normalization,
    )
    development_corpus = TemporalJepaCorpus(
        records=development_records,
        indices=selection.indices,
        episode_ids=fold.development_episode_ids,
        partition="development",
        sampling=config.temporal_sampling,
        normalization=normalization,
    )
    input_hashes = {
        "source_manifest": sha256_file(paths["source_manifest"]),
        "cache_manifest": sha256_file(paths["cache_manifest"]),
        "split_manifest": sha256_file(paths["split_manifest"]),
        "selection_manifest": sha256_file(paths["selection_manifest"]),
        "model_config": sha256_file(paths["model_config"]),
        "temporal_config": sha256_file(temporal_path),
        "training_config": sha256_file(paths["training_config"]),
    }
    monitor_indices = select_monitor_indices(
        indices=development_corpus.indices,
        contexts_per_episode=training.monitor_contexts_per_episode,
    )
    monitor_positions = {
        value: position for position, value in enumerate(development_corpus.indices)
    }
    monitor_dataset = Subset(
        development_corpus,
        [monitor_positions[value] for value in monitor_indices],
    )
    deployed_development_corpus = TemporalJepaDeployedEvaluationCorpus(development_corpus)
    deployed_monitor_dataset = Subset(
        deployed_development_corpus,
        [monitor_positions[value] for value in monitor_indices],
    )
    print(
        "[action-conditioned-jepa] stage=input_verification status=complete "
        f"fit_contexts={len(fit_corpus)} development_contexts={len(development_corpus)}",
        flush=True,
    )

    torch.manual_seed(model_seed)
    torch.cuda.manual_seed_all(model_seed)
    target_device = torch.device(device)
    model = ActionConditionedJepaPredictor(
        config=config,
        proprio_normalization=normalization,
        project_root=root,
    ).to(target_device)
    optimizer = build_upstream_aligned_optimizer(model=model, config=training)
    progress = JepaTrainingProgress(
        completed_epochs=0,
        optimizer_steps=0,
        examples_seen=0,
        model_seed=model_seed,
        fold_index=fold_index,
        temporal_config_id=config.temporal_sampling.config_id,
    )
    if resume:
        latest = json.loads((target / "latest.json").read_text(encoding="utf-8"))
        progress = load_temporal_jepa_training_checkpoint(
            output_dir=target / latest["checkpoint"],
            model=model,
            optimizer=optimizer,
            expected_training_config=training,
            expected_input_sha256=input_hashes,
            expected_temporal_config_id=config.temporal_sampling.config_id,
            expected_fold_index=fold_index,
            expected_model_seed=model_seed,
        )
    wandb_run = None
    if enable_wandb:
        wandb_spec = build_wandb_run_spec(
            config=wandb_config,
            selection_id=selection.manifest["selection_id"],
            level=3,
            temporal_config_id=config.temporal_sampling.config_id,
            fold_index=fold_index,
            model_seed=model_seed,
        )
        wandb_run = initialize_wandb_run(spec=wandb_spec)
        wandb_run.config.update(
            {
                "training": asdict(training),
                "model_parameter_count": model.parameter_count,
                "microbatch_size": microbatch_size,
                "gradient_accumulation_steps": preflight.gradient_accumulation_steps,
                "fit_context_count": len(fit_corpus),
                "development_context_count": len(development_corpus),
            },
            allow_val_change=False,
        )

    total_optimizer_steps = preflight.optimizer_steps_per_epoch * training.max_epochs
    started = time.perf_counter()
    history = load_completed_temporal_jepa_history(
        run_root=target,
        completed_epochs=progress.completed_epochs,
    )
    for epoch in range(progress.completed_epochs, target_epochs):
        epoch_started = time.perf_counter()
        batches = build_epoch_microbatch_indices(
            dataset_size=len(fit_corpus),
            logical_global_batch_size=training.logical_global_batch_size,
            microbatch_size=microbatch_size,
            seed=model_seed,
            epoch=epoch,
            optimizer_steps_per_epoch=preflight.optimizer_steps_per_epoch,
        )
        loader = build_temporal_training_loader(
            dataset=fit_corpus,
            batches=batches,
            num_workers=num_workers,
            collate_fn=collate_temporal_jepa_samples,
        )

        def log_step(values: dict[str, float | int]) -> None:
            epoch_step = int(values["optimizer_step"]) - progress.optimizer_steps
            if should_emit_temporal_training_progress(
                epoch_step=epoch_step,
                optimizer_steps_per_epoch=preflight.optimizer_steps_per_epoch,
                period=training.console_progress_period_optimizer_steps,
            ):
                print(
                    format_temporal_training_progress(
                        temporal_config_id=config.temporal_sampling.config_id,
                        fold_index=fold_index,
                        model_seed=model_seed,
                        epoch=epoch + 1,
                        max_epochs=target_epochs,
                        epoch_step=epoch_step,
                        optimizer_steps_per_epoch=preflight.optimizer_steps_per_epoch,
                        global_optimizer_step=int(values["optimizer_step"]),
                        total_optimizer_steps=total_optimizer_steps,
                        examples_seen=int(values["examples_seen"]),
                        total_loss=float(values["total_loss"]),
                        gradient_norm=float(values["gradient_norm"]),
                        learning_rate=float(values["learning_rate"]),
                        weight_decay=float(values["weight_decay"]),
                        elapsed_seconds=time.perf_counter() - epoch_started,
                    ),
                    flush=True,
                )
            if (
                wandb_run is not None
                and int(values["optimizer_step"]) % wandb_config.log_every_optimizer_steps == 0
            ):
                wandb_run.log(
                    {f"train/{key}": value for key, value in values.items()},
                    step=int(values["optimizer_step"]),
                )

        result = run_temporal_jepa_epoch(
            model=model,
            optimizer=optimizer,
            microbatches=(batch.to(target_device, non_blocking=True) for batch in loader),
            training_config=training,
            progress=progress,
            optimizer_steps_this_epoch=preflight.optimizer_steps_per_epoch,
            total_optimizer_steps=total_optimizer_steps,
            optimizer_step_callback=log_step,
        )
        del loader
        progress = result.progress
        epoch_payload: dict[str, object] = {
            "epoch": progress.completed_epochs,
            "optimizer_steps": progress.optimizer_steps,
            "examples_seen": progress.examples_seen,
            "mean_total_loss": result.mean_total_loss,
            "final_gradient_norm": result.final_gradient_norm,
            "learning_rate": result.final_learning_rate,
            "weight_decay": result.final_weight_decay,
        }
        should_monitor = (
            progress.completed_epochs % training.monitor_period_epochs == 0
            or progress.completed_epochs == target_epochs
        )
        if should_monitor:
            epoch_payload["development_monitor"] = _evaluate_dataset(
                model=model,
                dataset=monitor_dataset,
                batch_size=min(microbatch_size, 8),
                device=target_device,
            )
            epoch_payload["deployed_d20_monitor"] = _evaluate_deployed_dataset(
                model=model,
                dataset=deployed_monitor_dataset,
                batch_size=min(microbatch_size, 4),
                device=target_device,
            )
        _write_epoch_metrics(
            target / "metrics" / f"epoch-{progress.completed_epochs:03d}.json",
            epoch_payload,
        )
        publish_rolling_temporal_jepa_checkpoint(
            run_root=target,
            model=model,
            optimizer=optimizer,
            progress=progress,
            training_config=training,
            input_sha256=input_hashes,
        )
        history.append(epoch_payload)
        if wandb_run is not None:
            wandb_payload = {
                "epoch/train_loss": result.mean_total_loss,
                "epoch/gradient_norm": result.final_gradient_norm,
                "epoch/examples_seen": progress.examples_seen,
            }
            if "development_monitor" in epoch_payload:
                monitor = epoch_payload["development_monitor"]
                wandb_payload["development/source_mean_latent_rmse"] = monitor[
                    "source_mean_latent_rmse"
                ]
                wandb_payload["development/source_mean_proprio_rmse"] = monitor[
                    "source_mean_proprio_rmse"
                ]
                deployed = epoch_payload["deployed_d20_monitor"]
                wandb_payload["development_d20/source_mean_deployed_latent_rmse"] = deployed[
                    "source_mean_deployed_latent_rmse"
                ]
                wandb_payload["development_d20/source_mean_deployed_proprio_rmse"] = deployed[
                    "source_mean_deployed_proprio_rmse"
                ]
                wandb_payload["development_d20/source_mean_temporal_quantization_latent_rmse"] = (
                    deployed["source_mean_temporal_quantization_latent_rmse"]
                )
                wandb_payload["development_d20/source_mean_temporal_quantization_proprio_rmse"] = (
                    deployed["source_mean_temporal_quantization_proprio_rmse"]
                )
            wandb_run.log(wandb_payload, step=progress.optimizer_steps)
        print(
            "[action-conditioned-jepa] "
            f"config={config.temporal_sampling.config_id} fold={fold_index} "
            f"seed={model_seed} epoch={progress.completed_epochs}/{target_epochs} "
            f"loss={result.mean_total_loss:.6f}",
            flush=True,
        )

    final_development = None
    if qualification_max_epochs is None:
        final_development = {
            "native": _evaluate_dataset(
                model=model,
                dataset=development_corpus,
                batch_size=min(microbatch_size, 8),
                device=target_device,
            ),
            "deployed_d20": _evaluate_deployed_dataset(
                model=model,
                dataset=deployed_development_corpus,
                batch_size=min(microbatch_size, 4),
                device=target_device,
            ),
        }
    run_manifest = target / "manifest.json"
    if run_manifest.exists():
        raise FileExistsError(run_manifest)
    _write_file_fsynced(
        run_manifest,
        (
            json.dumps(
                {
                    "schema_version": 1,
                    "format_id": "action_conditioned_jepa_fold_run_v1",
                    "qualification_only": qualification_max_epochs is not None,
                    "preflight": preflight.to_mapping(),
                    "progress": asdict(progress),
                    "final_development": final_development,
                    "wall_seconds": time.perf_counter() - started,
                    "input_sha256": input_hashes,
                    "history": history,
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode(),
    )
    _fsync_directory(target)
    if wandb_run is not None:
        wandb_run.finish()
    return run_manifest
