"""Shared final training lifecycle for level-specific Action-Conditioned JEPA models."""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors
from torch.utils.data import Subset

from latency_meta_mdp.belief.jepa.ar.metrics import (
    evaluate_deployed_temporal_batch,
    evaluate_temporal_batch,
    select_monitor_indices,
    summarize_deployed_temporal_evaluation,
    summarize_temporal_evaluation,
)
from latency_meta_mdp.belief.jepa.ar.selection import (
    build_temporal_evaluation_loader,
    build_temporal_training_loader,
    should_emit_temporal_training_progress,
)
from latency_meta_mdp.belief.jepa.ar.temporal_view import (
    TemporalJepaCorpus,
    TemporalJepaDeployedEvaluationCorpus,
    build_shared_temporal_indices,
    collate_temporal_jepa_deployed_evaluation_samples,
    collate_temporal_jepa_evaluation_samples,
    collate_temporal_jepa_samples,
)
from latency_meta_mdp.belief.jepa.ar.training import (
    JepaAdmissionProgress,
    JepaAdmissionTrainingConfig,
    build_epoch_microbatch_indices,
    build_upstream_aligned_optimizer,
    load_completed_temporal_jepa_history,
    run_temporal_jepa_epoch,
)
from latency_meta_mdp.belief.jepa.backbone import (
    ActionConditionedJepaPredictor,
)
from latency_meta_mdp.belief.jepa.config import (
    load_action_conditioned_jepa_config,
    load_jepa_temporal_sampling,
)
from latency_meta_mdp.belief.jepa.corpus import (
    compute_jepa_proprio_normalization,
    load_jepa_proprio_normalization,
    load_verified_jepa_inputs,
    load_verified_jepa_record,
    write_jepa_proprio_normalization,
)
from latency_meta_mdp.belief.jepa.tracking import (
    build_jepa_admission_wandb_run_spec,
    credential_environment_status,
    initialize_wandb_run,
    load_jepa_wandb_config,
)
from latency_meta_mdp.data.collection.artifacts import (
    _fsync_directory,
    _hash_file,
    _rename_noreplace,
    _write_file_fsynced,
)
from latency_meta_mdp.io.artifacts import sha256_file

_TEMPORAL_CONFIG_ID = "stride4_80ms_history_160ms"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CHECKPOINT_INPUTS = frozenset(
    {
        "source_manifest",
        "cache_manifest",
        "split_manifest",
        "selection_manifest",
        "model_config",
        "temporal_config",
        "training_config",
    }
)


def load_jepa_admission_training_config(path: Path) -> JepaAdmissionTrainingConfig:
    raw: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    expected = {field.name for field in fields(JepaAdmissionTrainingConfig)}
    if type(raw) is not dict or set(raw) != expected:
        raise ValueError("JEPA admission training config fields are invalid")
    raw = dict(raw)
    raw["adamw_betas"] = tuple(raw["adamw_betas"])
    raw["model_seeds"] = tuple(raw["model_seeds"])
    raw["milestone_epochs"] = tuple(raw["milestone_epochs"])
    return JepaAdmissionTrainingConfig(**raw)


def _validated_checkpoint_inputs(value: dict[str, str]) -> dict[str, str]:
    if type(value) is not dict or set(value) != _CHECKPOINT_INPUTS:
        raise ValueError("JEPA admission checkpoint input inventory is invalid")
    if any(type(item) is not str or _SHA256.fullmatch(item) is None for item in value.values()):
        raise ValueError("JEPA admission checkpoint inputs must be SHA-256 digests")
    return dict(value)


def _json_mapping(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def write_jepa_admission_training_checkpoint(
    *,
    output_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    progress: JepaAdmissionProgress,
    training_config: JepaAdmissionTrainingConfig,
    input_sha256: dict[str, str],
) -> Path:
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch module")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer")
    if not isinstance(progress, JepaAdmissionProgress):
        raise TypeError("progress must be JepaAdmissionProgress")
    if not isinstance(training_config, JepaAdmissionTrainingConfig):
        raise TypeError("training_config must be JepaAdmissionTrainingConfig")
    if (
        progress.completed_epochs > training_config.max_epochs
        or progress.examples_seen
        != progress.optimizer_steps * training_config.logical_global_batch_size
    ):
        raise ValueError("JEPA admission checkpoint progress is inconsistent")
    inputs = _validated_checkpoint_inputs(input_sha256)
    target = Path(output_dir).absolute()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f".{target.name}.building-{os.getpid()}-{uuid.uuid4().hex}"
    building.mkdir()
    try:
        model_path = building / "model.safetensors"
        _write_file_fsynced(
            model_path,
            save_safetensors(
                {
                    name: value.detach().cpu().contiguous()
                    for name, value in model.state_dict().items()
                }
            ),
        )
        optimizer_buffer = io.BytesIO()
        torch.save(optimizer.state_dict(), optimizer_buffer)
        optimizer_path = building / "optimizer.pt"
        _write_file_fsynced(optimizer_path, optimizer_buffer.getvalue())
        artifacts = {
            path.name: {"bytes": path.stat().st_size, "sha256": _hash_file(path)}
            for path in (model_path, optimizer_path)
        }
        manifest = {
            "schema_version": 1,
            "format_id": f"action_conditioned_jepa_l{progress.level}_admission_checkpoint_v1",
            "progress": asdict(progress),
            "training_config": _json_mapping(asdict(training_config)),
            "parameter_count": sum(value.numel() for value in model.parameters()),
            "input_sha256": inputs,
            "artifacts": artifacts,
        }
        _write_file_fsynced(
            building / "manifest.json",
            (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
        )
        _fsync_directory(building)
        _rename_noreplace(building, target)
        _fsync_directory(target.parent)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return target / "manifest.json"


def load_jepa_admission_training_checkpoint(
    *,
    output_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_training_config: JepaAdmissionTrainingConfig,
    expected_input_sha256: dict[str, str],
    expected_model_seed: int,
    expected_level: int = 3,
) -> JepaAdmissionProgress:
    root = Path(output_dir).resolve()
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("JEPA admission checkpoint manifest is invalid") from error
    expected_fields = {
        "schema_version",
        "format_id",
        "progress",
        "training_config",
        "parameter_count",
        "input_sha256",
        "artifacts",
    }
    if (
        type(manifest) is not dict
        or set(manifest) != expected_fields
        or manifest["schema_version"] != 1
        or manifest["format_id"]
        != f"action_conditioned_jepa_l{expected_level}_admission_checkpoint_v1"
        or manifest["training_config"] != _json_mapping(asdict(expected_training_config))
        or manifest["parameter_count"] != sum(value.numel() for value in model.parameters())
        or _validated_checkpoint_inputs(manifest["input_sha256"])
        != _validated_checkpoint_inputs(expected_input_sha256)
    ):
        raise ValueError("JEPA admission checkpoint semantics are incompatible")
    artifacts = manifest["artifacts"]
    if type(artifacts) is not dict or set(artifacts) != {"model.safetensors", "optimizer.pt"}:
        raise ValueError("JEPA admission checkpoint artifact inventory is invalid")
    for name, metadata in artifacts.items():
        path = root / name
        if (
            type(metadata) is not dict
            or set(metadata) != {"bytes", "sha256"}
            or not path.is_file()
            or path.stat().st_size != metadata["bytes"]
            or _hash_file(path) != metadata["sha256"]
        ):
            raise ValueError("JEPA admission checkpoint artifact verification failed")
    progress = JepaAdmissionProgress(**manifest["progress"])
    if progress.model_seed != expected_model_seed or progress.level != expected_level:
        raise ValueError("JEPA admission checkpoint seed is incompatible")
    model.load_state_dict(load_safetensors((root / "model.safetensors").read_bytes()), strict=True)
    optimizer.load_state_dict(
        torch.load(
            io.BytesIO((root / "optimizer.pt").read_bytes()),
            map_location="cpu",
            weights_only=True,
        )
    )
    return progress


def publish_rolling_jepa_admission_checkpoint(
    *,
    run_root: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    progress: JepaAdmissionProgress,
    training_config: JepaAdmissionTrainingConfig,
    input_sha256: dict[str, str],
) -> Path:
    if progress.completed_epochs <= 0:
        raise ValueError("rolling checkpoint requires a completed epoch")
    root = Path(run_root).absolute()
    checkpoints = root / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    latest_path = root / "latest.json"
    previous_checkpoint = None
    if latest_path.exists():
        try:
            previous = json.loads(latest_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("rolling admission pointer is invalid") from error
        if type(previous) is not dict or set(previous) != {"checkpoint", "completed_epochs"}:
            raise ValueError("rolling admission pointer fields are invalid")
        previous_checkpoint = previous["checkpoint"]
    relative = f"checkpoints/epoch-{progress.completed_epochs:03d}"
    manifest = write_jepa_admission_training_checkpoint(
        output_dir=root / relative,
        model=model,
        optimizer=optimizer,
        progress=progress,
        training_config=training_config,
        input_sha256=input_sha256,
    )
    staging = root / f".latest.json.building-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        _write_file_fsynced(
            staging,
            (
                json.dumps(
                    {"checkpoint": relative, "completed_epochs": progress.completed_epochs},
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode(),
        )
        os.replace(staging, latest_path)
        _fsync_directory(root)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    if previous_checkpoint is not None:
        match = re.fullmatch(r"checkpoints/epoch-([0-9]{3})", previous_checkpoint)
        if match is None:
            raise ValueError("rolling admission previous checkpoint path is unsafe")
        previous_epoch = int(match.group(1))
        previous_path = root / previous_checkpoint
        if (
            previous_epoch not in training_config.milestone_epochs
            and previous_path != manifest.parent
            and previous_path.is_dir()
        ):
            shutil.rmtree(previous_path)
            _fsync_directory(checkpoints)
    return manifest


def require_jepa_formal_validation_gate(
    progress: JepaAdmissionProgress,
    *,
    config: JepaAdmissionTrainingConfig,
    expected_optimizer_steps: int = 13_200,
) -> None:
    if not isinstance(progress, JepaAdmissionProgress) or not isinstance(
        config, JepaAdmissionTrainingConfig
    ):
        raise TypeError("formal validation gate received incompatible contracts")
    if (
        progress.completed_epochs != config.max_epochs
        or progress.optimizer_steps != expected_optimizer_steps
        or progress.examples_seen != expected_optimizer_steps * config.logical_global_batch_size
    ):
        raise RuntimeError("formal validation remains closed before the fixed epoch-75 checkpoint")


def load_jepa_formal_validation_records(
    *,
    inputs: Any,
    episode_ids: tuple[str, ...],
    progress: JepaAdmissionProgress,
    config: JepaAdmissionTrainingConfig,
    expected_optimizer_steps: int = 13_200,
) -> tuple[Any, ...]:
    if (
        type(episode_ids) is not tuple
        or not episode_ids
        or episode_ids != tuple(sorted(set(episode_ids)))
    ):
        raise ValueError("formal validation episode IDs must be sorted and unique")
    require_jepa_formal_validation_gate(
        progress, config=config, expected_optimizer_steps=expected_optimizer_steps
    )
    return tuple(
        load_verified_jepa_record(
            inputs,
            episode_id=episode_id,
            level=progress.level,
            split="validation",
        )
        for episode_id in episode_ids
    )


@dataclass(frozen=True)
class JepaAdmissionPreflight:
    level: int
    stage_id: str
    temporal_config_id: str
    model_seed: int
    train_master_count: int
    train_episode_count: int
    formal_validation_master_count: int
    formal_validation_episode_count: int
    train_context_count: int
    logical_global_batch_size: int
    microbatch_size: int
    gradient_accumulation_steps: int
    optimizer_steps_per_epoch: int
    max_epochs: int
    total_optimizer_steps: int
    total_examples_seen: int
    num_workers: int
    device: str
    output_dir: str
    formal_validation_opened: bool
    wandb_api_key: str

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


def _canonical_paths(root: Path, *, level: int = 3) -> dict[str, Path]:
    return {
        "model_config": root / "configs/models/jepa/model.yaml",
        "level_config": root / f"configs/models/jepa/l{level}.yaml",
        "temporal_config": (root / "configs/models/jepa/stride4_80ms_history_160ms.yaml"),
        "training_config": (
            root
            / (
                "configs/legacy/training/l3_admission.yaml"
                if level == 3
                else "configs/training/belief/autoregressive.yaml"
            )
        ),
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


def _candidate_temporal_config_paths(root: Path) -> tuple[Path, ...]:
    config_root = root / "configs/models/jepa"
    return tuple(
        config_root / name
        for name in (
            "dense_20ms_history_100ms.yaml",
            "stride2_40ms_history_120ms.yaml",
            "stride4_80ms_history_160ms.yaml",
            "stride5_100ms_history_200ms.yaml",
        )
    )


def build_jepa_admission_preflight(
    *,
    project_root: Path,
    model_seed: int,
    microbatch_size: int,
    num_workers: int,
    device: str,
    output_dir: Path,
    resume: bool = False,
    level: int = 3,
) -> JepaAdmissionPreflight:
    if type(level) is not int or level not in (1, 2, 3):
        raise ValueError("level must be 1, 2, or 3")
    root = Path(project_root).resolve()
    paths = _canonical_paths(root, level=level)
    config = load_action_conditioned_jepa_config(
        model_path=paths["model_config"],
        level_path=paths["level_config"],
        temporal_sampling_path=paths["temporal_config"],
    )
    training = load_jepa_admission_training_config(paths["training_config"])
    if config.temporal_sampling.config_id != _TEMPORAL_CONFIG_ID:
        raise ValueError("JEPA admission must use the selected stride-4 temporal config")
    if model_seed not in training.model_seeds:
        raise ValueError("model seed is outside the approved JEPA admission protocol")
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

    # Preflight reads small metadata only; it never loads RGB or DINO payloads.
    import pyarrow.parquet as pq

    source = json.loads(paths["source_manifest"].read_text(encoding="utf-8"))
    split = json.loads(paths["split_manifest"].read_text(encoding="utf-8"))
    if (
        source.get("format_id") != "structured_expert_source_corpus_v3"
        or source.get("complete") is not True
        or split.get("source_manifest_sha256") != sha256_file(paths["source_manifest"])
    ):
        raise ValueError("admission source/split identity is invalid")
    train_ids = set(split["train_episode_ids"])
    validation_ids = set(split["validation_episode_ids"])
    if train_ids & validation_ids:
        raise ValueError("admission train/validation episodes overlap")
    metadata = pq.read_table(
        paths["source_manifest"].parent / "meta/episodes.parquet",
        columns=["episode_id", "level", "logical_master_task_index", "terminal_tick"],
    ).to_pylist()
    train_rows = [
        row for row in metadata if row["level"] == level and row["episode_id"] in train_ids
    ]
    validation_rows = [
        row for row in metadata if row["level"] == level and row["episode_id"] in validation_ids
    ]
    train_masters = {row["logical_master_task_index"] for row in train_rows}
    validation_masters = {row["logical_master_task_index"] for row in validation_rows}
    train_master_count, validation_master_count = len(train_masters), len(validation_masters)
    selected_ids = tuple(sorted(row["episode_id"] for row in train_rows))
    validation_episode_count = len(validation_rows)
    if (
        (train_master_count, validation_master_count) != (80, 20)
        or train_masters & validation_masters
        or train_masters != set(split["train_master_task_indices"])
        or validation_masters != set(split["validation_master_task_indices"])
        or len(selected_ids) != 320
        or len(set(selected_ids)) != 320
        or validation_episode_count != 80
    ):
        raise ValueError("admission train/formal-validation inventory is invalid")
    # Preserve the common source start used for the admitted L3 recipe (tick 10).
    minimum_source_tick = max(
        load_jepa_temporal_sampling(path).history_span_ticks
        for path in _candidate_temporal_config_paths(root)
    )
    train_context_count = sum(
        max(0, row["terminal_tick"] - minimum_source_tick) for row in train_rows
    )
    steps_per_epoch = train_context_count // training.logical_global_batch_size
    total_steps = steps_per_epoch * training.max_epochs
    return JepaAdmissionPreflight(
        level=level,
        stage_id=f"l{level}-stride4-final-admission-v1",
        temporal_config_id=config.temporal_sampling.config_id,
        model_seed=model_seed,
        train_master_count=train_master_count,
        train_episode_count=len(selected_ids),
        formal_validation_master_count=validation_master_count,
        formal_validation_episode_count=validation_episode_count,
        train_context_count=train_context_count,
        logical_global_batch_size=training.logical_global_batch_size,
        microbatch_size=microbatch_size,
        gradient_accumulation_steps=training.logical_global_batch_size // microbatch_size,
        optimizer_steps_per_epoch=steps_per_epoch,
        max_epochs=training.max_epochs,
        total_optimizer_steps=total_steps,
        total_examples_seen=total_steps * training.logical_global_batch_size,
        num_workers=num_workers,
        device=device,
        output_dir=str(target),
        formal_validation_opened=False,
        wandb_api_key=credential_environment_status()["WANDB_API_KEY"],
    )


def _write_epoch_metrics(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_file_fsynced(
        path,
        (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
    )
    _fsync_directory(path.parent)


def _evaluate_native(*, model, dataset, batch_size: int, device: torch.device) -> dict[str, object]:
    loader = build_temporal_evaluation_loader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_temporal_jepa_evaluation_samples,
    )
    model.eval()
    metrics = tuple(
        evaluate_temporal_batch(model=model, batch=batch.to(device, non_blocking=True))
        for batch in loader
    )
    return summarize_temporal_evaluation(metrics)


def _evaluate_deployed(
    *,
    model,
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
    metrics = tuple(
        evaluate_deployed_temporal_batch(model=model, batch=batch.to(device, non_blocking=True))
        for batch in loader
    )
    return summarize_deployed_temporal_evaluation(metrics)


def _format_progress(
    *,
    model_seed: int,
    epoch: int,
    max_epochs: int,
    epoch_step: int,
    steps_per_epoch: int,
    values: dict[str, float | int],
    total_steps: int,
    elapsed_seconds: float,
) -> str:
    return (
        "[action-conditioned-jepa-admission] "
        f"config={_TEMPORAL_CONFIG_ID} seed={model_seed} "
        f"epoch={epoch}/{max_epochs} epoch_step={epoch_step}/{steps_per_epoch} "
        f"global_step={int(values['optimizer_step'])}/{total_steps} "
        f"examples={int(values['examples_seen'])} loss={float(values['total_loss']):.6f} "
        f"grad_norm={float(values['gradient_norm']):.6f} "
        f"lr={float(values['learning_rate']):g} wd={float(values['weight_decay']):g} "
        f"elapsed={elapsed_seconds:.1f}s"
    )


def execute_jepa_admission_job(
    *,
    project_root: Path,
    model_seed: int,
    microbatch_size: int,
    num_workers: int,
    device: str,
    output_dir: Path,
    resume: bool = False,
    qualification_max_epochs: int | None = None,
    enable_wandb: bool = True,
    level: int = 3,
) -> Path:
    started = time.perf_counter()
    preflight = build_jepa_admission_preflight(
        project_root=project_root,
        model_seed=model_seed,
        microbatch_size=microbatch_size,
        num_workers=num_workers,
        device=device,
        output_dir=output_dir,
        resume=resume,
        level=level,
    )
    training = load_jepa_admission_training_config(
        _canonical_paths(Path(project_root).resolve(), level=level)["training_config"]
    )
    if qualification_max_epochs is not None and (
        type(qualification_max_epochs) is not int
        or not 1 <= qualification_max_epochs <= training.max_epochs
    ):
        raise ValueError("qualification_max_epochs must lie inside the formal epoch budget")
    if qualification_max_epochs is None and not enable_wandb:
        raise ValueError("formal JEPA admission training requires W&B")
    if enable_wandb and preflight.wandb_api_key != "SET":
        raise RuntimeError("WANDB_API_KEY is UNSET")
    target_epochs = (
        training.max_epochs if qualification_max_epochs is None else qualification_max_epochs
    )
    print(
        "[action-conditioned-jepa-admission] stage=preflight status=complete "
        f"level={level} seed={model_seed} device={device} train_masters=80 train_episodes=320 "
        f"contexts={preflight.train_context_count} microbatch={microbatch_size} "
        f"accumulation={preflight.gradient_accumulation_steps} "
        f"epochs={target_epochs} steps_per_epoch={preflight.optimizer_steps_per_epoch}",
        flush=True,
    )
    print(
        "[action-conditioned-jepa-admission] stage=train_input_verification status=start",
        flush=True,
    )

    root = Path(project_root).resolve()
    target = Path(output_dir).absolute()
    paths = _canonical_paths(root, level=level)
    config = load_action_conditioned_jepa_config(
        model_path=paths["model_config"],
        level_path=paths["level_config"],
        temporal_sampling_path=paths["temporal_config"],
    )
    inputs = load_verified_jepa_inputs(
        source_root=paths["source_manifest"].parent,
        cache_run_manifest=paths["cache_manifest"],
        split_manifest_path=paths["split_manifest"],
        config=config,
    )
    level_ids = set(inputs.source.episode_ids(level=level))
    train_ids = tuple(sorted(level_ids.intersection(inputs.split.train_episode_ids)))
    validation_ids = tuple(sorted(level_ids.intersection(inputs.split.validation_episode_ids)))
    if len(train_ids) != preflight.train_episode_count or len(validation_ids) != 80:
        raise ValueError("verified execution inputs disagree with the admission preflight")
    train_records = tuple(
        load_verified_jepa_record(
            inputs,
            episode_id=episode_id,
            level=level,
            split="train",
        )
        for episode_id in train_ids
    )
    input_hashes = {
        "source_manifest": sha256_file(paths["source_manifest"]),
        "cache_manifest": sha256_file(paths["cache_manifest"]),
        "split_manifest": sha256_file(paths["split_manifest"]),
        "selection_manifest": sha256_file(paths["selection_manifest"]),
        "model_config": sha256_file(paths["model_config"]),
        "temporal_config": sha256_file(paths["temporal_config"]),
        "training_config": sha256_file(paths["training_config"]),
    }
    normalization_path = target / "proprio_normalization.json"
    if resume:
        if (target / "manifest.json").exists():
            raise FileExistsError("completed JEPA admission run cannot be resumed")
        normalization = load_jepa_proprio_normalization(normalization_path)
        if normalization.episode_ids != train_ids:
            raise ValueError("resume normalization does not match all train-pool episodes")
    else:
        normalization = compute_jepa_proprio_normalization(
            records=train_records,
            source_manifest_sha256=inputs.source_manifest_sha256,
            split_manifest_sha256=inputs.split_manifest_sha256,
        )
        target.mkdir(parents=True, exist_ok=False)
        write_jepa_proprio_normalization(normalization_path, normalization)
    train_corpus = TemporalJepaCorpus(
        records=train_records,
        indices=build_shared_temporal_indices(
            records=train_records,
            samplings=tuple(
                load_jepa_temporal_sampling(path) for path in _candidate_temporal_config_paths(root)
            ),
        ),
        episode_ids=train_ids,
        partition="fit",
        sampling=config.temporal_sampling,
        normalization=normalization,
    )
    if len(train_corpus) != preflight.train_context_count:
        raise ValueError("train corpus context count changed after preflight")
    monitor_indices = select_monitor_indices(
        indices=train_corpus.indices,
        contexts_per_episode=training.monitor_contexts_per_episode,
    )
    positions = {value: index for index, value in enumerate(train_corpus.indices)}
    monitor_dataset = Subset(train_corpus, [positions[value] for value in monitor_indices])
    print(
        "[action-conditioned-jepa-admission] stage=train_input_verification status=complete "
        f"train_contexts={len(train_corpus)} train_pool_monitor_contexts={len(monitor_dataset)} "
        "formal_validation_opened=false",
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
    progress = JepaAdmissionProgress(
        completed_epochs=0,
        optimizer_steps=0,
        examples_seen=0,
        model_seed=model_seed,
        temporal_config_id=_TEMPORAL_CONFIG_ID,
        stage_id=preflight.stage_id,
        level=level,
    )
    history: list[dict[str, object]] = []
    if resume:
        latest = json.loads((target / "latest.json").read_text(encoding="utf-8"))
        progress = load_jepa_admission_training_checkpoint(
            output_dir=target / latest["checkpoint"],
            model=model,
            optimizer=optimizer,
            expected_training_config=training,
            expected_input_sha256=input_hashes,
            expected_model_seed=model_seed,
            expected_level=level,
        )
        history = load_completed_temporal_jepa_history(
            run_root=target,
            completed_epochs=progress.completed_epochs,
        )

    wandb_config = load_jepa_wandb_config(paths["wandb_config"])
    wandb_run = None
    if enable_wandb:
        wandb_spec = build_jepa_admission_wandb_run_spec(
            config=wandb_config,
            stage_id=preflight.stage_id,
            temporal_config_id=_TEMPORAL_CONFIG_ID,
            model_seed=model_seed,
            level=level,
        )
        wandb_run = initialize_wandb_run(spec=wandb_spec)
        wandb_run.config.update(
            {
                "training": asdict(training),
                "model_parameter_count": model.parameter_count,
                "microbatch_size": microbatch_size,
                "gradient_accumulation_steps": preflight.gradient_accumulation_steps,
                "train_context_count": len(train_corpus),
                "formal_validation_opened_during_training": False,
            },
            allow_val_change=False,
        )

    for epoch in range(progress.completed_epochs, target_epochs):
        epoch_started = time.perf_counter()
        batches = build_epoch_microbatch_indices(
            dataset_size=len(train_corpus),
            logical_global_batch_size=training.logical_global_batch_size,
            microbatch_size=microbatch_size,
            seed=model_seed,
            epoch=epoch,
            optimizer_steps_per_epoch=preflight.optimizer_steps_per_epoch,
        )
        loader = build_temporal_training_loader(
            dataset=train_corpus,
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
                    _format_progress(
                        model_seed=model_seed,
                        epoch=epoch + 1,
                        max_epochs=target_epochs,
                        epoch_step=epoch_step,
                        steps_per_epoch=preflight.optimizer_steps_per_epoch,
                        values=values,
                        total_steps=preflight.total_optimizer_steps,
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
            total_optimizer_steps=preflight.total_optimizer_steps,
            optimizer_step_callback=log_step,
        )
        del loader
        if not isinstance(result.progress, JepaAdmissionProgress):
            raise TypeError("admission epoch returned cross-validation progress")
        progress = result.progress
        epoch_payload: dict[str, object] = {
            "epoch": progress.completed_epochs,
            "optimizer_steps": progress.optimizer_steps,
            "examples_seen": progress.examples_seen,
            "mean_total_loss": result.mean_total_loss,
            "final_gradient_norm": result.final_gradient_norm,
            "learning_rate": result.final_learning_rate,
            "weight_decay": result.final_weight_decay,
            "formal_validation_opened": False,
        }
        should_monitor = (
            progress.completed_epochs % training.monitor_period_epochs == 0
            or progress.completed_epochs == target_epochs
        )
        if should_monitor:
            epoch_payload["train_pool_monitor"] = _evaluate_native(
                model=model,
                dataset=monitor_dataset,
                batch_size=min(microbatch_size, 8),
                device=target_device,
            )
        _write_epoch_metrics(
            target / "metrics" / f"epoch-{progress.completed_epochs:03d}.json",
            epoch_payload,
        )
        publish_rolling_jepa_admission_checkpoint(
            run_root=target,
            model=model,
            optimizer=optimizer,
            progress=progress,
            training_config=training,
            input_sha256=input_hashes,
        )
        history.append(epoch_payload)
        if wandb_run is not None:
            wandb_payload: dict[str, float | int] = {
                "epoch/train_loss": result.mean_total_loss,
                "epoch/gradient_norm": result.final_gradient_norm,
                "epoch/examples_seen": progress.examples_seen,
            }
            if "train_pool_monitor" in epoch_payload:
                monitor = epoch_payload["train_pool_monitor"]
                wandb_payload["train_pool_monitor/native_latent_rmse"] = monitor[
                    "source_mean_latent_rmse"
                ]
                wandb_payload["train_pool_monitor/native_proprio_rmse"] = monitor[
                    "source_mean_proprio_rmse"
                ]
            wandb_run.log(wandb_payload, step=progress.optimizer_steps)
        print(
            "[action-conditioned-jepa-admission] "
            f"seed={model_seed} epoch={progress.completed_epochs}/{target_epochs} "
            f"loss={result.mean_total_loss:.6f} formal_validation_opened=false",
            flush=True,
        )

    formal_validation = None
    formal_validation_opened = False
    if qualification_max_epochs is None:
        final_checkpoint = target / "checkpoints/epoch-075/manifest.json"
        if not final_checkpoint.is_file():
            raise RuntimeError("formal validation requires the published epoch-75 checkpoint")
        validation_records = load_jepa_formal_validation_records(
            inputs=inputs,
            episode_ids=validation_ids,
            progress=progress,
            config=training,
            expected_optimizer_steps=preflight.total_optimizer_steps,
        )
        samplings = tuple(
            load_jepa_temporal_sampling(path) for path in _candidate_temporal_config_paths(root)
        )
        validation_indices = build_shared_temporal_indices(
            records=validation_records,
            samplings=samplings,
        )
        validation_corpus = TemporalJepaCorpus(
            records=validation_records,
            indices=validation_indices,
            episode_ids=validation_ids,
            partition="development",
            sampling=config.temporal_sampling,
            normalization=normalization,
        )
        formal_validation_opened = True
        print(
            "[action-conditioned-jepa-admission] stage=formal_validation status=start "
            f"seed={model_seed} contexts={len(validation_corpus)}",
            flush=True,
        )
        formal_validation = {
            "native": _evaluate_native(
                model=model,
                dataset=validation_corpus,
                batch_size=min(microbatch_size, 8),
                device=target_device,
            ),
            "deployed_d20": _evaluate_deployed(
                model=model,
                dataset=TemporalJepaDeployedEvaluationCorpus(validation_corpus),
                batch_size=min(microbatch_size, 4),
                device=target_device,
            ),
        }
        print(
            "[action-conditioned-jepa-admission] stage=formal_validation status=complete "
            f"seed={model_seed}",
            flush=True,
        )

    run_manifest = target / "manifest.json"
    if run_manifest.exists():
        raise FileExistsError(run_manifest)
    _write_file_fsynced(
        run_manifest,
        (
            json.dumps(
                {
                    "schema_version": 1,
                    "format_id": f"action_conditioned_jepa_l{level}_admission_run_v1",
                    "qualification_only": qualification_max_epochs is not None,
                    "preflight": preflight.to_mapping(),
                    "progress": asdict(progress),
                    "formal_validation_opened": formal_validation_opened,
                    "final_formal_validation": formal_validation,
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
