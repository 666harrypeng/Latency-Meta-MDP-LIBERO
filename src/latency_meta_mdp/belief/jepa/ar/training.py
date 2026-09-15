"""Training objective for the temporal Action-Conditioned JEPA predictor."""

from __future__ import annotations

import io
import json
import math
import os
import re
import shutil
import uuid
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
import yaml
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

from latency_meta_mdp.belief.jepa.ar.temporal_view import TemporalJepaBatch
from latency_meta_mdp.data.collection.artifacts import (
    _fsync_directory,
    _hash_file,
    _rename_noreplace,
    _write_file_fsynced,
)

_CHECKPOINT_FORMAT = "action_conditioned_jepa_epoch_checkpoint_v1"
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


@dataclass(frozen=True)
class TemporalJepaLossTerms:
    teacher_visual: torch.Tensor
    teacher_proprio: torch.Tensor
    rollout_visual: torch.Tensor
    rollout_proprio: torch.Tensor
    total: torch.Tensor


@dataclass(frozen=True)
class TemporalJepaTrainingConfig:
    schema_version: int
    config_id: str
    max_epochs: int
    logical_global_batch_size: int
    learning_rate_start: float
    learning_rate_reference: float
    learning_rate_final: float
    weight_decay_start: float
    weight_decay_final: float
    schedule_scale: float
    warmup_epochs: int
    adamw_betas: tuple[float, float]
    adamw_epsilon: float
    gradient_clip_norm: float
    selection_seed: int
    confirmation_seeds: tuple[int, int]
    monitor_period_epochs: int
    monitor_contexts_per_episode: int
    checkpoint_period_epochs: int
    milestone_epochs: tuple[int, int, int]
    console_progress_period_optimizer_steps: int

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.config_id != "l3_temporal_selection_jepa_wm_recipe":
            raise ValueError("unsupported JEPA temporal-selection training config")
        if self.max_epochs != 50 or self.logical_global_batch_size != 256:
            raise ValueError("training must preserve the JEPA-WM 50-epoch global-256 recipe")
        floats = (
            self.learning_rate_start,
            self.learning_rate_reference,
            self.learning_rate_final,
            self.weight_decay_start,
            self.weight_decay_final,
            self.schedule_scale,
            self.adamw_epsilon,
            self.gradient_clip_norm,
        )
        if any(
            type(value) is not float or not math.isfinite(value) or value <= 0 for value in floats
        ):
            raise ValueError("JEPA training floating-point values must be finite and positive")
        if (
            self.learning_rate_start,
            self.learning_rate_reference,
            self.learning_rate_final,
            self.weight_decay_start,
            self.weight_decay_final,
            self.schedule_scale,
            self.warmup_epochs,
            self.adamw_betas,
            self.adamw_epsilon,
            self.gradient_clip_norm,
        ) != (5e-4, 5e-4, 5e-4, 1e-7, 1e-6, 1.0, 0, (0.9, 0.999), 1e-8, 1.0):
            raise ValueError("training optimizer semantics disagree with the upstream recipe")
        if (
            type(self.selection_seed) is not int
            or self.selection_seed != 7
            or self.confirmation_seeds != (17, 27)
        ):
            raise ValueError("training seed protocol is invalid")
        if (
            self.monitor_period_epochs,
            self.monitor_contexts_per_episode,
            self.checkpoint_period_epochs,
        ) != (5, 4, 1):
            raise ValueError("training monitoring/checkpoint protocol is invalid")
        if self.milestone_epochs != (20, 40, 50):
            raise ValueError("training milestone epochs are invalid")
        if self.console_progress_period_optimizer_steps != 5:
            raise ValueError("training console progress period is invalid")


@dataclass(frozen=True)
class JepaAdmissionTrainingConfig:
    schema_version: int
    config_id: str
    max_epochs: int
    logical_global_batch_size: int
    learning_rate_start: float
    learning_rate_reference: float
    learning_rate_final: float
    weight_decay_start: float
    weight_decay_final: float
    schedule_scale: float
    warmup_epochs: int
    adamw_betas: tuple[float, float]
    adamw_epsilon: float
    gradient_clip_norm: float
    model_seeds: tuple[int, int, int]
    monitor_period_epochs: int
    monitor_contexts_per_episode: int
    checkpoint_period_epochs: int
    milestone_epochs: tuple[int, int, int]
    console_progress_period_optimizer_steps: int

    def __post_init__(self) -> None:
        if self.config_id not in {
            "l3_stride4_admission_jepa_wm_recipe",
            "stride4_admission_jepa_wm_recipe",
        }:
            raise ValueError("unsupported JEPA admission training config")
        observed = (
            self.schema_version,
            self.config_id,
            self.max_epochs,
            self.logical_global_batch_size,
            self.learning_rate_start,
            self.learning_rate_reference,
            self.learning_rate_final,
            self.weight_decay_start,
            self.weight_decay_final,
            self.schedule_scale,
            self.warmup_epochs,
            self.adamw_betas,
            self.adamw_epsilon,
            self.gradient_clip_norm,
            self.model_seeds,
            self.monitor_period_epochs,
            self.monitor_contexts_per_episode,
            self.checkpoint_period_epochs,
            self.milestone_epochs,
            self.console_progress_period_optimizer_steps,
        )
        expected = (
            1,
            self.config_id,
            75,
            256,
            5e-4,
            5e-4,
            5e-4,
            1e-7,
            1e-6,
            1.0,
            0,
            (0.9, 0.999),
            1e-8,
            1.0,
            (7, 17, 27),
            5,
            4,
            1,
            (25, 50, 75),
            5,
        )
        if observed != expected:
            raise ValueError("unsupported JEPA admission training config")


@dataclass(frozen=True)
class JepaTrainingProgress:
    completed_epochs: int
    optimizer_steps: int
    examples_seen: int
    model_seed: int
    fold_index: int
    temporal_config_id: str

    def __post_init__(self) -> None:
        for name in (
            "completed_epochs",
            "optimizer_steps",
            "examples_seen",
            "model_seed",
            "fold_index",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if type(self.temporal_config_id) is not str or not self.temporal_config_id:
            raise ValueError("temporal_config_id cannot be empty")


@dataclass(frozen=True)
class JepaAdmissionProgress:
    completed_epochs: int
    optimizer_steps: int
    examples_seen: int
    model_seed: int
    temporal_config_id: str
    stage_id: str
    level: int

    def __post_init__(self) -> None:
        for name in ("completed_epochs", "optimizer_steps", "examples_seen", "model_seed"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.temporal_config_id != "stride4_80ms_history_160ms":
            raise ValueError("JEPA admission progress must use stride-4")
        if type(self.level) is not int or self.level not in (1, 2, 3):
            raise ValueError("admission level must be 1, 2, or 3")
        if self.stage_id != f"l{self.level}-stride4-final-admission-v1":
            raise ValueError("JEPA admission progress identity is invalid")
        if self.model_seed not in (7, 17, 27):
            raise ValueError("JEPA admission progress seed is invalid")


@dataclass(frozen=True)
class TemporalJepaEpochResult:
    progress: JepaTrainingProgress | JepaAdmissionProgress
    microbatch_count: int
    optimizer_step_count: int
    mean_total_loss: float
    final_gradient_norm: float
    final_learning_rate: float
    final_weight_decay: float


def load_temporal_jepa_training_config(path: Path) -> TemporalJepaTrainingConfig:
    raw: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    expected = {field.name for field in fields(TemporalJepaTrainingConfig)}
    if type(raw) is not dict or set(raw) != expected:
        raise ValueError("JEPA training config fields are invalid")
    raw = dict(raw)
    raw["adamw_betas"] = tuple(raw["adamw_betas"])
    raw["confirmation_seeds"] = tuple(raw["confirmation_seeds"])
    raw["milestone_epochs"] = tuple(raw["milestone_epochs"])
    return TemporalJepaTrainingConfig(**raw)


def build_upstream_aligned_optimizer(
    *,
    model: torch.nn.Module,
    config: TemporalJepaTrainingConfig | JepaAdmissionTrainingConfig,
) -> torch.optim.AdamW:
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch module")
    if not isinstance(config, (TemporalJepaTrainingConfig, JepaAdmissionTrainingConfig)):
        raise TypeError("config must be a supported JEPA training config")
    decayed = []
    excluded = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "bias" in name or parameter.ndim == 1:
            excluded.append(parameter)
        else:
            decayed.append(parameter)
    if not decayed or not excluded:
        raise ValueError("model parameters cannot form upstream AdamW groups")
    return torch.optim.AdamW(
        (
            {"params": decayed, "weight_decay": config.weight_decay_start},
            {
                "params": excluded,
                "weight_decay": 0.0,
                "WD_exclude": True,
            },
        ),
        lr=config.learning_rate_reference,
        betas=config.adamw_betas,
        eps=config.adamw_epsilon,
    )


def apply_upstream_optimizer_schedule(
    *,
    optimizer: torch.optim.Optimizer,
    config: TemporalJepaTrainingConfig | JepaAdmissionTrainingConfig,
    optimizer_step: int,
    total_optimizer_steps: int,
) -> tuple[float, float]:
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer")
    if not isinstance(config, (TemporalJepaTrainingConfig, JepaAdmissionTrainingConfig)):
        raise TypeError("config must be a supported JEPA training config")
    if (
        type(optimizer_step) is not int
        or type(total_optimizer_steps) is not int
        or not 1 <= optimizer_step <= total_optimizer_steps
        or total_optimizer_steps <= 0
    ):
        raise ValueError("optimizer schedule step is invalid")
    progress = optimizer_step / total_optimizer_steps
    learning_rate = config.learning_rate_final + (
        config.learning_rate_reference - config.learning_rate_final
    ) * 0.5 * (1.0 + math.cos(math.pi * progress))
    weight_decay = config.weight_decay_final + (
        config.weight_decay_start - config.weight_decay_final
    ) * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
        if not group.get("WD_exclude", False):
            group["weight_decay"] = weight_decay
    return learning_rate, weight_decay


def build_epoch_microbatch_indices(
    *,
    dataset_size: int,
    logical_global_batch_size: int,
    microbatch_size: int,
    seed: int,
    epoch: int,
    optimizer_steps_per_epoch: int,
) -> tuple[tuple[int, ...], ...]:
    for name, value in (
        ("dataset_size", dataset_size),
        ("logical_global_batch_size", logical_global_batch_size),
        ("microbatch_size", microbatch_size),
        ("optimizer_steps_per_epoch", optimizer_steps_per_epoch),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(seed) is not int or seed < 0 or type(epoch) is not int or epoch < 0:
        raise ValueError("seed and epoch must be nonnegative integers")
    if logical_global_batch_size % microbatch_size:
        raise ValueError("microbatch_size must divide the logical global batch")
    example_count = optimizer_steps_per_epoch * logical_global_batch_size
    if example_count > dataset_size:
        raise ValueError("optimizer_steps_per_epoch exceeds the available dataset")
    generator = torch.Generator().manual_seed(seed + epoch)
    order = torch.randperm(dataset_size, generator=generator)[:example_count].tolist()
    return tuple(
        tuple(order[start : start + microbatch_size])
        for start in range(0, example_count, microbatch_size)
    )


def _json_mapping(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def _validated_checkpoint_inputs(value: dict[str, str]) -> dict[str, str]:
    if type(value) is not dict or set(value) != _CHECKPOINT_INPUTS:
        raise ValueError("training checkpoint input inventory is invalid")
    if any(type(item) is not str or _SHA256.fullmatch(item) is None for item in value.values()):
        raise ValueError("training checkpoint inputs must be SHA-256 digests")
    return dict(value)


def write_temporal_jepa_training_checkpoint(
    *,
    output_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    progress: JepaTrainingProgress,
    training_config: TemporalJepaTrainingConfig,
    input_sha256: dict[str, str],
) -> Path:
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch module")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer")
    if not isinstance(progress, JepaTrainingProgress):
        raise TypeError("progress must be JepaTrainingProgress")
    if not isinstance(training_config, TemporalJepaTrainingConfig):
        raise TypeError("training_config must be TemporalJepaTrainingConfig")
    inputs = _validated_checkpoint_inputs(input_sha256)
    target = Path(output_dir).absolute()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f".{target.name}.building-{os.getpid()}-{uuid.uuid4().hex}"
    building.mkdir()
    try:
        model_payload = save_safetensors(
            {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
        )
        model_path = building / "model.safetensors"
        _write_file_fsynced(model_path, model_payload)
        optimizer_buffer = io.BytesIO()
        torch.save(optimizer.state_dict(), optimizer_buffer)
        optimizer_path = building / "optimizer.pt"
        _write_file_fsynced(optimizer_path, optimizer_buffer.getvalue())
        artifacts = {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": _hash_file(path),
            }
            for path in (model_path, optimizer_path)
        }
        manifest = {
            "schema_version": 1,
            "format_id": _CHECKPOINT_FORMAT,
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


def load_temporal_jepa_training_checkpoint(
    *,
    output_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_training_config: TemporalJepaTrainingConfig,
    expected_input_sha256: dict[str, str],
    expected_temporal_config_id: str,
    expected_fold_index: int,
    expected_model_seed: int,
) -> JepaTrainingProgress:
    root = Path(output_dir).resolve()
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("training checkpoint manifest is invalid") from error
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
        or manifest["format_id"] != _CHECKPOINT_FORMAT
        or manifest["training_config"] != _json_mapping(asdict(expected_training_config))
        or manifest["parameter_count"] != sum(value.numel() for value in model.parameters())
    ):
        raise ValueError("training checkpoint semantics are incompatible")
    expected_inputs = _validated_checkpoint_inputs(expected_input_sha256)
    if _validated_checkpoint_inputs(manifest["input_sha256"]) != expected_inputs:
        raise ValueError("training checkpoint inputs are incompatible")
    artifacts = manifest["artifacts"]
    if type(artifacts) is not dict or set(artifacts) != {
        "model.safetensors",
        "optimizer.pt",
    }:
        raise ValueError("training checkpoint artifact inventory is invalid")
    for name, metadata in artifacts.items():
        path = root / name
        if (
            type(metadata) is not dict
            or set(metadata) != {"bytes", "sha256"}
            or not path.is_file()
            or path.stat().st_size != metadata["bytes"]
            or _hash_file(path) != metadata["sha256"]
        ):
            raise ValueError("training checkpoint artifact verification failed")
    progress = JepaTrainingProgress(**manifest["progress"])
    if (
        progress.temporal_config_id != expected_temporal_config_id
        or progress.fold_index != expected_fold_index
        or progress.model_seed != expected_model_seed
    ):
        raise ValueError("training checkpoint progress identity is incompatible")
    model.load_state_dict(
        load_safetensors((root / "model.safetensors").read_bytes()),
        strict=True,
    )
    optimizer_state = torch.load(
        io.BytesIO((root / "optimizer.pt").read_bytes()),
        map_location="cpu",
        weights_only=True,
    )
    optimizer.load_state_dict(optimizer_state)
    return progress


def load_completed_temporal_jepa_history(
    *,
    run_root: Path,
    completed_epochs: int,
) -> list[dict[str, object]]:
    if type(completed_epochs) is not int or completed_epochs < 0:
        raise ValueError("completed_epochs must be a nonnegative integer")
    metrics_root = Path(run_root).resolve() / "metrics"
    history = []
    for epoch in range(1, completed_epochs + 1):
        path = metrics_root / f"epoch-{epoch:03d}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("completed epoch history is invalid") from error
        if type(payload) is not dict or payload.get("epoch") != epoch:
            raise ValueError("completed epoch history is invalid")
        history.append(payload)
    return history


def publish_rolling_temporal_jepa_checkpoint(
    *,
    run_root: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    progress: JepaTrainingProgress,
    training_config: TemporalJepaTrainingConfig,
    input_sha256: dict[str, str],
) -> Path:
    if progress.completed_epochs <= 0:
        raise ValueError("rolling checkpoint requires at least one completed epoch")
    root = Path(run_root).absolute()
    checkpoints = root / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    latest_path = root / "latest.json"
    previous_checkpoint = None
    if latest_path.exists():
        try:
            previous = json.loads(latest_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("rolling latest pointer is invalid") from error
        if type(previous) is not dict or set(previous) != {
            "checkpoint",
            "completed_epochs",
        }:
            raise ValueError("rolling latest pointer fields are invalid")
        previous_checkpoint = previous["checkpoint"]
    relative = f"checkpoints/epoch-{progress.completed_epochs:03d}"
    manifest = write_temporal_jepa_training_checkpoint(
        output_dir=root / relative,
        model=model,
        optimizer=optimizer,
        progress=progress,
        training_config=training_config,
        input_sha256=input_sha256,
    )
    pointer = {
        "checkpoint": relative,
        "completed_epochs": progress.completed_epochs,
    }
    staging = root / f".latest.json.building-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        _write_file_fsynced(
            staging,
            (json.dumps(pointer, indent=2, sort_keys=True) + "\n").encode(),
        )
        os.replace(staging, latest_path)
        _fsync_directory(root)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    if previous_checkpoint is not None:
        match = re.fullmatch(r"checkpoints/epoch-([0-9]{3})", previous_checkpoint)
        if match is None:
            raise ValueError("rolling previous checkpoint path is unsafe")
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


def run_temporal_jepa_epoch(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    microbatches: Iterable[TemporalJepaBatch],
    training_config: TemporalJepaTrainingConfig | JepaAdmissionTrainingConfig,
    progress: JepaTrainingProgress | JepaAdmissionProgress,
    optimizer_steps_this_epoch: int,
    total_optimizer_steps: int,
    optimizer_step_callback: Callable[[dict[str, float | int]], None] | None = None,
) -> TemporalJepaEpochResult:
    if not isinstance(model, torch.nn.Module) or not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("model and optimizer have invalid types")
    if not isinstance(
        training_config,
        (TemporalJepaTrainingConfig, JepaAdmissionTrainingConfig),
    ) or not isinstance(progress, (JepaTrainingProgress, JepaAdmissionProgress)):
        raise TypeError("training config or progress has an invalid type")
    if (
        type(optimizer_steps_this_epoch) is not int
        or optimizer_steps_this_epoch <= 0
        or progress.optimizer_steps + optimizer_steps_this_epoch > total_optimizer_steps
    ):
        raise ValueError("epoch optimizer-step range is invalid")
    iterator = iter(microbatches)
    model.train()
    loss_sum = 0.0
    microbatch_count = 0
    examples_seen = progress.examples_seen
    final_gradient_norm = 0.0
    final_learning_rate = 0.0
    final_weight_decay = 0.0
    parameter = next(model.parameters())
    device_type = parameter.device.type

    for step_offset in range(optimizer_steps_this_epoch):
        try:
            first = next(iterator)
        except StopIteration as error:
            raise ValueError("epoch ended before one logical batch was complete") from error
        if not isinstance(first, TemporalJepaBatch):
            raise TypeError("microbatches must yield TemporalJepaBatch")
        if training_config.logical_global_batch_size % first.batch_size:
            raise ValueError("hardware microbatch must divide the logical global batch")
        accumulation_steps = training_config.logical_global_batch_size // first.batch_size
        final_learning_rate, final_weight_decay = apply_upstream_optimizer_schedule(
            optimizer=optimizer,
            config=training_config,
            optimizer_step=progress.optimizer_steps + step_offset + 1,
            total_optimizer_steps=total_optimizer_steps,
        )
        optimizer.zero_grad(set_to_none=True)
        batch = first
        logical_loss_sum = 0.0
        for microbatch_index in range(accumulation_steps):
            if (
                not isinstance(batch, TemporalJepaBatch)
                or batch.batch_size != first.batch_size
                or batch.temporal_config_id != progress.temporal_config_id
            ):
                raise ValueError("logical batch contains incompatible microbatches")
            with torch.autocast(
                device_type=device_type,
                dtype=torch.bfloat16,
                enabled=device_type == "cuda",
            ):
                terms = compute_temporal_jepa_loss(model=model, batch=batch)
                scaled_loss = terms.total / accumulation_steps
            scaled_loss.backward()
            loss_sum += float(terms.total.detach())
            logical_loss_sum += float(terms.total.detach())
            microbatch_count += 1
            examples_seen += batch.batch_size
            if microbatch_index + 1 < accumulation_steps:
                try:
                    batch = next(iterator)
                except StopIteration as error:
                    raise ValueError("epoch ended inside a logical batch") from error
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            training_config.gradient_clip_norm,
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("JEPA gradient norm is nonfinite")
        final_gradient_norm = float(gradient_norm.detach())
        optimizer.step()
        if optimizer_step_callback is not None:
            optimizer_step_callback(
                {
                    "optimizer_step": progress.optimizer_steps + step_offset + 1,
                    "examples_seen": examples_seen,
                    "total_loss": logical_loss_sum / accumulation_steps,
                    "gradient_norm": final_gradient_norm,
                    "learning_rate": final_learning_rate,
                    "weight_decay": final_weight_decay,
                }
            )
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise ValueError("epoch yielded more microbatches than declared")
    completed = replace(
        progress,
        completed_epochs=progress.completed_epochs + 1,
        optimizer_steps=progress.optimizer_steps + optimizer_steps_this_epoch,
        examples_seen=examples_seen,
    )
    return TemporalJepaEpochResult(
        progress=completed,
        microbatch_count=microbatch_count,
        optimizer_step_count=optimizer_steps_this_epoch,
        mean_total_loss=loss_sum / microbatch_count,
        final_gradient_norm=final_gradient_norm,
        final_learning_rate=final_learning_rate,
        final_weight_decay=final_weight_decay,
    )


def compute_temporal_jepa_loss(
    *,
    model: torch.nn.Module,
    batch: TemporalJepaBatch,
    visual_weight: float = 1.0,
    proprio_weight: float = 1.0,
) -> TemporalJepaLossTerms:
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch module")
    if not isinstance(batch, TemporalJepaBatch):
        raise TypeError("batch must be a TemporalJepaBatch")
    if (
        type(visual_weight) is not float
        or type(proprio_weight) is not float
        or visual_weight <= 0
        or proprio_weight <= 0
    ):
        raise ValueError("modality weights must be positive floats")
    context = batch.context
    teacher_visual_prediction, teacher_proprio_prediction = model.predict_next(
        vision_history=context.vision_history,
        proprio_history=context.proprio_history,
        executed_controls=context.executed_controls,
        outgoing_control=context.executable_controls[:, 0],
    )
    rollout_visual_prediction, rollout_proprio_prediction = model.rollout_endpoint_for_loss(
        context,
        horizon=2,
    )
    teacher_visual = functional.mse_loss(
        teacher_visual_prediction.float(),
        batch.target_visual_latents[:, 0].float(),
    )
    teacher_proprio = functional.mse_loss(
        teacher_proprio_prediction.float(),
        batch.target_proprio_normalized[:, 0].float(),
    )
    rollout_visual = functional.mse_loss(
        rollout_visual_prediction.float(),
        batch.target_visual_latents[:, 1].float(),
    )
    rollout_proprio = functional.mse_loss(
        rollout_proprio_prediction.float(),
        batch.target_proprio_normalized[:, 1].float(),
    )
    total = visual_weight * (teacher_visual + rollout_visual) + proprio_weight * (
        teacher_proprio + rollout_proprio
    )
    terms = (teacher_visual, teacher_proprio, rollout_visual, rollout_proprio, total)
    if not all(bool(torch.isfinite(value)) for value in terms):
        raise FloatingPointError("JEPA training loss contains a nonfinite value")
    return TemporalJepaLossTerms(
        teacher_visual=teacher_visual,
        teacher_proprio=teacher_proprio,
        rollout_visual=rollout_visual,
        rollout_proprio=rollout_proprio,
        total=total,
    )
