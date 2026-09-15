"""Secret-free deterministic Weights & Biases run configuration."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class JepaWandbConfig:
    schema_version: int
    config_id: str
    enabled: bool
    project: str
    group: str
    api_key_env: str
    resume: str
    log_every_optimizer_steps: int
    log_media: bool

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.config_id != "action_conditioned_jepa_wandb"
            or self.enabled is not True
            or self.project != "latency-meta-mdp-action-conditioned-jepa"
            or self.group != "l3-temporal-selection"
            or self.api_key_env != "WANDB_API_KEY"
            or self.resume != "allow"
            or self.log_media is not False
        ):
            raise ValueError("unsupported JEPA W&B semantics")
        if type(self.log_every_optimizer_steps) is not int or self.log_every_optimizer_steps <= 0:
            raise ValueError("W&B log period must be a positive integer")


@dataclass(frozen=True)
class WandbRunSpec:
    project: str
    group: str
    name: str
    run_id: str
    resume: str
    logged_config: dict[str, Any]


def load_jepa_wandb_config(path: Path) -> JepaWandbConfig:
    raw: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    expected = {field.name for field in fields(JepaWandbConfig)}
    if type(raw) is not dict or set(raw) != expected:
        raise ValueError("JEPA W&B config fields are invalid")
    return JepaWandbConfig(**raw)


def build_wandb_run_spec(
    *,
    config: JepaWandbConfig,
    selection_id: str,
    level: int,
    temporal_config_id: str,
    fold_index: int,
    model_seed: int,
) -> WandbRunSpec:
    if not isinstance(config, JepaWandbConfig):
        raise TypeError("config must be JepaWandbConfig")
    if type(selection_id) is not str or not selection_id:
        raise ValueError("selection_id cannot be empty")
    if type(level) is not int or level not in (1, 2, 3):
        raise ValueError("level must be 1, 2, or 3")
    if type(temporal_config_id) is not str or not temporal_config_id:
        raise ValueError("temporal_config_id cannot be empty")
    if type(fold_index) is not int or fold_index < 0:
        raise ValueError("fold_index must be nonnegative")
    if type(model_seed) is not int or model_seed < 0:
        raise ValueError("model_seed must be nonnegative")
    logged = {
        "fold_index": fold_index,
        "level": level,
        "model_seed": model_seed,
        "selection_id": selection_id,
        "temporal_config_id": temporal_config_id,
    }
    identity = ":".join(str(logged[name]) for name in sorted(logged))
    return WandbRunSpec(
        project=config.project,
        group=config.group,
        name=f"jepa-l{level}-{temporal_config_id}-fold{fold_index}-seed{model_seed}",
        run_id=hashlib.sha256(identity.encode()).hexdigest()[:16],
        resume=config.resume,
        logged_config=logged,
    )


def build_jepa_admission_wandb_run_spec(
    *,
    config: JepaWandbConfig,
    stage_id: str,
    temporal_config_id: str,
    model_seed: int,
    level: int = 3,
) -> WandbRunSpec:
    if not isinstance(config, JepaWandbConfig):
        raise TypeError("config must be JepaWandbConfig")
    if type(level) is not int or level not in (1, 2, 3):
        raise ValueError("admission level must be 1, 2, or 3")
    if stage_id != f"l{level}-stride4-final-admission-v1":
        raise ValueError("unsupported JEPA admission stage")
    if temporal_config_id != "stride4_80ms_history_160ms":
        raise ValueError("JEPA admission W&B run must use stride-4")
    if model_seed not in (7, 17, 27):
        raise ValueError("JEPA admission W&B seed is invalid")
    logged = {
        "level": level,
        "model_seed": model_seed,
        "stage_id": stage_id,
        "temporal_config_id": temporal_config_id,
    }
    identity = ":".join(str(logged[name]) for name in sorted(logged))
    return WandbRunSpec(
        project=config.project,
        group=f"l{level}-final-admission",
        name=f"jepa-l{level}-{temporal_config_id}-admission-seed{model_seed}",
        run_id=hashlib.sha256(identity.encode()).hexdigest()[:16],
        resume=config.resume,
        logged_config=logged,
    )


def credential_environment_status() -> dict[str, str]:
    return {
        name: "SET" if bool(os.environ.get(name)) else "UNSET"
        for name in ("HF_TOKEN", "WANDB_API_KEY")
    }


def initialize_wandb_run(*, spec: WandbRunSpec):
    if not isinstance(spec, WandbRunSpec):
        raise TypeError("spec must be WandbRunSpec")
    if credential_environment_status()["WANDB_API_KEY"] != "SET":
        raise RuntimeError("WANDB_API_KEY is UNSET")
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("wandb is absent from the JEPA environment") from error
    return wandb.init(
        project=spec.project,
        group=spec.group,
        name=spec.name,
        id=spec.run_id,
        resume=spec.resume,
        config=spec.logged_config,
    )
