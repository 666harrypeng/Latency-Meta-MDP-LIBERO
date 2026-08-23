"""Validated π0.5 SFT profile independent of the OpenPI runtime."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import yaml

from latency_meta_mdp.temporal_contract import TemporalContract, load_temporal_contract

_GIT_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_CONFIG_NAME = re.compile(r"^[a-z0-9][a-z0-9_]*$")


@dataclass(frozen=True)
class SFTLevelProfile:
    config_name: str
    repo_id: str

    def __post_init__(self) -> None:
        if _SAFE_CONFIG_NAME.fullmatch(self.config_name) is None:
            raise ValueError("SFT config name is invalid")
        if self.repo_id.count("/") != 1 or any(
            not component for component in self.repo_id.split("/")
        ):
            raise ValueError("SFT dataset repo id must contain one namespace separator")


@dataclass(frozen=True)
class SFTProfile:
    schema_version: int
    profile_id: str
    openpi_revision: str
    openpi_patch_sha256: str
    base_checkpoint: str
    full_parameter: bool
    temporal_contract: TemporalContract
    fps: int
    state_dim: int
    source_action_dim: int
    extra_delta_transform: bool
    drop_n_last_frames: int
    batch_size: int
    num_workers: int
    num_train_steps: int
    warmup_steps: int
    peak_learning_rate: float
    decay_learning_rate: float
    save_interval: int
    keep_period: int
    log_interval: int
    fsdp_devices: int
    ema_decay: float
    levels: Mapping[int, SFTLevelProfile]

    def __post_init__(self) -> None:
        levels = dict(self.levels)
        object.__setattr__(self, "levels", MappingProxyType(levels))
        if self.schema_version != 2 or self.profile_id != "pi05_panda_ball_full_sft_h50_v2":
            raise ValueError("unsupported SFT profile schema or identifier")
        if not self.full_parameter:
            raise ValueError("the canonical Panda-ball profile requires full-parameter SFT")
        if not _GIT_SHA1.fullmatch(self.openpi_revision) or not _SHA256.fullmatch(
            self.openpi_patch_sha256
        ):
            raise ValueError("OpenPI revision and patch hash have invalid digest formats")
        if not self.base_checkpoint:
            raise ValueError("base_checkpoint must be non-empty")
        if not isinstance(self.temporal_contract, TemporalContract):
            raise TypeError("temporal_contract must be a TemporalContract")
        if self.drop_n_last_frames != self.action_horizon - 1:
            raise ValueError("drop_n_last_frames must equal action_horizon - 1")
        if (self.fps, self.state_dim, self.source_action_dim) != (50, 8, 7):
            raise ValueError("SFT data must satisfy the 50 Hz 8D/7D policy contract")
        if self.extra_delta_transform:
            raise ValueError("native OSC delta actions must not receive another delta transform")
        integer_fields = (
            self.batch_size,
            self.num_workers,
            self.num_train_steps,
            self.warmup_steps,
            self.save_interval,
            self.keep_period,
            self.log_interval,
            self.fsdp_devices,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in integer_fields
        ):
            raise ValueError("SFT integer settings must be positive")
        if self.warmup_steps >= self.num_train_steps:
            raise ValueError("warmup_steps must be smaller than num_train_steps")
        if self.num_train_steps % self.save_interval or self.keep_period % self.save_interval:
            raise ValueError("checkpoint intervals must align with the training schedule")
        if not 0 < self.decay_learning_rate <= self.peak_learning_rate:
            raise ValueError("SFT learning rates are invalid")
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError("ema_decay must be between zero and one")
        if set(levels) != {1, 2, 3}:
            raise ValueError("SFT profile must define L1, L2, and L3")
        if len({profile.config_name for profile in levels.values()}) != len(levels):
            raise ValueError("level config names must be unique")
        if len({profile.repo_id for profile in levels.values()}) != len(levels):
            raise ValueError("level dataset repo ids must be unique")

    @property
    def action_horizon(self) -> int:
        return self.temporal_contract.prediction_horizon

    @property
    def launch_trigger_horizon(self) -> int:
        return self.temporal_contract.launch_trigger_horizon


def load_sft_profile(path: Path) -> SFTProfile:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("levels"), dict):
        raise ValueError("SFT profile must be a mapping with level definitions")
    level_rows = raw.pop("levels")
    temporal_path = raw.get("temporal_contract")
    if not isinstance(temporal_path, str) or not temporal_path:
        raise ValueError("SFT temporal_contract must be a non-empty relative path")
    raw["temporal_contract"] = load_temporal_contract(
        (path.parent / temporal_path).resolve()
    )
    if any(not isinstance(row, dict) for row in level_rows.values()):
        raise ValueError("SFT level definitions must be mappings")
    levels = {
        level: SFTLevelProfile(**row)
        for level, row in level_rows.items()
    }
    return SFTProfile(**raw, levels=levels)
