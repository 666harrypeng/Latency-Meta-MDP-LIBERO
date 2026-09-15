"""Configuration for frozen-Encoder fresh-Decoder sufficiency probes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class FreshDecoderProbeConfig:
    schema_version: int
    probe_id: str
    decoder_seeds: tuple[int, ...]
    max_epochs: int
    early_stopping_patience: int
    rmse_ratio_max: float

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.probe_id != "dinov3_flow_belief_fresh_decoder_probe_v1":
            raise ValueError("unsupported fresh Decoder probe config")
        if (
            len(self.decoder_seeds) != 3
            or self.decoder_seeds != tuple(sorted(set(self.decoder_seeds)))
            or any(
                isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
                for seed in self.decoder_seeds
            )
        ):
            raise ValueError("fresh Decoder seeds must contain three sorted unique values")
        if (
            isinstance(self.max_epochs, bool)
            or not isinstance(self.max_epochs, int)
            or self.max_epochs <= 0
            or isinstance(self.early_stopping_patience, bool)
            or not isinstance(self.early_stopping_patience, int)
            or not 0 < self.early_stopping_patience <= self.max_epochs
        ):
            raise ValueError("fresh Decoder epoch controls are invalid")
        if not 1.0 <= self.rmse_ratio_max <= 2.0:
            raise ValueError("fresh Decoder RMSE ratio gate is invalid")


def load_fresh_decoder_probe_config(path: Path) -> FreshDecoderProbeConfig:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = set(FreshDecoderProbeConfig.__dataclass_fields__)
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("fresh Decoder probe config fields are invalid")
    value["decoder_seeds"] = tuple(value["decoder_seeds"])
    return FreshDecoderProbeConfig(**value)
