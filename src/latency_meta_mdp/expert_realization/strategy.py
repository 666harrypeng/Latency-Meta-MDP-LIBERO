"""Deterministic episode-level strategy sampling for structured expert realizations."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from latency_meta_mdp.expert_realization.config import (
    CANONICAL_FAMILIES,
    StructuredExpertConfig,
    load_structured_expert_config,
)
from latency_meta_mdp.expert_realization.contracts import (
    ExpertRealizationKey,
    StrategyParameters,
    sample_strategy_parameters,
)
from latency_meta_mdp.expert_realization.task_instance import MaterializedTaskInstance

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class StructuredStrategyConfig:
    """One reviewed strategy config together with its raw-file identity."""

    expert: StructuredExpertConfig
    source_sha256: str
    source_bytes: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.expert, StructuredExpertConfig):
            raise TypeError("expert must be a StructuredExpertConfig")
        if type(self.source_sha256) is not str or _SHA256.fullmatch(self.source_sha256) is None:
            raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
        if type(self.source_bytes) is not bytes:
            raise TypeError("source_bytes must be bytes")
        if hashlib.sha256(self.source_bytes).hexdigest() != self.source_sha256:
            raise ValueError("source_sha256 does not bind source_bytes")
        raw = yaml.safe_load(self.source_bytes)
        if raw != self.expert.to_mapping():
            raise ValueError("source_bytes do not encode the bound structured expert config")

    @classmethod
    def from_path(cls, path: Path) -> StructuredStrategyConfig:
        path = Path(path)
        payload = path.read_bytes()
        return cls(
            expert=load_structured_expert_config(path),
            source_sha256=hashlib.sha256(payload).hexdigest(),
            source_bytes=payload,
        )


def sample_strategy(
    task_instance: MaterializedTaskInstance,
    realization_key: ExpertRealizationKey,
    config: StructuredStrategyConfig,
) -> StrategyParameters:
    """Sample one complete factual strategy from its canonical realization identity."""
    if not isinstance(task_instance, MaterializedTaskInstance):
        raise TypeError("task_instance must be a MaterializedTaskInstance")
    if not isinstance(realization_key, ExpertRealizationKey):
        raise TypeError("realization_key must be an ExpertRealizationKey")
    if not isinstance(config, StructuredStrategyConfig):
        raise TypeError("config must be a StructuredStrategyConfig")
    task_instance.validate_publication_consistency()
    if realization_key.task_instance_id != task_instance.task_instance_id:
        raise ValueError("realization key does not belong to the task instance")
    expected_key = ExpertRealizationKey(
        task_instance.task_instance_id,
        realization_key.realization_index,
        realization_key.realization_namespace_sha256,
    )
    if realization_key.realization_namespace_sha256 != config.source_sha256:
        raise ValueError("realization key does not match the structured strategy config")
    if realization_key != expected_key:
        raise ValueError("realization key does not match the structured strategy config")
    family_index = realization_key.realization_index // config.expert.samples_per_family
    if not 0 <= family_index < len(CANONICAL_FAMILIES):
        raise ValueError("realization index does not map to a canonical strategy family")
    return sample_strategy_parameters(
        config.expert,
        family=CANONICAL_FAMILIES[family_index],
        realization_seed=realization_key.realization_seed,
    )
