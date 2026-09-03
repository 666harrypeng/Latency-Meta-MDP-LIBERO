"""Deterministic episode-level strategy sampling for structured expert realizations."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from latency_meta_mdp.expert_realization.config import (
    StructuredExpertConfig,
    load_structured_expert_config,
)
from latency_meta_mdp.expert_realization.contracts import (
    ExpertRealizationKey,
    FormalRealizationDrawRequest,
    FormalRealizationRequest,
    FormalRequestUniverse,
    StrategyFamily,
    StrategyParameters,
    build_formal_realization_draw_request,
    build_formal_realization_requests,
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
    *,
    assigned_family: StrategyFamily,
) -> StrategyParameters:
    """Sample one complete factual strategy from its canonical realization identity."""
    if not isinstance(task_instance, MaterializedTaskInstance):
        raise TypeError("task_instance must be a MaterializedTaskInstance")
    if not isinstance(realization_key, ExpertRealizationKey):
        raise TypeError("realization_key must be an ExpertRealizationKey")
    if not isinstance(config, StructuredStrategyConfig):
        raise TypeError("config must be a StructuredStrategyConfig")
    if not isinstance(assigned_family, StrategyFamily):
        raise TypeError("assigned_family must be a StrategyFamily")
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
    return sample_strategy_parameters(
        config.expert,
        family=assigned_family,
        realization_seed=realization_key.realization_seed,
    )


def sample_requested_strategy(
    task_instance: MaterializedTaskInstance,
    request: FormalRealizationRequest | FormalRealizationDrawRequest,
    config: StructuredStrategyConfig,
    *,
    universe: FormalRequestUniverse,
) -> StrategyParameters:
    """Resolve one formal request without reconstructing its family or random seed."""
    if not isinstance(task_instance, MaterializedTaskInstance):
        raise TypeError("task_instance must be a MaterializedTaskInstance")
    if not isinstance(request, (FormalRealizationRequest, FormalRealizationDrawRequest)):
        raise TypeError("request must be a formal realization request")
    if not isinstance(config, StructuredStrategyConfig):
        raise TypeError("config must be a StructuredStrategyConfig")
    if not isinstance(universe, FormalRequestUniverse):
        raise TypeError("universe must be a FormalRequestUniverse")
    task_instance.validate_publication_consistency()
    if universe.structured_expert_config_sha256 != config.source_sha256:
        raise ValueError("formal request universe does not bind the structured strategy config")
    if request.task_instance_id != task_instance.task_instance_id:
        raise ValueError("formal realization request does not belong to the task instance")
    if isinstance(request, FormalRealizationDrawRequest):
        expected = build_formal_realization_draw_request(
            universe,
            task_instance.task_instance_id,
            request.realization_draw_index,
        )
        if request != expected:
            raise ValueError("formal realization draw is not part of its request universe")
    elif request not in build_formal_realization_requests(
        universe, task_instance.task_instance_id
    ):
        raise ValueError("formal realization request is not part of its request universe")
    key = request.to_expert_realization_key()
    return sample_strategy_parameters(
        config.expert,
        family=request.assigned_family,
        realization_seed=key.realization_seed,
    )
