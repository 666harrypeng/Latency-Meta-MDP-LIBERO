"""Deterministic smooth latency-law variation assigned once per episode."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.special import betainc

_SUPPORTED_FAMILIES = {
    "truncated_beta_family_5_26_400ms_v1": (
        "truncated_beta_5_26_400ms_v1",
        5.0,
        26.0,
        0.04,
    ),
    "truncated_beta_family_8_65_400ms_v1": (
        "truncated_beta_8_65_400ms_v1",
        8.0,
        65.0,
        0.02,
    ),
}


@dataclass(frozen=True)
class EpisodeLatencyLaw:
    delay_ticks: tuple[int, ...]
    probabilities: np.ndarray
    shape_alpha: float
    shape_beta: float
    uniform_floor: float
    generation_seed: int
    continuous_mean_seconds: float
    continuous_mode_seconds: float
    effective_mean_seconds: float
    probability_sha256: str

    def __post_init__(self) -> None:
        probability = np.array(self.probabilities, dtype=np.float64, copy=True)
        if (
            self.delay_ticks != tuple(range(1, 21))
            or probability.shape != (20,)
            or not np.all(np.isfinite(probability))
            or np.any(probability <= 0.0)
            or not np.isclose(probability.sum(), 1.0, atol=1e-12, rtol=0)
            or self.shape_alpha <= 1.0
            or self.shape_beta <= 1.0
            or not 0.0 <= self.uniform_floor < 1.0
            or isinstance(self.generation_seed, bool)
            or not isinstance(self.generation_seed, int)
            or self.generation_seed < 0
            or len(self.probability_sha256) != 64
        ):
            raise ValueError("episode latency law is invalid")
        probability.setflags(write=False)
        object.__setattr__(self, "probabilities", probability)

    @property
    def region_probabilities(self) -> tuple[float, float, float, float]:
        return tuple(
            float(self.probabilities[start : start + 5].sum()) for start in range(0, 20, 5)
        )


@dataclass(frozen=True)
class EpisodeLatencyLawFamily:
    schema_version: int
    family_id: str
    base_law_id: str
    base_alpha: float
    base_beta: float
    latency_deadline_seconds: float
    control_tick_seconds: float
    mean_logit_std: float
    log_concentration_std: float
    uniform_floor_max: float
    family_seed: int
    assignment_granularity: str
    shared_master_seed_across_levels: bool

    def __post_init__(self) -> None:
        expected = _SUPPORTED_FAMILIES.get(self.family_id)
        if (
            self.schema_version != 1
            or expected is None
            or (
                self.base_law_id,
                self.base_alpha,
                self.base_beta,
                self.uniform_floor_max,
            )
            != expected
            or self.assignment_granularity != "episode"
            or self.shared_master_seed_across_levels is not True
        ):
            raise ValueError("unsupported episode latency-law family")
        numeric = (
            self.base_alpha,
            self.base_beta,
            self.latency_deadline_seconds,
            self.control_tick_seconds,
            self.mean_logit_std,
            self.log_concentration_std,
            self.uniform_floor_max,
        )
        if (
            any(not np.isfinite(value) for value in numeric)
            or self.base_alpha <= 1.0
            or self.base_beta <= 1.0
            or self.latency_deadline_seconds != 0.4
            or self.control_tick_seconds != 0.02
            or self.mean_logit_std <= 0.0
            or self.log_concentration_std <= 0.0
            or not 0.0 <= self.uniform_floor_max <= 0.1
            or isinstance(self.family_seed, bool)
            or not isinstance(self.family_seed, int)
            or self.family_seed < 0
        ):
            raise ValueError("episode latency-law family parameters are invalid")

    def _seed_for_scene(self, scene_seed: int) -> int:
        digest = hashlib.sha256(
            f"{self.family_id}:{self.family_seed}:{scene_seed}".encode()
        ).digest()
        return int.from_bytes(digest[:8], byteorder="little", signed=False)

    def _build_law(
        self,
        *,
        alpha: float,
        beta: float,
        uniform_floor: float,
        generation_seed: int,
    ) -> EpisodeLatencyLaw:
        edges = np.arange(21, dtype=np.float64) * self.control_tick_seconds
        cdf = betainc(alpha, beta, edges)
        normalizer = float(cdf[-1])
        if not 0.0 < normalizer <= 1.0:
            raise RuntimeError("episode latency-law truncation is invalid")
        beta_mass = np.diff(cdf) / normalizer
        probability = (1.0 - uniform_floor) * beta_mass + uniform_floor / 20.0
        probability /= probability.sum()
        continuous_mean = (
            alpha
            / (alpha + beta)
            * betainc(alpha + 1.0, beta, self.latency_deadline_seconds)
            / betainc(alpha, beta, self.latency_deadline_seconds)
        )
        continuous_mode = min(
            (alpha - 1.0) / (alpha + beta - 2.0),
            self.latency_deadline_seconds,
        )
        effective_mean = float(
            np.dot(
                np.arange(1, 21, dtype=np.float64) * self.control_tick_seconds,
                probability,
            )
        )
        probability_sha = hashlib.sha256(
            np.asarray(probability, dtype="<f8").tobytes(order="C")
        ).hexdigest()
        return EpisodeLatencyLaw(
            delay_ticks=tuple(range(1, 21)),
            probabilities=probability,
            shape_alpha=alpha,
            shape_beta=beta,
            uniform_floor=uniform_floor,
            generation_seed=generation_seed,
            continuous_mean_seconds=float(continuous_mean),
            continuous_mode_seconds=float(continuous_mode),
            effective_mean_seconds=effective_mean,
            probability_sha256=probability_sha,
        )

    def build_shifted_law(
        self,
        *,
        name: str,
        mean_logit_offset: float,
        log_concentration_offset: float,
        uniform_floor: float,
    ) -> EpisodeLatencyLaw:
        if not name or not 0.0 <= uniform_floor <= 0.1:
            raise ValueError("shifted latency-law request is invalid")
        base_mean = self.base_alpha / (self.base_alpha + self.base_beta)
        base_logit = math.log(base_mean / (1.0 - base_mean))
        mean = 1.0 / (1.0 + math.exp(-(base_logit + mean_logit_offset)))
        concentration = (self.base_alpha + self.base_beta) * math.exp(log_concentration_offset)
        alpha = mean * concentration
        beta = (1.0 - mean) * concentration
        if alpha <= 1.0 or beta <= 1.0:
            raise ValueError("shifted latency law has invalid Beta parameters")
        generation_seed = int.from_bytes(
            hashlib.sha256(f"{self.family_id}:shifted:{name}".encode()).digest()[:8],
            byteorder="little",
            signed=False,
        )
        return self._build_law(
            alpha=alpha,
            beta=beta,
            uniform_floor=uniform_floor,
            generation_seed=generation_seed,
        )

    def sample_for_episode(
        self,
        *,
        level: int,
        episode_id: str,
        scene_seed: int,
    ) -> EpisodeLatencyLaw:
        if (
            level not in (1, 2, 3)
            or isinstance(scene_seed, bool)
            or not isinstance(scene_seed, int)
            or scene_seed < 0
            or episode_id != f"l{level}-seed-{scene_seed:06d}-attempt-000"
        ):
            raise ValueError("episode latency-law identity is invalid")
        return self.sample_for_key(assignment_key=scene_seed)

    def sample_for_key(self, *, assignment_key: int) -> EpisodeLatencyLaw:
        """Assign the same law to a stable source key without inventing an episode ID."""
        if type(assignment_key) is not int or assignment_key < 0:
            raise ValueError("latency-law assignment key must be a non-negative integer")
        generation_seed = self._seed_for_scene(assignment_key)
        rng = np.random.default_rng(generation_seed)
        base_mean = self.base_alpha / (self.base_alpha + self.base_beta)
        base_logit = math.log(base_mean / (1.0 - base_mean))
        base_concentration = self.base_alpha + self.base_beta
        for _ in range(100):
            mean = 1.0 / (
                1.0 + math.exp(-(base_logit + self.mean_logit_std * float(rng.standard_normal())))
            )
            concentration = base_concentration * math.exp(
                self.log_concentration_std * float(rng.standard_normal())
            )
            alpha = mean * concentration
            beta = (1.0 - mean) * concentration
            if alpha > 1.0 and beta > 1.0:
                break
        else:
            raise RuntimeError("failed to sample a valid episode latency law")
        uniform_floor = float(rng.uniform(0.0, self.uniform_floor_max))
        return self._build_law(
            alpha=alpha,
            beta=beta,
            uniform_floor=uniform_floor,
            generation_seed=generation_seed,
        )


def load_episode_latency_law_family(path: Path) -> EpisodeLatencyLawFamily:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = set(EpisodeLatencyLawFamily.__dataclass_fields__)
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("episode latency-law family config fields are invalid")
    return EpisodeLatencyLawFamily(**value)
