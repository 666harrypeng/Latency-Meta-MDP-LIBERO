"""Known nominal latency law on the locked 20 ms client grid."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from scipy.special import betainc

_SUPPORTED_LAWS = {
    "truncated_beta_5_26_400ms_v1": (5.0, 26.0),
    "truncated_beta_8_65_400ms_v1": (8.0, 65.0),
}


@dataclass(frozen=True)
class TruncatedBetaLatencyLaw:
    schema_version: int
    law_id: str
    distribution_family: str
    shape_alpha: float
    shape_beta: float
    latency_deadline_seconds: float
    control_tick_seconds: float
    sampling_mode: str
    jitter_enabled: bool
    zero_delay_included: bool
    law_visibility: str
    parameter_status: str

    def __post_init__(self) -> None:
        expected_shape = _SUPPORTED_LAWS.get(self.law_id)
        if self.schema_version != 1 or expected_shape is None:
            raise ValueError("unsupported latency-law schema or identifier")
        if self.distribution_family != "truncated_beta_seconds":
            raise ValueError("latency law must use the truncated Beta family")
        if (
            not np.isfinite(self.shape_alpha)
            or not np.isfinite(self.shape_beta)
            or self.shape_alpha <= 1.0
            or self.shape_beta <= 1.0
            or (self.shape_alpha, self.shape_beta) != expected_shape
        ):
            raise ValueError(
                "latency Beta shape parameters are invalid or inconsistent with law identifier"
            )
        if (
            not np.isfinite(self.latency_deadline_seconds)
            or not np.isfinite(self.control_tick_seconds)
            or not 0.0 < self.control_tick_seconds < self.latency_deadline_seconds <= 1.0
        ):
            raise ValueError("latency deadline and control tick are invalid")
        ratio = self.latency_deadline_seconds / self.control_tick_seconds
        if not np.isclose(ratio, round(ratio), atol=1e-12, rtol=0):
            raise ValueError("latency deadline must contain an integer number of control ticks")
        if self.sampling_mode != "categorical_bin_mass":
            raise ValueError("latency sampling must use categorical bin masses")
        if self.jitter_enabled or self.zero_delay_included:
            raise ValueError("nominal stochastic latency excludes jitter and zero delay")
        if self.law_visibility != "known_nominal":
            raise ValueError("primary belief setting requires a known nominal law")
        if self.parameter_status != "locked_nominal":
            raise ValueError("latency-law parameters must be locked")

    @property
    def bin_count(self) -> int:
        return round(self.latency_deadline_seconds / self.control_tick_seconds)

    @property
    def delay_ticks(self) -> tuple[int, ...]:
        return tuple(range(1, self.bin_count + 1))

    @property
    def probabilities(self) -> np.ndarray:
        edges = np.arange(self.bin_count + 1, dtype=np.float64) * self.control_tick_seconds
        cdf = betainc(self.shape_alpha, self.shape_beta, edges)
        normalizer = float(cdf[-1])
        if not 0.0 < normalizer <= 1.0:
            raise ValueError("latency-law truncation has invalid probability mass")
        probabilities = np.diff(cdf) / normalizer
        probabilities /= probabilities.sum()
        probabilities.setflags(write=False)
        return probabilities

    @property
    def condition_vector(self) -> np.ndarray:
        return self.probabilities

    @property
    def continuous_mode_seconds(self) -> float:
        mode = (self.shape_alpha - 1.0) / (
            self.shape_alpha + self.shape_beta - 2.0
        )
        return min(mode, self.latency_deadline_seconds)

    @property
    def continuous_mean_seconds(self) -> float:
        denominator = betainc(
            self.shape_alpha,
            self.shape_beta,
            self.latency_deadline_seconds,
        )
        numerator = (
            self.shape_alpha
            / (self.shape_alpha + self.shape_beta)
            * betainc(
                self.shape_alpha + 1.0,
                self.shape_beta,
                self.latency_deadline_seconds,
            )
        )
        return float(numerator / denominator)

    @property
    def effective_mean_seconds(self) -> float:
        ticks = np.asarray(self.delay_ticks, dtype=np.float64)
        return float(np.dot(ticks * self.control_tick_seconds, self.probabilities))

    @property
    def region_probabilities(self) -> tuple[float, float, float, float]:
        bins_per_region = round(0.1 / self.control_tick_seconds)
        if bins_per_region * 4 != self.bin_count:
            raise ValueError("latency-law regions require four 100 ms intervals")
        return tuple(
            float(self.probabilities[start : start + bins_per_region].sum())
            for start in range(0, self.bin_count, bins_per_region)
        )


@dataclass
class CategoricalDelaySampler:
    law: TruncatedBetaLatencyLaw
    seed: int
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.law, TruncatedBetaLatencyLaw):
            raise TypeError("law must be a TruncatedBetaLatencyLaw")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("latency sampler seed must be a non-negative integer")
        self._rng = np.random.default_rng(self.seed)

    def __call__(self) -> int:
        return int(self._rng.choice(self.law.delay_ticks, p=self.law.probabilities))


def load_latency_law(path: Path) -> TruncatedBetaLatencyLaw:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = set(TruncatedBetaLatencyLaw.__dataclass_fields__)
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("latency-law config fields are invalid")
    return TruncatedBetaLatencyLaw(**raw)
