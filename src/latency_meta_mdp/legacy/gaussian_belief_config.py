"""Compatibility exports for the isolated Gaussian baseline configuration."""

from latency_meta_mdp.legacy.belief.gaussian.config import (
    GaussianBeliefConfig,
    load_gaussian_belief_config,
)

__all__ = ["GaussianBeliefConfig", "load_gaussian_belief_config"]
