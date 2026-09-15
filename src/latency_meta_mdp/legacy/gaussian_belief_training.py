"""Compatibility exports for isolated Gaussian baseline training."""

from latency_meta_mdp.legacy.belief.gaussian.training import (
    GaussianBeliefBatch,
    collate_gaussian_belief_items,
    train_level_gaussian_belief,
)

__all__ = [
    "GaussianBeliefBatch",
    "collate_gaussian_belief_items",
    "train_level_gaussian_belief",
]
