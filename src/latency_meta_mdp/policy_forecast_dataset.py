"""Compatibility import used by the pinned OpenPI patch; implementation lives in its package."""

from latency_meta_mdp.data.forecast.dataset import load_forecast_policy_dataset

__all__ = ["load_forecast_policy_dataset"]
