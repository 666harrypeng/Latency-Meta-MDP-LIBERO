"""Compatibility import used by the pinned OpenPI patch; implementation lives in its package."""

from latency_meta_mdp.policy.openpi.rtc import build_rtc_weights, rtc_guided_velocity

__all__ = ["rtc_guided_velocity", "build_rtc_weights"]
