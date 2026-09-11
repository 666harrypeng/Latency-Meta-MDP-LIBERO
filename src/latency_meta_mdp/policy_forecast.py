"""Observation-indexed sources preserving action targets when forecast inputs are absent.

The recording provides a nominal executable prefix only up to its final real action.
Missing future controls are not invented; those sources retain native action supervision
with an explicitly unavailable forecast. Runtime may supply a real longer queued plan.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from latency_meta_mdp.policy_data import PolicyEpisode, _readonly, materialize_policy_action_target


@dataclass(frozen=True)
class ForecastPolicySource:
    episode_id: str
    source_tick: int
    query_ticks: int
    history_ticks: tuple[int, int, int] | None
    executed_controls: np.ndarray | None
    executable_controls: np.ndarray | None
    recorded_control_mask: np.ndarray
    gt_endpoint_recorded: bool  # Dataset/evaluator metadata, never a policy input.
    policy_inputs: dict

    @property
    def forecast_ready(self) -> bool:
        return self.history_ticks is not None and self.executable_controls is not None

    def with_forecast(self, *, rgb, proprio, source_tick: int, target_tick: int) -> dict:
        """Attach only a prediction made for this source and requested time."""
        if not self.forecast_ready:
            raise ValueError("source lacks real history or executable forecast controls")
        if source_tick != self.source_tick or target_tick != source_tick + self.query_ticks:
            raise ValueError("forecast source/target identity does not match the policy source")
        return {
            **self.policy_inputs,
            "forecast": {"rgb": rgb, "proprio": proprio, "query_ticks": self.query_ticks},
        }


def materialize_forecast_policy_source(
    episode: PolicyEpisode, *, source_tick: int, query_ticks: int
) -> ForecastPolicySource:
    """Keep every real action start; q changes only the forecast, not its H50 target."""
    if type(query_ticks) is not int or not 0 <= query_ticks <= 20:
        raise ValueError("forecast query must be an integer in0..20")
    if (
        episode.state_contract != "joint_qpos_qvel_gripper_width_velocity"
        or episode.state.shape != (len(episode.actions), 16)
        or episode.action_horizon != 50
    ):
        raise ValueError("forecast policy requires current16D proprio and H50 source")
    target, target_mask = materialize_policy_action_target(episode, target_start_tick=source_tick)
    count = min(20, len(episode.actions) - source_tick)
    controls = np.zeros((20, 7), np.float32)
    controls[:count] = episode.actions[source_tick : source_tick + count]
    if not np.isfinite(controls).all() or np.any(np.abs(controls) > 1):
        raise ValueError("forecast buffer must contain finite controller-native actions")
    ready = source_tick >= 10
    return ForecastPolicySource(
        episode_id=episode.episode_id,
        source_tick=source_tick,
        query_ticks=query_ticks,
        history_ticks=(source_tick - 8, source_tick - 4, source_tick) if ready else None,
        executed_controls=_readonly(episode.actions[source_tick - 8 : source_tick].reshape(2, 4, 7))
        if ready
        else None,
        executable_controls=_readonly(controls) if count >= query_ticks else None,
        recorded_control_mask=_readonly(np.arange(20) < count),
        gt_endpoint_recorded=source_tick + query_ticks <= len(episode.actions),
        policy_inputs={
            "observation/image": episode.agentview_rgb[source_tick],
            "observation/wrist_image": episode.wrist_rgb[source_tick],
            "observation/state": episode.state[source_tick],
            "prompt": episode.instruction,
            "actions": target,
            "actions_is_pad": _readonly(~target_mask),
            "forecast": {"rgb": None, "proprio": None, "query_ticks": query_ticks},
        },
    )
