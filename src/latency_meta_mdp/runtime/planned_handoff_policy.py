"""Native two-image policy with a causal, explicitly indexed handoff at h+q.

The returned H50 plan remains observation-indexed for the existing RTC client.
No realized arrival time is available here. Guidance is evaluated during generation,
on the same absolute action times as the policy's input observation.
"""

from dataclasses import replace

import numpy as np

from latency_meta_mdp.runtime.rtc_protocol import RtcInferenceContext

PLAN_CONSTRUCTION = "rtc_planned_handoff_h50_v1"


class PlannedHandoffPolicy:
    action_alignment = "observation_time"
    plan_construction = PLAN_CONSTRUCTION

    def __init__(self, native_policy, *, mode):
        if mode not in {"current", "forecast"} or not callable(native_policy):
            raise ValueError("planned handoff requires a native policy and current/forecast mode")
        if getattr(native_policy, "uses_forecast", True):
            raise ValueError("planned handoff requires a native two-camera policy")
        self.native_policy, self.mode = native_policy, mode

    def __call__(self, observation, context=None):
        if context is None:
            return self.native_policy(observation)
        if not isinstance(context, RtcInferenceContext) or (
            observation.formal_tick != context.origin_tick
            or any(
                not np.array_equal(getattr(observation, key), getattr(context.observation, key))
                for key in ("image", "wrist_image", "state", "prompt")
            )
        ):
            raise ValueError("handoff observation must match its source context")
        q = context.estimated_delay_ticks
        old, valid = context.previous_actions, context.previous_action_mask
        if (
            type(q) is not int
            or not 0 <= q <= 20
            or old.shape != (50, 7)
            or not np.isfinite(old).all()
            or valid.shape != (50,)
            or valid.dtype != np.bool_
            or not np.array_equal(valid, np.arange(50) < valid.sum())
            or not valid[:q].all()
        ):
            raise ValueError("handoff needs a real, contiguous old prefix through q")
        prediction = context.forecast
        if self.mode == "forecast" and (
            prediction is None or prediction.target_tick != context.origin_tick + q
        ):
            raise ValueError("forecast handoff needs an explicitly available/missing q prediction")
        use_future = self.mode == "forecast" and prediction.available
        native_context = replace(context, forecast=None)
        if use_future:
            observation = replace(
                observation,
                formal_tick=prediction.target_tick,
                image=prediction.rgb[0],
                wrist_image=prediction.rgb[1],
                state=prediction.proprio,
            )
            shifted, shifted_mask = np.zeros_like(old), np.zeros_like(valid)
            shifted[: 50 - q], shifted_mask[: 50 - q] = old[q:], valid[q:]
            native_context = replace(
                native_context,
                observation=observation,
                origin_tick=prediction.target_tick,
                previous_actions=shifted,
                previous_action_mask=shifted_mask,
                estimated_delay_ticks=0,
            )
        # At the last covered tick there may be no old actions after h+q.
        # No overlap means ordinary sampling, not fabricated hold guidance or
        # an all-masked training target passed through the native transforms.
        output = (
            self.native_policy(observation, native_context)
            if native_context.previous_action_mask.any()
            else self.native_policy(observation)
        )
        generated = np.asarray(output["actions"] if isinstance(output, dict) else output)
        if generated.shape != (50, 7) or not np.isfinite(generated).all():
            raise ValueError("native handoff policy must return finite H50x7 actions")
        # Current control already starts at h; predicted observations start at h+q.
        suffix = generated[: 50 - q] if use_future else generated[q:]
        return {
            "actions": np.concatenate((old[:q], suffix), axis=0),
            "handoff": {
                "plan_construction": PLAN_CONSTRUCTION,
                "mode": self.mode,
                "planned_handoff_tick": context.origin_tick + q,
                "policy_input_tick": observation.formal_tick,
                "prefix_ticks": q,
                "forecast_used": bool(use_future),
                "guidance_overlap_ticks": int(native_context.previous_action_mask.sum()),
            },
        }
