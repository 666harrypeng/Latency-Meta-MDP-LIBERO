"""Forecast inputs with raw H50 output for return-time index-zero replacement."""

from dataclasses import replace

from latency_meta_mdp.runtime.rtc_protocol import RtcInferenceContext


class SharpForecastPolicy:
    """No RTC guidance, old-prefix stitching, or elapsed-action skipping."""

    def __init__(self, native_policy, *, forecast_only):
        self.native_policy = native_policy
        self.forecast_only = forecast_only

    def __call__(self, observation, context):
        if (
            not isinstance(context, RtcInferenceContext)
            or context.origin_tick != observation.formal_tick
            or context.forecast is None
        ):
            raise ValueError("sharp forecast requires a matching source-time forecast context")
        forecast = context.forecast
        if self.forecast_only:
            if forecast.available:
                observation = replace(
                    observation,
                    formal_tick=forecast.target_tick,
                    image=forecast.rgb[0],
                    wrist_image=forecast.rgb[1],
                    state=forecast.proprio,
                )
            return self.native_policy(observation, None)
        return self.native_policy(observation, forecast.to_condition())
