"""Decoded future observations through native SigLIP and pi0.5 state/text tokens."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
from openpi import transforms

from latency_meta_mdp.openpi_policy_data import StructuredPolicyDataConfig, StructuredPolicyInputs


@dataclasses.dataclass(frozen=True)
class ForecastPolicyInputs(StructuredPolicyInputs):
    state_norm_stats: Any
    use_quantiles: bool = True

    def __call__(self, data: dict) -> dict:
        forecast = data.get("forecast")
        if not isinstance(forecast, dict) or set(forecast) != {"rgb", "proprio", "query_ticks"}:
            raise ValueError("forecast requires exactly rgb/proprio/query_ticks")
        q = forecast["query_ticks"]
        if isinstance(q, np.ndarray) and q.shape == () and np.issubdtype(q.dtype, np.integer):
            q = int(q)
        if type(q) is not int or not 0 <= q <= 20:
            raise ValueError("forecast query must be an integer in0..20")
        rgb, state = forecast["rgb"], forecast["proprio"]
        available = rgb is not None
        if available != (state is not None):
            raise ValueError("forecast RGB and proprio must share availability")
        inputs = super().__call__(data)
        inputs["image"].pop("right_wrist_0_rgb")
        inputs["image_mask"].pop("right_wrist_0_rgb")
        if available:
            rgb, state = np.asarray(rgb), np.asarray(state)
            if rgb.shape != (2, 224, 224, 3) or rgb.dtype != np.uint8:
                raise ValueError("decoded forecast RGB must be uint8[2,224,224,3]")
            if state.shape != (16,) or not np.isfinite(state).all():
                raise ValueError("forecast requires finite physical16D proprio")
            if self.state_norm_stats is None or np.shape(self.state_norm_stats.mean) != (16,):
                raise ValueError("forecast requires clean policy16D normalization")
            inputs["forecast_proprio"] = state.astype(np.float32)
            inputs = transforms.Normalize(
                {"forecast_proprio": self.state_norm_stats},
                use_quantiles=self.use_quantiles,
                strict=True,
            )(inputs)
            if not np.isfinite(inputs["forecast_proprio"]).all():
                raise ValueError("normalized forecast proprio must be finite")
        else:
            rgb = np.zeros((2, 224, 224, 3), dtype=np.uint8)
            inputs["forecast_proprio"] = None
        for name, image in zip(
            ("forecast_base_0_rgb", "forecast_left_wrist_0_rgb"), rgb, strict=True
        ):
            inputs["image"][name] = image
            inputs["image_mask"][name] = np.bool_(available)
        inputs["forecast_query_ticks"] = q
        return inputs


@dataclasses.dataclass(frozen=True)
class ForecastTokenizePrompt(transforms.TokenizePrompt):
    def __call__(self, data: dict) -> dict:
        if not self.discrete_state_input:
            raise ValueError("RTC forecast requires native state token inputs")
        data = dict(data)
        tokens, mask = self.tokenizer.tokenize_forecast(
            data.pop("prompt"),
            data["state"],
            future_state=data.pop("forecast_proprio"),
            query_ticks=data.pop("forecast_query_ticks"),
        )
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": mask}


@dataclasses.dataclass(frozen=True)
class ForecastPolicyDataConfig(StructuredPolicyDataConfig):
    """A source provider must attach forecast fields; the clean dataset alone fails closed."""

    forecast_policy_view: dict | None = None

    def create(self, assets_dirs, model_config):
        if not getattr(model_config, "use_rtc_forecast", False):
            raise ValueError("forecast data requires the native four-image policy config")
        config = super().create(assets_dirs, model_config)
        if config.norm_stats is None or "state" not in config.norm_stats:
            raise ValueError("forecast data requires matched clean policy state statistics")
        structure = dict(config.repack_transforms.inputs[0].structure)
        structure["forecast"] = {
            key: f"forecast/{key}" for key in ("rgb", "proprio", "query_ticks")
        }
        view_kwargs = {}
        if self.forecast_policy_view is not None:
            if not hasattr(config, "forecast_policy_view"):
                raise ValueError("forecast data loading requires OpenPI patch0009")
            view_kwargs["forecast_policy_view"] = self.forecast_policy_view
        return dataclasses.replace(
            config,
            **view_kwargs,
            repack_transforms=transforms.Group(inputs=[transforms.RepackTransform(structure)]),
            data_transforms=transforms.Group(
                inputs=[
                    ForecastPolicyInputs(
                        model_type=model_config.model_type,
                        state_norm_stats=config.norm_stats["state"],
                        use_quantiles=config.use_quantile_norm,
                    )
                ],
                outputs=config.data_transforms.outputs,
            ),
        )
