"""Forecast replacement through the unchanged native pi0.5 input and model stack."""

import dataclasses
import json
from math import gcd
from pathlib import Path

from latency_meta_mdp.data.forecast.only_dataset import eligible_counts
from latency_meta_mdp.policy.conditioning import conditioning_contract
from latency_meta_mdp.policy.openpi.data import StructuredPolicyDataConfig


@dataclasses.dataclass(frozen=True)
class ForecastOnlyDataConfig(StructuredPolicyDataConfig):
    forecast_policy_view: dict | None = None

    def create(self, assets_dirs, model_config):
        if getattr(model_config, "use_rtc_forecast", False):
            raise ValueError("forecast-only requires the native clean model input structure")
        config = super().create(assets_dirs, model_config)
        return dataclasses.replace(config, forecast_policy_view=self.forecast_policy_view)


def build_forecast_only_train_config(
    *, clean_config, clean_checkpoint, forecast_identity, experiment_name, forecast_view=None
):
    from latency_meta_mdp.policy.openpi.training import build_forecast_policy_train_config

    # Reuse identity validation, optimizer and trainable scope; restore the native
    # model and transforms before any model or data-loader construction occurs.
    common = build_forecast_policy_train_config(
        clean_config=clean_config,
        clean_checkpoint=clean_checkpoint,
        forecast_identity=forecast_identity,
        experiment_name=experiment_name,
    )
    schedule, metadata = {}, {}
    if forecast_view is not None:
        if forecast_view.get("input_mode") != "forecast_only" or any(
            forecast_view["bindings"].get(k) != v for k, v in forecast_identity.items()
        ):
            raise ValueError("forecast-only view identity differs")
        manifest = json.loads((Path(forecast_view["cache_root"]) / "manifest.json").read_text())
        if (
            manifest.get("complete") is not True
            or manifest["bindings"] != forecast_view["bindings"]
        ):
            raise ValueError("forecast-only requires the complete matching cache")
        counts = eligible_counts(manifest["episodes"])
        if min(counts) < 1:
            raise ValueError("forecast-only requires real targets at every query")
        batch = clean_config.batch_size
        multiple = batch // gcd(batch, 20)
        per_query = ((max(counts) + multiple - 1) // multiple) * multiple
        epoch = 20 * per_query // batch
        schedule = dict(
            num_train_steps=2 * epoch,
            keep_period=epoch,
            lr_schedule=dataclasses.replace(
                clean_config.lr_schedule, warmup_steps=max(1, epoch // 10), decay_steps=2 * epoch
            ),
        )
        metadata = dict(
            forecast_training_budget_resolved=True,
            balanced_pair_epochs=2,
            eligible_pairs_by_query=counts,
            balanced_examples_per_epoch=20 * per_query,
            training_examples=40 * per_query,
        )
    return dataclasses.replace(
        common,
        **schedule,
        name=clean_config.name + "_forecast_only",
        model=clean_config.model,
        data=ForecastOnlyDataConfig(
            forecast_policy_view=forecast_view,
            **{
                field.name: getattr(common.data, field.name)
                for field in dataclasses.fields(StructuredPolicyDataConfig)
            },
        ),
        policy_metadata={
            **common.policy_metadata,
            **metadata,
            **conditioning_contract("forecast_only"),
            "input_mode": "forecast_only",
            "source_tail_policy": "forecast_endpoint_real_action_mask_v1",
        },
    )
