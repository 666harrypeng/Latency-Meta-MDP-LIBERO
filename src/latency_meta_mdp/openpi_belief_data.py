"""Validated return-mixture inputs using the matched clean policy's state statistics."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
from openpi import transforms

from latency_meta_mdp.openpi_policy_data import StructuredPolicyDataConfig, StructuredPolicyInputs


@dataclasses.dataclass(frozen=True)
class ReturnBeliefInputs(StructuredPolicyInputs):
    state_norm_stats: Any
    use_quantiles: bool = True

    def __call__(self, data: dict) -> dict:
        belief = data.get("return_belief")
        expected = {"visual", "proprio", "delay_ticks", "probabilities"}
        if not isinstance(belief, dict) or set(belief) != expected:
            raise ValueError(
                "return Belief fields must be exactly visual/proprio/delay_ticks/probabilities"
            )
        shapes = {
            "visual": (5, 2, 196, 384),
            "proprio": (5, 16),
            "delay_ticks": (5,),
            "probabilities": (5,),
        }
        arrays = {key: np.asarray(value) for key, value in belief.items()}
        if any(
            arrays[key].shape != shape or not np.isfinite(arrays[key]).all()
            for key, shape in shapes.items()
        ):
            raise ValueError("return Belief fields must be finite and match the five-anchor shapes")
        if not np.array_equal(arrays["delay_ticks"], [4, 8, 12, 16, 20]):
            raise ValueError("return Belief must use the admitted stride4 anchor times")
        probabilities = arrays["probabilities"]
        if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0, atol=1e-6, rtol=0):
            raise ValueError("return Belief probabilities must be nonnegative and sum to one")
        if self.state_norm_stats is None or np.asarray(self.state_norm_stats.mean).shape != (16,):
            raise ValueError("return Belief requires the matched clean current16 state statistics")
        inputs = super().__call__(data)
        inputs.update(
            {
                "return_belief_visual": arrays["visual"].astype(np.float16),
                "return_belief_proprio": arrays["proprio"].astype(np.float32),
                "return_belief_delay_ticks": arrays["delay_ticks"].astype(np.float32),
                "return_belief_probabilities": probabilities.astype(np.float32),
            }
        )
        inputs = transforms.Normalize(
            {"return_belief_proprio": self.state_norm_stats},
            use_quantiles=self.use_quantiles,
            strict=True,
        )(inputs)
        inputs["return_belief_proprio"] = inputs["return_belief_proprio"].astype(np.float32)
        if not all(
            np.isfinite(inputs[key]).all()
            for key in (
                "return_belief_visual",
                "return_belief_proprio",
            )
        ):
            raise ValueError("return Belief fields overflowed during conversion or normalization")
        return inputs


@dataclasses.dataclass(frozen=True)
class ReturnBeliefDataConfig(StructuredPolicyDataConfig):
    """Transforms for a provider that attaches a return mixture to each policy sample.

    The existing clean LeRobot dataset alone does not contain these fields. A
    matched predicted/GT/oracle provider must supply them; missing Belief fails.
    This factory does not select labels, sample a realized delay, or train JEPA.
    """

    def create(self, assets_dirs, model_config):
        if not getattr(model_config, "use_return_belief", False):
            raise ValueError("return Belief data requires a conditioned policy model")
        config = super().create(assets_dirs, model_config)
        if config.norm_stats is None or "state" not in config.norm_stats:
            raise ValueError(
                "conditioned data requires the matched clean policy's state statistics"
            )
        structure = dict(config.repack_transforms.inputs[0].structure)
        structure["return_belief"] = {
            key: f"return_belief/{key}"
            for key in ("visual", "proprio", "delay_ticks", "probabilities")
        }
        return dataclasses.replace(
            config,
            repack_transforms=transforms.Group(inputs=[transforms.RepackTransform(structure)]),
            data_transforms=transforms.Group(
                inputs=[
                    ReturnBeliefInputs(
                        model_type=model_config.model_type,
                        state_norm_stats=config.norm_stats["state"],
                        use_quantiles=config.use_quantile_norm,
                    )
                ],
                outputs=config.data_transforms.outputs,
            ),
        )
