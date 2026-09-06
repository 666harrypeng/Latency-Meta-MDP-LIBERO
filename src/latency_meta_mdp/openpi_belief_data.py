"""Validated return-mixture inputs using the matched clean policy's state statistics."""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar

import numpy as np
from openpi import transforms

from latency_meta_mdp.openpi_policy_data import StructuredPolicyDataConfig, StructuredPolicyInputs


@dataclasses.dataclass(frozen=True)
class ReturnBeliefInputs(StructuredPolicyInputs):
    allow_empty_action_targets: ClassVar[bool] = True
    require_native_anchors: ClassVar[bool] = True
    state_norm_stats: Any
    use_quantiles: bool = True

    def _read_belief(self, data: dict) -> dict:
        belief = data.get("return_belief")
        expected = {"visual", "proprio", "delay_ticks", "probabilities"}
        if not isinstance(belief, dict) or set(belief) != expected:
            raise ValueError(
                "return Belief fields must be exactly visual/proprio/delay_ticks/probabilities"
            )
        return belief

    def __call__(self, data: dict) -> dict:
        belief = self._read_belief(data)
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
        if self.require_native_anchors and not np.array_equal(
            arrays["delay_ticks"], [4, 8, 12, 16, 20]
        ):
            raise ValueError("return Belief must use the admitted stride4 anchor times")
        probabilities = arrays["probabilities"]
        if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0, atol=1e-6, rtol=0):
            raise ValueError("return Belief probabilities must be nonnegative and sum to one")
        if self.state_norm_stats is None or np.asarray(self.state_norm_stats.mean).shape != (16,):
            raise ValueError("return Belief requires the matched clean current16 state statistics")
        inputs = super().__call__(data)
        if "action_loss_weight" in data:
            from openpi.models.model import Observation

            if "action_loss_weight" not in Observation.__dataclass_fields__:
                raise RuntimeError("weighted return-policy training requires OpenPI patch 0005")
            weight = np.asarray(data["action_loss_weight"], dtype=np.float32)
            if weight.shape != () or not np.isfinite(weight) or not 0 <= weight <= 20:
                raise ValueError("D20 importance weight must be a finite scalar in [0,20]")
            inputs["action_loss_weight"] = weight
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
class KnownDelayOracleInputs(ReturnBeliefInputs):
    """Privileged lane: exact known D20 delay, never the main actor input schema.

    Offline this is the supervision delay. Online it must be the actual delay
    supplied by the privileged evaluation harness, never a hypothetical draw.
    """

    require_native_anchors: ClassVar[bool] = False

    def _read_belief(self, data: dict) -> dict:
        oracle = data.get("known_delay_oracle")
        if not isinstance(oracle, dict) or set(oracle) != {
            "visual",
            "proprio",
            "known_delay_ticks",
        }:
            raise ValueError("known-delay oracle fields are invalid")
        delay = oracle["known_delay_ticks"]
        if (
            np.asarray(delay).shape != ()
            or not np.issubdtype(np.asarray(delay).dtype, np.integer)
            or not 1 <= int(delay) <= 20
        ):
            raise ValueError("known oracle delay must be an integer on the original D20 grid")
        visual, proprio = np.asarray(oracle["visual"]), np.asarray(oracle["proprio"])
        if visual.shape != (2, 196, 384) or proprio.shape != (16,):
            raise ValueError("known-delay oracle future state shape is invalid")
        return {
            "visual": np.broadcast_to(visual, (5, *visual.shape)),
            "proprio": np.broadcast_to(proprio, (5, 16)),
            "delay_ticks": np.full(5, int(delay), dtype=np.float32),
            "probabilities": np.array([1, 0, 0, 0, 0], dtype=np.float32),
        }


@dataclasses.dataclass(frozen=True)
class ReturnBeliefDataConfig(StructuredPolicyDataConfig):
    """Transforms for a provider that attaches a return mixture to each policy sample.

    The existing clean LeRobot dataset alone does not contain these fields. A
    matched predicted/GT/oracle provider must supply them; missing Belief fails.
    This factory does not select labels, sample a realized delay, or train JEPA.
    """

    return_policy_view: dict[str, Any] | None = None
    input_type: ClassVar[type] = ReturnBeliefInputs
    envelope: ClassVar[str] = "return_belief"
    belief_fields: ClassVar[tuple[str, ...]] = ("visual", "proprio", "delay_ticks", "probabilities")

    def create(self, assets_dirs, model_config):
        if not getattr(model_config, "use_return_belief", False):
            raise ValueError("return Belief data requires a conditioned policy model")
        config = super().create(assets_dirs, model_config)
        if config.norm_stats is None or "state" not in config.norm_stats:
            raise ValueError(
                "conditioned data requires the matched clean policy's state statistics"
            )
        structure = dict(config.repack_transforms.inputs[0].structure)
        structure[self.envelope] = {key: f"{self.envelope}/{key}" for key in self.belief_fields}
        view_kwargs = {}
        if self.return_policy_view is not None:
            oracle = self.return_policy_view.get("mode") == "known_delay_oracle"
            if oracle != (self.envelope == "known_delay_oracle"):
                raise ValueError("return-policy view and privileged input route disagree")
            structure["action_loss_weight"] = "action_loss_weight"
            view_kwargs["return_policy_view"] = self.return_policy_view
        return dataclasses.replace(
            config,
            repack_transforms=transforms.Group(inputs=[transforms.RepackTransform(structure)]),
            data_transforms=transforms.Group(
                inputs=[
                    self.input_type(
                        model_type=model_config.model_type,
                        state_norm_stats=config.norm_stats["state"],
                        use_quantiles=config.use_quantile_norm,
                    )
                ],
                outputs=config.data_transforms.outputs,
            ),
            **view_kwargs,
        )


@dataclasses.dataclass(frozen=True)
class KnownDelayOracleDataConfig(ReturnBeliefDataConfig):
    input_type: ClassVar[type] = KnownDelayOracleInputs
    envelope: ClassVar[str] = "known_delay_oracle"
    belief_fields: ClassVar[tuple[str, ...]] = ("visual", "proprio", "known_delay_ticks")
