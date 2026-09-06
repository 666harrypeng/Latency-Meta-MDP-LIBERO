"""Current16 structured policy inputs for the pinned OpenPI data pipeline."""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import numpy as np
from openpi import transforms
from openpi.models.model import Observation
from openpi.policies.libero_policy import LiberoInputs
from openpi.training.config import LeRobotLiberoDataConfig


@dataclasses.dataclass(frozen=True)
class StructuredPolicyInputs(LiberoInputs):
    allow_empty_action_targets: ClassVar[bool] = False

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"])
        if state.shape != (16,) or not np.isfinite(state).all():
            raise ValueError("structured policy requires finite current16 state")
        inputs = super().__call__(data)
        if "actions" in inputs:
            actions = np.asarray(inputs["actions"])
            is_pad = np.asarray(data["actions_is_pad"])
            if actions.shape != (50, 7) or is_pad.shape != (50,) or is_pad.dtype != np.bool_:
                raise ValueError(
                    "structured policy requires H50/7D actions and boolean episode mask"
                )
            empty_allowed = self.allow_empty_action_targets and is_pad.all()
            if (is_pad[0] and not empty_allowed) or np.any(np.diff(is_pad.astype(np.int8)) < 0):
                raise ValueError(
                    "each source must start at a real action with a contiguous valid prefix"
                )
            mask = np.broadcast_to(~is_pad[:, None], actions.shape).copy()
            if not np.isfinite(actions[mask]).all():
                raise ValueError("real action targets must be finite")
            inputs["actions"] = np.where(mask, actions, 0.0)
            inputs["action_loss_mask"] = mask
        return inputs


@dataclasses.dataclass(frozen=True)
class StructuredPolicyDataConfig(LeRobotLiberoDataConfig):
    def create(self, assets_dirs, model_config):
        if "action_loss_mask" not in Observation.__dataclass_fields__:
            raise RuntimeError("structured training requires the OpenPI masked-action loss patch")
        if not model_config.pi05 or not model_config.discrete_state_input:
            raise ValueError("structured policy requires pi0.5 state token inputs")
        config = super().create(assets_dirs, model_config)
        if config.drop_n_last_frames != 0 or self.extra_delta_transform:
            raise ValueError("structured policy requires all sources and controller-native actions")
        return dataclasses.replace(
            config,
            repack_transforms=transforms.Group(
                inputs=[
                    transforms.RepackTransform(
                        {
                            "observation/image": "image",
                            "observation/wrist_image": "wrist_image",
                            "observation/state": "state",
                            "actions": "actions",
                            "actions_is_pad": "actions_is_pad",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            data_transforms=transforms.Group(
                inputs=[StructuredPolicyInputs(model_type=model_config.model_type)],
                outputs=config.data_transforms.outputs,
            ),
        )
