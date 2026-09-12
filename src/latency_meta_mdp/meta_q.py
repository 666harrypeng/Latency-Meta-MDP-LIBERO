"""Small spatial-feature Q model for the frozen-policy Launch/Wait semi-MDP."""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from latency_meta_mdp.meta_replay import FEATURE_FORMAT, meta_features


def validate_policy_binding(config, identity):
    for key, value in config["policy_binding"].items():
        if identity.get(key) != value:
            raise ValueError(f"Meta checkpoint policy binding differs: {key}")


class MetaQNetwork(nn.Module):
    def __init__(self, *, vector_mean, vector_scale, use_future=True):
        super().__init__()
        self.use_future = use_future
        self.register_buffer("vector_mean", torch.as_tensor(vector_mean).float().clone())
        self.register_buffer("vector_scale", torch.as_tensor(vector_scale).float().clone())
        if self.vector_mean.shape != (501,) or self.vector_scale.shape != (501,):
            raise ValueError("Meta normalization requires 501 public scalar features")
        self.visual_norm = nn.LayerNorm(384, elementwise_affine=False)
        self.visual_projection = nn.Linear(384, 16)
        self.head = nn.Sequential(
            nn.Linear(4 * 196 * 16 + 501, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
        )

    def forward(self, visual, vector):
        visual = visual.float()
        vector = (vector.float() - self.vector_mean) / self.vector_scale
        if not self.use_future:
            visual = visual.clone()
            visual[:, 1] = 0
            vector = vector.clone()
            vector[:, 16:32] = 0
        encoded = self.visual_projection(self.visual_norm(visual)).flatten(1)
        return self.head(torch.cat((encoded, vector), dim=1))


def fitted_q_targets(
    *, reward, action, discount, next_online, next_target, next_legal, call_cost, forecast_cost
):
    """Double-Q target; discount is already gamma**physical_duration (zero at terminal)."""
    if call_cost < 0 or forecast_cost < 0:
        raise ValueError("Meta compute costs must be nonnegative")
    bootstrap = torch.zeros_like(reward)
    active = discount > 0
    if active.any():
        if not next_legal[active].any(dim=1).all():
            raise ValueError("nonterminal replay has no admissible next action")
        choices = next_online[active].masked_fill(~next_legal[active], -torch.inf).argmax(dim=1)
        bootstrap[active] = next_target[active].gather(1, choices[:, None]).squeeze(1).float()
    target = reward - call_cost * action.float() - forecast_cost + discount * bootstrap
    # At most one unit terminal success, nonnegative costs, and gamma<=1 imply V<=1.
    return target.clamp(max=1.0)


class FittedQScheduler:
    def __init__(self, checkpoint_dir: Path, *, device):
        from safetensors.torch import load_file

        self.config = json.loads((checkpoint_dir / "config.json").read_text())
        if self.config["feature_format"] != FEATURE_FORMAT:
            raise ValueError("unsupported Meta feature checkpoint")
        self.device = torch.device(device)
        self.model = MetaQNetwork(
            vector_mean=np.zeros(501),
            vector_scale=np.ones(501),
            use_future=self.config["use_future"],
        )
        self.model.load_state_dict(load_file(checkpoint_dir / "model.safetensors"), strict=True)
        self.model.to(self.device).eval().requires_grad_(False)
        if any(not torch.isfinite(v).all() for v in self.model.state_dict().values()):
            raise ValueError("nonfinite Meta checkpoint")
        if not (self.model.vector_scale > 0).all():
            raise ValueError("Meta normalization scale must be positive")

    def __call__(self, state, observation, belief):
        visual, vector, legal = meta_features(
            {"observation": observation, "buffer": state, "belief": belief}
        )
        with (
            torch.inference_mode(),
            torch.autocast(
                self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"
            ),
        ):
            q = self.model(
                torch.from_numpy(visual[None]).to(self.device),
                torch.from_numpy(vector[None]).to(self.device),
            )[0].float()
            self.last_q_values = q.cpu().tolist()
            q = q.masked_fill(~torch.from_numpy(legal).to(self.device), -torch.inf)
        return bool(q.argmax().item() == 1)
